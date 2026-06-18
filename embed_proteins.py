"""Precompute and cache fixed-size protein embeddings (the protein "encoder").

Protein embeddings are computed ONCE and cached to `cache/<name>.pt` as a
`[P, D]` tensor, then loaded by train.py. This decouples the (heavy, frozen)
protein representation from training, and makes the protein encoder swappable:
build a different cache and point `esm_cache` in config.yaml at it.

Registered embedders (`--embedder`):
  esm2_t36_3B_UR50D     frozen ESM-2 3B,   mean-pooled  (2560-d, best quality)
  esm2_t33_650M_UR50D   frozen ESM-2 650M, mean-pooled  (1280-d, good quality)
  esm2_t30_150M_UR50D   frozen ESM-2 150M, mean-pooled  (640-d, good MPS default)
  esm2_t12_35M_UR50D    frozen ESM-2 35M, mean-pooled   (480-d, fastest)
  kmer3                 3-mer amino-acid composition    (8000-d, no model; baseline)

Add a new one with @register_embedder("name"); it must return a [P, D] tensor.

Two cache layouts:
  - pooled (default): {"embeddings": [P, D]} -- one vector per protein.
  - per-residue (`--per-residue`): {"embeddings": [P, R, D] fp16, "lengths": [P],
    "per_residue": True} -- padded residue reps + true lengths, for the
    attn-pool-tokens protein encoder / cross-attention head.

Run once, e.g.:
    python embed_proteins.py --embedder esm2_t30_150M_UR50D
    python embed_proteins.py --embedder esm2_t33_650M_UR50D --per-residue
    python embed_proteins.py --embedder kmer3            # cheap protein-encoder baseline
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch

from data import read_lines
from utils import pick_device

EMBEDDERS: dict[str, callable] = {}


def register_embedder(name):
    def deco(fn):
        EMBEDDERS[name] = fn
        return fn
    return deco


# --- ESM-2 family (frozen, mean-pooled over residues) ----------------------
_ESM_LOADERS = {
    "esm2_t36_3B_UR50D": 36,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t12_35M_UR50D": 12,
}

# --- Amino-acid physicochemical features -----------------------------------
# Per-residue features appended to ESM vectors when --physchem is used.
# Features:
#   charge, hydrophobicity, polarity, aromatic, positive, negative
_AA_HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    "X": 0.0,
}

_AA_POLARITY = {
    "A": 0.0, "C": 0.0, "D": 1.0, "E": 1.0, "F": 0.0,
    "G": 0.0, "H": 1.0, "I": 0.0, "K": 1.0, "L": 0.0,
    "M": 0.0, "N": 1.0, "P": 0.0, "Q": 1.0, "R": 1.0,
    "S": 1.0, "T": 1.0, "V": 0.0, "W": 0.0, "Y": 1.0,
    "X": 0.0,
}

_AA_AROMATIC = set("FWYH")
_AA_POSITIVE = set("KRH")
_AA_NEGATIVE = set("DE")


def aa_physchem_features(seq: str, v2: bool = False) -> torch.Tensor:
    """Return per-residue physicochemical features.

    v1 features [L, 6]:
      charge, hydrophobicity, polarity, aromatic, positive, negative.

    v2 features [L, 10]:
      v1 + molecular_size, small_residue, glycine, proline.

    These are explicit biological signals that may help the protein tower attend
    to DNA-binding residues and structurally important residues.
    """
    # Approximate molecular weights, normalized by 204.23 (tryptophan).
    aa_weight = {
        "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
        "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
        "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
        "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
        "X": 120.00,
    }

    small_residues = set("AGSCT")

    feats = []
    for aa in seq:
        aa = aa.upper()
        positive = 1.0 if aa in _AA_POSITIVE else 0.0
        negative = 1.0 if aa in _AA_NEGATIVE else 0.0

        # Treat H as weakly positive; K/R as strongly positive.
        if aa in {"K", "R"}:
            charge = 1.0
        elif aa == "H":
            charge = 0.5
        elif aa in _AA_NEGATIVE:
            charge = -1.0
        else:
            charge = 0.0

        # Kyte-Doolittle hydropathy roughly ranges [-4.5, 4.5].
        hydrophobicity = _AA_HYDROPATHY.get(aa, 0.0) / 4.5

        polarity = _AA_POLARITY.get(aa, 0.0)
        aromatic = 1.0 if aa in _AA_AROMATIC else 0.0

        row = [
            charge,
            hydrophobicity,
            polarity,
            aromatic,
            positive,
            negative,
        ]

        if v2:
            molecular_size = aa_weight.get(aa, 120.0) / 204.23
            small_residue = 1.0 if aa in small_residues else 0.0
            glycine = 1.0 if aa == "G" else 0.0
            proline = 1.0 if aa == "P" else 0.0

            row.extend([
                molecular_size,
                small_residue,
                glycine,
                proline,
            ])

        feats.append(row)

    return torch.tensor(feats, dtype=torch.float32)

def _pool_residues(reps_seq: torch.Tensor, pool: str) -> torch.Tensor:
    """Pool per-residue reps [L, D] -> fixed vector. `meanmax`/`meanmaxstd`
    concatenate complementary statistics: mean (overall composition), max (the
    strongest residue signal, where DNA-binding-domain motifs tend to show up),
    and std (residue-level variability). Richer than mean alone for capturing
    which residues drive specificity."""
    if pool == "mean":
        return reps_seq.mean(dim=0)
    if pool == "meanmax":
        return torch.cat([reps_seq.mean(dim=0), reps_seq.amax(dim=0)])
    if pool == "meanmaxstd":
        return torch.cat([reps_seq.mean(dim=0), reps_seq.amax(dim=0), reps_seq.std(dim=0)])
    raise ValueError(f"unknown pool {pool!r}")


def _make_esm_embedder(model_name: str, repr_layer: int):
    @torch.no_grad()
    def _embed(proteins: list[str], device: torch.device, batch_size: int = 4, 
               pool: str = "mean", per_residue: bool = False, 
               physchem: bool = False):    
        """Pooled mode -> [P, D] tensor. Per-residue mode -> (padded [P, R, D]
        fp16, lengths [P] int) where R = max protein length and rows past each
        length are zero (a mask is rebuilt from lengths at train time)."""
        import esm

        loader = getattr(esm.pretrained, model_name)
        model, alphabet = loader()
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        embs: list[torch.Tensor] = []          # pooled: [D]; per-residue: [L_i, D]
        for start in range(0, len(proteins), batch_size):
            chunk = proteins[start : start + batch_size]
            labeled = [(str(start + k), s) for k, s in enumerate(chunk)]
            _, _, tokens = batch_converter(labeled)
            out = model(tokens.to(device), repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]
            for k, seq in enumerate(chunk):
                res = reps[k, 1 : len(seq) + 1].float().cpu()  # [L_i, D], drop BOS/pad

                if physchem:
                    phys = aa_physchem_features(seq)           # [L_i, 6]
                    res = torch.cat([res, phys], dim=-1)       # [L_i, D + 6]

                embs.append(res.half() if per_residue else _pool_residues(res, pool))
            print(f"  embedded {min(start + batch_size, len(proteins))}/{len(proteins)}")
        if not per_residue:
            return torch.stack(embs, dim=0)
        lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
        R, D = int(lengths.max()), embs[0].shape[1]
        padded = torch.zeros(len(embs), R, D, dtype=torch.float16)
        for i, e in enumerate(embs):
            padded[i, : e.shape[0]] = e
        return padded, lengths

    return _embed


for _name, _layer in _ESM_LOADERS.items():
    register_embedder(_name)(_make_esm_embedder(_name, _layer))


# --- k-mer composition baseline (no model; demonstrates encoder swapping) --
@register_embedder("kmer3")
def _kmer3(proteins: list[str], device: torch.device, batch_size: int = 4, **kwargs) -> torch.Tensor:
    """Normalized 3-mer amino-acid composition; a cheap protein-encoder baseline."""
    aas = "ACDEFGHIKLMNPQRSTVWY"
    kmers = ["".join(t) for t in itertools.product(aas, repeat=3)]
    index = {km: i for i, km in enumerate(kmers)}
    out = torch.zeros(len(proteins), len(kmers))
    for i, seq in enumerate(proteins):
        for j in range(len(seq) - 2):
            idx = index.get(seq[j : j + 3])
            if idx is not None:
                out[i, idx] += 1.0
        total = out[i].sum()
        if total > 0:
            out[i] /= total
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="esm2_t30_150M_UR50D", choices=list(EMBEDDERS))
    ap.add_argument("--dbp", default="training_DBPs_small.txt")
    ap.add_argument("--out", default=None, help="output .pt path (auto if omitted)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pool", default="mean", choices=["mean", "meanmax", "meanmaxstd"],
                    help="pooled-mode residue aggregation (ignored with --per-residue)")
    ap.add_argument("--per-residue", action="store_true",
                    help="cache padded per-residue reps + lengths (for cross-attention)")
    ap.add_argument("--physchem", action="store_true",
                    help="append 6 amino-acid physicochemical features per residue")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    proteins = read_lines(os.path.join(here, args.dbp))
    device = pick_device()
    print(f"device={device}  embedder={args.embedder}  n_proteins={len(proteins)}"
          f"  per_residue={args.per_residue}")

    kwargs = (
        {}
        if args.embedder == "kmer3"
        else {
            "pool": args.pool,
            "per_residue": args.per_residue,
            "physchem": args.physchem,
        }
    )
    result = EMBEDDERS[args.embedder](proteins, device, args.batch_size, **kwargs)

    os.makedirs(os.path.join(here, "cache"), exist_ok=True)
    if args.per_residue:
        emb, lengths = result
        assert emb.shape[0] == len(proteins) == lengths.shape[0]
        assert torch.isfinite(emb.float()).all(), "embeddings contain NaN/Inf"
        suffix = "_physchem_perres" if args.physchem else "_perres"
        out = args.out or os.path.join(here, "cache", f"{args.embedder}{suffix}.pt")
        torch.save({"embedder": args.embedder, "per_residue": True,
                    "embeddings": emb, "lengths": lengths}, out)
        print(f"saved per-residue embeddings {tuple(emb.shape)} (lengths "
              f"{int(lengths.min())}..{int(lengths.max())}) -> {out}")
    else:
        emb = result
        assert emb.shape[0] == len(proteins)
        assert torch.isfinite(emb).all(), "embeddings contain NaN/Inf"
        out = args.out or os.path.join(here, "cache", f"{args.embedder}.pt")
        torch.save({"embedder": args.embedder, "embeddings": emb}, out)
        print(f"saved embeddings {tuple(emb.shape)} -> {out}")


if __name__ == "__main__":
    main()
