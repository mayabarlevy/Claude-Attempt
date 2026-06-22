"""Leave-proteins-out cross-validation training for the binding predictor.

For each fold: train on all (DNA, protein) pairs whose protein is in the
training group, validate on pairs whose protein is held out (never seen during
training). Early stopping tracks the mean per-held-out-protein Pearson r.
Best-epoch validation predictions are cached for evaluate.py.

Usage:
    python train.py                       # full CV from config.yaml
    python train.py --folds 1 --epochs 2  # quick smoke test
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    PairDataset,
    TargetTransform,
    build_long_format,
    load_data,
    protein_group_folds,
    reverse_complement_onehot,
)
from metrics import per_protein_correlations
from model import build_model
from utils import load_config, pick_device


def pairs_for_proteins(dna_idx, prot_idx, aff, prot_ids: np.ndarray):
    """Subset the long-format arrays to rows whose protein is in prot_ids."""
    mask = np.isin(prot_idx, prot_ids)
    return dna_idx[mask], prot_idx[mask], aff[mask]


def _pair_predict(model, d_emb, p_emb):
    """Predict the full probe x protein grid from precomputed tower outputs.

    d_emb (one row/block per probe), p_emb (one row/token-set per protein).
    Interaction heads that expose `score_grid` (e.g. cross_attn) compute the grid
    directly in a batched einsum -- no [Bn*P, ...] materialisation. Otherwise we
    fall back to expanding to the Bn*P pair grid (probe-major: pair i -> probe
    i//P, protein i%P) and running the interaction head once. Returns [Bn, P].
    """
    inter = model.interaction
    if hasattr(inter, "score_grid"):
        return inter.score_grid(p_emb, d_emb)  # p [P,K,Dp], d [Bn,L,Dd] -> [Bn,P]
    Bn, P = d_emb.shape[0], p_emb.shape[0]
    d_rep = d_emb.repeat_interleave(P, dim=0)  # [Bn*P, Dd]
    p_rep = p_emb.repeat(Bn, 1)                # [Bn*P, Dp]
    return inter(p_rep, d_rep).view(Bn, P)


def _encode_proteins(model, esm_emb, esm_mask, prot_t, noise: float = 0.0):
    """Run the protein tower on the given protein ids. Passes a residue mask when
    the cache is per-residue ([P, R, D]); optionally adds Gaussian noise to the
    (frozen) features as a regulariser against protein-tower overfitting (320 TFs)."""
    x = esm_emb[prot_t]
    if noise > 0:
        x = x + noise * torch.randn_like(x)
    if esm_mask is not None:
        return model.protein_encoder(x, esm_mask[prot_t])
    return model.protein_encoder(x)


def pearson_corr_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - mean per-protein Pearson r, correlating over the probe block (rows) for
    each protein (column). A stochastic estimate of the eval metric, so optimising
    it directly aligns training with leave-proteins-out per-protein correlation."""
    p = pred - pred.mean(dim=0, keepdim=True)
    t = tgt - tgt.mean(dim=0, keepdim=True)
    num = (p * t).sum(dim=0)
    den = torch.sqrt((p * p).sum(dim=0) * (t * t).sum(dim=0) + eps)
    return 1.0 - (num / (den + eps)).mean()


def train_epoch_blocked(model, dna_onehot, esm_emb, probe_ids, prot_ids, tmat,
                        device, rc_prob, block, optimizer, loss_fn,
                        esm_mask=None, esm_noise=0.0, corr_lambda=0.0):
    """One training pass in outer-product form.

    Each step samples a block of `block` unique probes, encodes them once with the
    DNA tower, encodes all `prot_ids` proteins once with the protein tower, then
    scores the full block x protein grid through the interaction head. `tmat` is the
    [N, P] per-protein z-scored target matrix (rows = probe id, cols = position in
    prot_ids). The loss blends the elementwise `loss_fn` (Huber) with a per-protein
    correlation term weighted by `corr_lambda` (aligned to the eval metric).
    """
    model.train(True)
    prot_t = torch.as_tensor(prot_ids, device=device)
    perm = torch.randperm(len(probe_ids))
    total, n = 0.0, 0
    for s in range(0, len(perm), block):
        pb = perm[s : s + block]  # torch LongTensor of probe ids (probe_ids is arange)
        dna = dna_onehot[pb.to(device)].clone()
        if rc_prob > 0:  # per-probe reverse-complement augmentation
            flip = torch.rand(dna.shape[0], device=device) < rc_prob
            if flip.any():
                dna[flip] = reverse_complement_onehot(dna[flip])
        tgt = tmat[pb].to(device)  # [Bn, P]

        d_emb = model.dna_encoder(dna)
        p_emb = _encode_proteins(model, esm_emb, esm_mask, prot_t, noise=esm_noise)
        pred = _pair_predict(model, d_emb, p_emb)
        loss = loss_fn(pred, tgt)
        if corr_lambda > 0:
            loss = loss + corr_lambda * pearson_corr_loss(pred, tgt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = pred.numel()
        total += loss.item() * bs
        n += bs
    return total / n


@torch.no_grad()
def evaluate_blocked(model, dna_onehot, esm_emb, prot_ids, aff_raw, device,
                     block=4096, tta_rc=False, esm_mask=None):
    """Score every probe x protein pair for `prot_ids` and return per-protein
    correlations against the raw affinity matrix `aff_raw` [N, P_all].

    Encodes all proteins once, then streams probe blocks through the DNA tower.
    Correlation is invariant to the target transform, so we compare model output
    directly to raw affinity -- no inverse transform needed.
    """
    model.train(False)
    prot_t = torch.as_tensor(prot_ids, device=device)
    p_emb = _encode_proteins(model, esm_emb, esm_mask, prot_t)  # [P, Dp] or [P, K, Dp]
    N = dna_onehot.shape[0]
    preds = np.empty((N, len(prot_ids)), dtype=np.float32)
    for s in range(0, N, block):
        idx = torch.arange(s, min(s + block, N), device=device)
        dna = dna_onehot[idx]
        d_emb = model.dna_encoder(dna)
        pred = _pair_predict(model, d_emb, p_emb)
        if tta_rc:
            d_rc = model.dna_encoder(reverse_complement_onehot(dna))
            pred = 0.5 * (pred + _pair_predict(model, d_rc, p_emb))
        preds[s : s + dna.shape[0]] = pred.cpu().numpy()

    # flatten to long form for the metric (and for saving)
    y_true = aff_raw[:, prot_ids].T.reshape(-1)        # protein-major
    y_pred = preds.T.reshape(-1)
    prot_idx = np.repeat(prot_ids, N)
    dna_idx = np.tile(np.arange(N), len(prot_ids))
    return prot_idx, dna_idx, y_true, y_pred

def _zscore_np(x: np.ndarray) -> np.ndarray:
    """Z-score one prediction/profile vector.

    Pearson is invariant to affine scaling, but blending two sources works better
    when the model profile and nearest-neighbor profile are placed on the same
    scale first.
    """
    x = x.astype(np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def _pooled_protein_embeddings_np(esm_emb: torch.Tensor, esm_mask=None) -> np.ndarray:
    """Convert per-residue protein representations to one vector per protein.

    The neural model still uses the full per-residue cache. This pooled view is
    used only for the nearest-neighbor search in ESM/physchem embedding space.
    """
    with torch.no_grad():
        if esm_emb.ndim == 3:
            if esm_mask is not None:
                mask = esm_mask.to(dtype=esm_emb.dtype).unsqueeze(-1)  # [P, R, 1]
                pooled = (esm_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                pooled = esm_emb.mean(dim=1)
        else:
            pooled = esm_emb

        pooled = pooled.detach().cpu().float().numpy()

    pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8)
    return pooled


def apply_knn_blend_predictions(
    prot_idx: np.ndarray,
    dna_idx: np.ndarray,
    y_pred: np.ndarray,
    aff_raw: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    protein_sim_emb: np.ndarray,
    alpha_model: float,
) -> np.ndarray:
    """Blend model predictions with the nearest training protein affinity profile.

    For each held-out validation protein:
      1. Find nearest training protein by cosine similarity in pooled ESM space.
      2. Take that training protein's true affinity profile over the same DNA probes.
      3. Z-score both profiles.
      4. Blend them with a fixed global alpha.

    This is a prediction-time prior, not a train-time loss term.
    """
    train_ids = np.asarray(train_ids)
    y_blend = np.empty_like(y_pred, dtype=np.float32)

    for pid in val_ids:
        m = prot_idx == pid

        sims = protein_sim_emb[pid] @ protein_sim_emb[train_ids].T
        nearest_pid = int(train_ids[int(np.argmax(sims))])

        model_profile = y_pred[m]
        nearest_profile = aff_raw[dna_idx[m], nearest_pid]

        y_blend[m] = (
            alpha_model * _zscore_np(model_profile)
            + (1.0 - alpha_model) * _zscore_np(nearest_profile)
        ).astype(np.float32)

    return y_blend

def build_scheduler(optimizer, cfg):
    """Returns (scheduler, kind). kind='cosine' steps once per epoch with no args
    (linear warmup -> cosine decay over cfg['epochs']); kind='plateau' steps on
    the val metric. Cosine is the default: val Pearson here keeps climbing for
    many epochs, so a fixed warmup+decay horizon trains longer and more stably
    than decaying the moment the metric stalls."""
    kind = cfg.get("scheduler", "cosine")
    if kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return sched, "plateau"
    epochs = cfg["epochs"]
    warmup = cfg.get("warmup_epochs", 5)
    min_lr_frac = cfg.get("min_lr_frac", 0.02)  # floor so late epochs still learn

    def lr_lambda(e):  # e is 0-indexed epoch count
        if e < warmup:
            return (e + 1) / max(1, warmup)
        prog = (e - warmup) / max(1, epochs - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "cosine"


def build_target_matrix(tfm: TargetTransform, aff: np.ndarray, prot_ids: np.ndarray) -> torch.Tensor:
    """[N, len(prot_ids)] per-protein-transformed target matrix (col k = prot_ids[k]).

    Each column is the protein's affinity profile pushed through the fitted
    TargetTransform (per-protein z-score by default). Held-out proteins absent
    from the fit fall back to the global train stats -- fine, since the selection
    metric is correlation (affine-invariant); these targets only feed the loss.
    """
    N = aff.shape[0]
    cols = [tfm.transform(np.full(N, pid), aff[:, pid]) for pid in prot_ids]
    return torch.from_numpy(np.stack(cols, axis=1).astype(np.float32))


def train_one_fold(fold, train_ids, val_ids, data, dna_onehot, esm_emb, cfg, device, out_dir,
                   esm_mask=None):
    aff = data.affinity  # [N, P] raw
    N = data.n_dna
    train_ids = np.asarray(train_ids)
    val_ids = np.asarray(val_ids)

    # Fit the target transform on TRAIN pairs only (all probes x train proteins).
    tcfg = cfg["target"]
    fit_pidx = np.repeat(train_ids, N)
    fit_vals = aff[:, train_ids].T.reshape(-1)
    tfm = TargetTransform(
        mode=tcfg["mode"], log1p=tcfg["log1p"], clip_quantile=tcfg.get("clip_quantile")
    ).fit(fit_pidx, fit_vals)
    tmat_train = build_target_matrix(tfm, aff, train_ids)  # [N, n_train]

    probe_ids = np.arange(N)
    block = cfg.get("probe_block", 512)

    seq_len = data.dna_onehot.shape[2]
    model = build_model(cfg, esm_dim=esm_emb.shape[-1], seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler, sched_kind = build_scheduler(optimizer, cfg)
    lcfg = cfg.get("loss", {})
    loss_fn = nn.HuberLoss(delta=lcfg.get("huber_delta", 1.0))
    corr_lambda = lcfg.get("corr_lambda", 0.0)
    esm_noise = cfg.get("esm_noise", 0.0)

    best_pearson = -np.inf
    best_state = None
    best_preds = None
    patience = cfg["early_stop_patience"]
    bad_epochs = 0

    knn_cfg = cfg.get("knn_blend", {})
    knn_blend_enabled = bool(knn_cfg.get("enabled", False))
    knn_alpha = float(knn_cfg.get("alpha_model", 1.0))

    if knn_blend_enabled:
        protein_sim_emb = _pooled_protein_embeddings_np(esm_emb, esm_mask=esm_mask)
    else:
        protein_sim_emb = None

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch_blocked(
            model, dna_onehot, esm_emb, probe_ids, train_ids, tmat_train,
            device, cfg["rc_prob"], block, optimizer, loss_fn,
            esm_mask=esm_mask, esm_noise=esm_noise, corr_lambda=corr_lambda,
        )
        vpidx, vdidx, vtrue, vpred = evaluate_blocked(
            model, dna_onehot, esm_emb, val_ids, aff, device,
            tta_rc=cfg.get("tta_rc", False), esm_mask=esm_mask,
        )
        corr_model = per_protein_correlations(vpidx, vtrue, vpred)
        pe_model, sp_model = corr_model["pearson_mean"], corr_model["spearman_mean"]

        vpred_for_metric = vpred

        if knn_blend_enabled:
            vpred_blend = apply_knn_blend_predictions(
                prot_idx=vpidx,
                dna_idx=vdidx,
                y_pred=vpred,
                aff_raw=aff,
                train_ids=train_ids,
                val_ids=val_ids,
                protein_sim_emb=protein_sim_emb,
                alpha_model=knn_alpha,
            )
            corr_blend = per_protein_correlations(vpidx, vtrue, vpred_blend)
            pe, sp = corr_blend["pearson_mean"], corr_blend["spearman_mean"]
            vpred_for_metric = vpred_blend
        else:
            pe, sp = pe_model, sp_model

        scheduler.step(pe) if sched_kind == "plateau" else scheduler.step()

        if knn_blend_enabled:
            print(
                f"  fold {fold} epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
                f"val_pearson_model={pe_model:.4f}  val_pearson_blend={pe:.4f}  "
                f"val_spearman_model={sp_model:.4f}  val_spearman_blend={sp:.4f}"
            )
        else:
            print(
                f"  fold {fold} epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
                f"val_pearson={pe:.4f}  val_spearman={sp:.4f}"
            )

        if pe > best_pearson:
            best_pearson = pe
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = {
                "prot_idx": vpidx,
                "dna_idx": vdidx,
                "y_true": vtrue,              # raw affinity
                "y_pred": vpred_for_metric,   # final prediction used for validation metric
                "y_pred_model": vpred,        # raw neural model prediction, kept for debugging
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  fold {fold}: early stop at epoch {epoch}")
                break

    # persist best checkpoint + predictions for this fold
    torch.save(best_state, os.path.join(out_dir, f"model_fold{fold}.pt"))
    np.savez(os.path.join(out_dir, f"preds_fold{fold}.npz"), **best_preds)
    print(f"  fold {fold} BEST val_pearson={best_pearson:.4f}")
    return best_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds", type=int, default=None, help="limit #folds (smoke test)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (smoke test)")
    ap.add_argument("--out-dir", default="runs", help="dir for checkpoints/preds (per-experiment)")
    ap.add_argument("--esm-cache", default=None, help="override cfg['esm_cache'] (A/B different protein towers)")
    ap.add_argument("--seed", type=int, default=None, help="override cfg['seed'] (seed-ensembling)")
    ap.add_argument("--probe-subsample", type=int, default=None,
                    help="train/eval on a random subset of N probes (fast, fair A/B smoke)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(here, args.config))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.esm_cache is not None:
        cfg["esm_cache"] = args.esm_cache
    if args.seed is not None:
        cfg["seed"] = args.seed

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device()

    data = load_data(
        os.path.join(here, cfg["dbp_path"]),
        os.path.join(here, cfg["seq_path"]),
        os.path.join(here, cfg["data_path"]),
    )
    if args.probe_subsample is not None and args.probe_subsample < data.n_dna:
        # Deterministic probe subset (seeded) so an A/B over esm_cache sees the
        # exact same probes -- only the protein tower's input changes.
        rng = np.random.default_rng(cfg["seed"])
        keep = np.sort(rng.choice(data.n_dna, args.probe_subsample, replace=False))
        data.dna_seqs = [data.dna_seqs[i] for i in keep]
        data.affinity = data.affinity[keep]
        data.dna_onehot = data.dna_onehot[keep]
        print(f"probe subsample: {len(keep)}/{args.probe_subsample} probes (seed {cfg['seed']})")
    ckpt = torch.load(os.path.join(here, cfg["esm_cache"]), map_location="cpu")
    embedder = ckpt.get("embedder", ckpt.get("model", "?"))
    per_residue = ckpt.get("per_residue", False)
    esm_emb = ckpt["embeddings"].float().to(device)  # [P, D] or [P, R, D]
    esm_mask = None
    if per_residue:  # build [P, R] validity mask from true lengths (ignore padding)
        R = esm_emb.shape[1]
        ar = torch.arange(R, device=device)
        esm_mask = ar[None, :] < ckpt["lengths"].to(device)[:, None]  # [P, R] bool
    if esm_emb.shape[0] != data.n_prot:
        raise ValueError(
            f"protein-embedding cache {cfg['esm_cache']!r} has {esm_emb.shape[0]} "
            f"rows but the data has {data.n_prot} proteins. The cache is stale for "
            f"the current {cfg['dbp_path']!r}; rebuild it with:\n"
            f"    python embed_proteins.py --embedder {embedder} --dbp {cfg['dbp_path']}"
        )
    # Both lookup tables are device-resident so the per-step cost is the towers,
    # not host->device transfer (per-residue reps are ~1.7 GB for 650M).
    dna_onehot = torch.from_numpy(data.dna_onehot).to(device)
    print(f"device={device}  protein_emb={tuple(esm_emb.shape)} ({embedder}"
          f"{', per-residue' if per_residue else ''})  pairs={data.n_dna*data.n_prot:,}")

    out_dir = os.path.join(here, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    folds = list(protein_group_folds(data.n_prot, n_splits=cfg["n_splits"], seed=cfg["seed"]))
    if args.folds is not None:
        folds = folds[: args.folds]

    fold_means = []
    for fold, (train_ids, val_ids) in enumerate(folds):
        print(f"\n=== Fold {fold}: train {len(train_ids)} proteins, hold out {len(val_ids)} ===")
        preds = train_one_fold(fold, train_ids, val_ids, data, dna_onehot, esm_emb, cfg, device, out_dir,
                               esm_mask=esm_mask)
        corr = per_protein_correlations(preds["prot_idx"], preds["y_true"], preds["y_pred"])
        fold_means.append((corr["pearson_mean"], corr["spearman_mean"]))

    fm = np.array(fold_means)
    print("\n=== Cross-validation summary (leave-proteins-out) ===")
    print(f"Pearson : {fm[:,0].mean():.4f} +/- {fm[:,0].std():.4f}")
    print(f"Spearman: {fm[:,1].mean():.4f} +/- {fm[:,1].std():.4f}")
    print(f"Per-fold predictions saved in {out_dir}/preds_fold*.npz")


if __name__ == "__main__":
    main()
