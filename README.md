# Protein–DNA Binding Affinity Predictor

Predicts binding affinity between DNA-binding proteins (transcription factors)
and short DNA probes, designed to **generalize to proteins never seen during
training**.

A two-tower model fuses a learned CNN over one-hot DNA with a **frozen ESM-2**
protein encoder. The default architecture is a **cross-attention "motif-match"
head**: the protein becomes a small set of learned *specificity tokens* that
scan the DNA's per-position feature map, modelling binding as a learned motif
search rather than a dot product of two globally pooled vectors. Generalization
is measured with **leave-proteins-out** cross-validation, and the honest bar is
beating a *k*-nearest-neighbour-in-ESM baseline.

---

## Contents
- [Data](#data)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Pipeline stages](#pipeline-stages)
- [Configuration reference](#configuration-reference)
- [Swapping components (registries)](#swapping-components-registries)
- [Domain extraction](#domain-extraction)
- [Training internals](#training-internals)
- [Evaluation & baselines](#evaluation--baselines)
- [Files](#files)
- [Reading the results](#reading-the-results)

---

## Data

Three text files, one record per line, aligned by index. Counts are read from
the files at load time, not hardcoded.

| File | Shape | Contents |
|---|---|---|
| [training_DBPs_small.txt](training_DBPs_small.txt) | `P = 400` | protein amino-acid sequences |
| [training_seqs_small.txt](training_seqs_small.txt) | `N = 30,000` | DNA probes, 36 bp, alphabet `{A,C,G,T}` |
| `training_data_small.txt` | `N × P = 30,000 × 400` | affinity matrix (space-separated); row *i* = DNA probe *i*, column *j* = protein *j* |

`training_data_small.txt` is git-ignored (it ships as a symlink locally); the two
sequence files are tracked. [data.py](data.py) validates that the three shapes
line up and raises if they don't.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0. inspect the data and sanity-check the data pipeline
python explore_data.py                 # EDA: distribution / skew / per-protein scale
python data.py                         # parsing, folds, reverse-complement, target transform
python homeodomain.py                  # DBD-extraction coverage + self-validation

# 1. cache frozen protein embeddings (run once each; heavy)
python embed_proteins.py --embedder esm2_t33_650M_UR50D --per-residue   # full protein
python embed_domains.py  --embedder esm2_t33_650M_UR50D                 # trimmed DNA-binding domain

# 2. verify every encoder/interaction combination wires up
python model.py

# 3. train + evaluate
python train.py --folds 1 --epochs 2                   # quick smoke test (1 fold, 2 epochs)
python train.py                                        # full leave-proteins-out CV (full protein)
python train.py --config config_homeodomain.yaml --out-dir runs_hd     # domain-trimmed A/B
python evaluate.py                                     # aggregate metrics + kNN / mean baselines
```

The device is selected automatically by [utils.py](utils.py): **Apple MPS →
CUDA → CPU**.

---

## Architecture

```
dna_onehot [B,4,L] ──► dna_encoder      ──► per-position map  [B, L, Dd]
esm reps   [P,R,D] ──► protein_encoder  ──► specificity tokens [B, K, Dp]
                       (cross_attn) tokens scan positions ──► affinity scalar [B]
```

Each of the three slots is chosen by `name` in [config.yaml](config.yaml) and
built from a registry in [encoders.py](encoders.py).
[model.py](model.py)'s `build_model` wires them together by reading each
component's declared `out_dim`, so the towers stay decoupled.

The **default** stack (cross-attention motif-match):

- **DNA tower — `cnn_deep_pos`** ([encoders.py](encoders.py#L200)). A residual
  conv trunk (stem kernel 11, then residual blocks 7→5→3 with BatchNorm/GELU)
  that returns the **per-position** feature map `[B, L, out_dim]` — every
  position is a candidate motif site for the cross-attention head, so the length
  axis is kept rather than pooled away.
- **Protein tower — `attn_pool_tokens`** ([encoders.py](encoders.py#L299)).
  Takes the per-residue ESM reps `[B, R, D]` plus a validity mask and produces
  `K` learned **specificity query tokens** `[B, K, out_dim]` via masked
  multi-query attention pooling. Instead of mean-pooling the whole protein
  (which dilutes the DNA-binding domain), each of the `K` queries pulls out a
  distinct learned summary — a small per-TF set of "motif probes".
- **Interaction — `cross_attn`** ([encoders.py](encoders.py#L384)). Projects the
  `K` protein tokens and `L` DNA positions to a shared space; the match score of
  token *k* at position *l* is their scaled dot product. Each token's scan over
  positions is summarised by `[max, mean, std]` (max = "this motif occurs
  *somewhere*"; mean/std = overall presence/sharpness), embedded by a small MLP,
  pooled over tokens (mean+max, so it is agnostic to `K`), and read out to a
  scalar. It exposes `score_grid(p, d)` to compute the full probe × protein grid
  in one batched einsum for the outer-product trainer.

A simpler **pooled** stack is also available (e.g. `cnn` + `mlp` +
`concat_hadamard`); see [Swapping components](#swapping-components-registries)
for which combinations are compatible.

---

## Pipeline stages

| Stage | Command | What it does |
|---|---|---|
| EDA | `python explore_data.py` | distribution, skew, outliers, per-protein scale spread, duplicate probes |
| Data sanity | `python data.py` | parsing, one-hot, reverse-complement involution, fold disjointness, `TargetTransform` invariants |
| Domain report | `python homeodomain.py` | DBD-extraction coverage and self-validation against the homeodomain anchor |
| Embed (full) | `python embed_proteins.py …` | cache frozen ESM-2 / k-mer protein embeddings (pooled or per-residue) |
| Embed (DBD) | `python embed_domains.py …` | same, but over the trimmed ~60-residue DNA-binding domain |
| Wiring test | `python model.py` | forward/backward over **every** registered encoder × interaction combo |
| Train | `python train.py …` | leave-proteins-out CV loop; saves per-fold checkpoints + predictions |
| Evaluate | `python evaluate.py …` | aggregate metrics + per-protein-mean and kNN-in-ESM baselines |
| Ensemble | `python ensemble.py <dirs…>` | average held-out predictions across seeds; report lift |
| Diagnose | `python analyze_preds.py <npz>` | per-protein Pearson distribution for one fold |

---

## Configuration reference

All hyperparameters and component choices live in [config.yaml](config.yaml)
(full-protein default) and [config_homeodomain.yaml](config_homeodomain.yaml)
(identical except `esm_cache` points at trimmed-domain embeddings — a clean A/B
of "domain extraction beats full sequence"). Pass a different file with
`--config`.

```yaml
# data files
dbp_path / seq_path / data_path   # the three input files

# protein embedding cache (built by embed_proteins.py / embed_domains.py)
esm_cache: cache/esm2_t33_650M_UR50D_perres.pt   # pooled [P,D] OR per-residue [P,R,D]

# target transform (fit on TRAIN proteins only — see explore_data.py)
target:
  mode: per_protein     # per_protein (recommended) | global
  log1p: false          # compress the right skew (skew 0.62 -> ~0)
  clip_quantile: 0.001  # winsorize to [0.1%, 99.9%] train quantiles; null to disable

# model components (each `name` indexes a registry in encoders.py)
dna_encoder:      { name: cnn_deep_pos, channels: 256, out_dim: 128, dropout: 0.2 }
protein_encoder:  { name: attn_pool_tokens, out_dim: 128, n_tokens: 8, hidden: 512, attn_dim: 128, dropout: 0.3 }
interaction:      { name: cross_attn, attn_dim: 64, tok_embed: 32, hidden: 128, dropout: 0.3 }

# training
probe_block: 256       # unique probes encoded per outer-product step
batch_size: 2048       # legacy; UNUSED by the blocked trainer
epochs: 100
lr: 0.0015
weight_decay: 0.0001

# loss = Huber + corr_lambda * (1 - per-protein Pearson)
loss: { huber_delta: 1.0, corr_lambda: 1.0 }
esm_noise: 0.05        # Gaussian noise on frozen residue reps (protein-side regulariser)
scheduler: cosine      # cosine (warmup -> cosine decay) | plateau
warmup_epochs: 5
min_lr_frac: 0.02      # cosine LR floor as a fraction of peak lr
rc_prob: 0.5           # reverse-complement augmentation probability per probe
tta_rc: true           # eval-time: average prediction over a probe + its reverse-complement
early_stop_patience: 20

# cross-validation
n_splits: 5            # leave-proteins-out GroupKFold (400 proteins -> ~80 held out/fold)
seed: 0
```

### Target transform rationale ([explore_data.py](explore_data.py))
Protein mean affinity spans a large range across this dataset, so a **global**
z-score would let strong binders dominate the loss. **`per_protein`** z-scoring
(the default) makes every protein contribute its *specificity pattern* equally —
exactly what the per-protein correlation metric rewards, and the only thing
knowable for an unseen protein. Stats are fit on **training proteins only**;
unseen proteins fall back to global train stats (used only for monitoring val
loss, since the selection metric is correlation, which is invariant to this
affine map). `log1p` tames the right skew; `clip_quantile` winsorizes the thin
upper tail.

---

## Swapping components (registries)

Drop in a new component by subclassing the relevant base in
[encoders.py](encoders.py) and decorating it
(`@register_dna_encoder("name")`), then set that `name` in config — **no
training-code changes needed**. `_build` instantiates the class with only the
spec keys its constructor accepts, so one config block can carry the union of
options' params (the flip side: a typo'd key is silently ignored).

| Slot | Config key | Built-in options |
|---|---|---|
| DNA encoder | `dna_encoder.name` | `cnn`, `cnn_attn`, `cnn_deep` (pooled `[B,Dd]`); `cnn_deep_pos` (per-position `[B,L,Dd]`); `rnn` |
| Protein head | `protein_encoder.name` | `mlp`, `mlp_res`, `linear`, `identity` (pooled `[B,Dp]`); `attn_pool_tokens` (tokens `[B,K,Dp]`) |
| Interaction | `interaction.name` | `concat_hadamard`, `concat`, `bilinear` (pooled inputs); `cross_attn` (token × position inputs) |
| Protein embedding | `esm_cache` | `esm2_t36_3B`, `esm2_t33_650M`, `esm2_t30_150M`, `esm2_t12_35M`, `kmer3` — pooled or per-residue |

**Compatibility.** Two coherent families:
- **Cross-attention (default):** `cnn_deep_pos` + `attn_pool_tokens` +
  `cross_attn`, fed by a **per-residue** `esm_cache`. The token/position tensors
  flow end-to-end.
- **Pooled (classic two-tower):** any pooled DNA encoder (`cnn` / `cnn_attn` /
  `cnn_deep` / `rnn`) + a pooled protein head (`mlp` / `mlp_res` / `linear` /
  `identity`) + a pooled interaction (`concat_hadamard` / `concat` /
  `bilinear`), fed by a **pooled** `esm_cache`.

`model.py`'s smoke test exercises every combination with synthetic 2-D inputs
(the `cross_attn` head unsqueezes 2-D inputs to a single token/position), so it
passes for all pairings even though the trainer uses the families above.

Swap the **protein representation** by building a different cache and pointing at
it:
```bash
python embed_proteins.py --embedder esm2_t33_650M_UR50D --per-residue   # higher quality (1280-d/residue)
python embed_proteins.py --embedder kmer3                               # cheap no-ESM baseline (8000-d pooled)
# then set  esm_cache: cache/<name>.pt  in config.yaml
```

### Embedders ([embed_proteins.py](embed_proteins.py))

| `--embedder` | Dim | Notes |
|---|---|---|
| `esm2_t36_3B_UR50D` | 2560 | frozen ESM-2 3B, best quality |
| `esm2_t33_650M_UR50D` | 1280 | frozen ESM-2 650M, good quality (default) |
| `esm2_t30_150M_UR50D` | 640 | frozen ESM-2 150M, good MPS default |
| `esm2_t12_35M_UR50D` | 480 | frozen ESM-2 35M, fastest |
| `kmer3` | 8000 | normalized 3-mer composition, no model; baseline |

Two cache layouts:
- **Pooled** (default): `{"embeddings": [P, D]}` — one vector per protein.
  Residue pooling is `--pool {mean,meanmax,meanmaxstd}`.
- **Per-residue** (`--per-residue`): `{"embeddings": [P, R, D] fp16,
  "lengths": [P], "per_residue": True}` — padded residue reps + true lengths,
  **required** by `attn_pool_tokens` / `cross_attn`.

---

## Domain extraction

The single biggest lever in the project brief: *don't embed the whole protein.*
Binding specificity lives in a short, structured DNA-binding domain (DBD); the
rest of the chain is disordered linker and activation domains that add noise and
cost ESM compute. [homeodomain.py](homeodomain.py) trims each TF to its DBD with
two tools, then [embed_domains.py](embed_domains.py) embeds the trimmed domain.

- **`find_homeodomain`** — precise, motif-anchored. The homeodomain recognition
  helix carries the near-invariant **WFxNxR** signature; it anchors there and
  cuts the canonical 60-residue domain at standard numbering (W at position 48).
- **`find_dbd_window`** — family-agnostic fallback. DBDs are enriched in basic
  residues (K/R, grip the backbone) and aromatics/Asn (read bases); it slides a
  window and keeps the one maximizing that composition.
- **`extract_dbd`** uses the precise anchor when a homeodomain is detected and
  the generic window otherwise, returning the trimmed sequence plus auditable
  metadata (`method`, `start`, `length`, `anchor`).

`python homeodomain.py` prints coverage and **self-validates**: where the
precise anchor exists, the generic window should land on the same recognition
helix (high overlap ⇒ the fallback is trustworthy on families it cannot anchor).
[embed_domains.py](embed_domains.py) writes
`cache/<embedder>_homeodomain_perres.pt` (~14× smaller than the full-protein
cache); [config_homeodomain.yaml](config_homeodomain.yaml) points at it so the
full-protein-vs-domain comparison is a clean A/B (only the protein input region
changes).

---

## Training internals

[train.py](train.py) runs **leave-proteins-out** CV: for each fold, train on all
`(DNA, protein)` pairs whose protein is in the training group, validate on pairs
whose protein is held out (never seen in training, via `protein_group_folds`).

- **Blocked outer-product step.** Each step samples `probe_block` unique probes,
  encodes them **once** with the DNA tower, encodes **all** train proteins once
  with the protein tower, then scores the full `block × protein` grid through the
  interaction head — no redundant convolutions. With `cross_attn`, the grid is a
  single batched einsum via `score_grid`; otherwise it falls back to expanding
  the pair grid. `batch_size` in config is legacy and unused here.
- **Loss.** Elementwise Huber + `corr_lambda · (1 − per-protein Pearson)` over
  each probe block, so training is aligned with the eval metric.
- **Regularization.** `esm_noise` adds Gaussian noise to the frozen residue reps
  (the protein side has only ~400 TFs and overfits easily); `rc_prob` applies
  reverse-complement augmentation; gradients are clipped to norm 1.
- **Schedule.** Cosine with linear warmup and an LR floor (`min_lr_frac`), or
  `plateau` on the val metric.
- **Selection.** Early stopping on mean per-held-out-protein Pearson; the
  best-epoch checkpoint (`model_fold{k}.pt`) and validation predictions
  (`preds_fold{k}.npz`) are saved per fold under `--out-dir` (default `runs/`).
- **Eval-time.** `tta_rc` averages each probe with its reverse-complement.

Useful `train.py` flags: `--folds N` (limit folds), `--epochs N` (override),
`--esm-cache PATH` (A/B different protein towers), `--seed N` (seed-ensembling),
`--probe-subsample N` (deterministic probe subset for fast, fair A/B smoke runs).

---

## Evaluation & baselines

[evaluate.py](evaluate.py) concatenates the per-fold prediction files and reports
mean ± std per-protein Pearson/Spearman for:

- **Trained model** — from `runs/preds_fold*.npz`.
- **Per-protein-mean** — trivial floor; ~0 correlation by construction.
- **kNN-in-ESM** — predict a held-out protein's profile by averaging the binding
  profiles of its *k* nearest training proteins in ESM space (cosine, same folds
  as training). This is the **honest bar**: the model is only useful if it beats
  "copy the nearest known protein." Set *k* with `--knn-k`.

> The kNN baseline expects a **pooled** `esm_cache` (`[P, D]`). When training
> with a per-residue cache, point `--config`/`esm_cache` at a pooled cache (or
> build one) before running `evaluate.py`.

It also saves a predicted-vs-true scatter (`runs/scatter_heldout.png`) for a few
held-out proteins.

[ensemble.py](ensemble.py) averages held-out predictions across runs trained with
different seeds (aligned by `(prot_idx, dna_idx)`) and reports the lift over the
best single run — the cheap "a few seeds averaged" bump.

[analyze_preds.py](analyze_preds.py) takes one `preds_fold*.npz` and reports the
**distribution** of per-protein Pearson (histogram, worst offenders, and the
correlation between a protein's target spread and its predictability) — to tell
whether a low mean is a tail of near-constant "unpredictable" TFs (a ceiling) or
a model-wide weakness.

---

## Files

| File | Role |
|---|---|
| [explore_data.py](explore_data.py) | EDA: distribution, skew, outliers, per-protein scale |
| [data.py](data.py) | parsing, one-hot + reverse-complement, protein-grouped folds, `TargetTransform`, `PairDataset` |
| [homeodomain.py](homeodomain.py) | DNA-binding-domain extraction (homeodomain anchor + generic K/R window) |
| [embed_proteins.py](embed_proteins.py) | protein-embedder registry (ESM-2 / k-mer) → cached pooled `[P,D]` or per-residue `[P,R,D]` |
| [embed_domains.py](embed_domains.py) | same, over the trimmed DBD (~60 aa) |
| [encoders.py](encoders.py) | swappable DNA / protein / interaction registries |
| [model.py](model.py) | `BindingModel` + `build_model` factory + all-combos wiring test |
| [metrics.py](metrics.py) | per-protein Pearson/Spearman (NaN-safe for constant inputs) |
| [train.py](train.py) | leave-proteins-out CV loop, blocked trainer, early stopping, per-fold predictions |
| [evaluate.py](evaluate.py) | aggregate metrics + per-protein-mean and kNN-in-ESM baselines + scatter plot |
| [ensemble.py](ensemble.py) | seed-ensemble held-out predictions; report lift over best single run |
| [analyze_preds.py](analyze_preds.py) | per-protein Pearson distribution diagnostic for one fold |
| [utils.py](utils.py) | device selection (MPS/CUDA/CPU) + YAML config loader |
| [config.yaml](config.yaml) | full-protein hyperparameters + component selection |
| [config_homeodomain.yaml](config_homeodomain.yaml) | domain-trimmed variant (only `esm_cache` differs) |

Generated artifacts (git-ignored): `cache/` (protein embeddings), `runs*/`
(per-fold checkpoints, predictions, plots).

---

## Reading the results

Success = mean held-out-protein **Spearman ρ meaningfully above the kNN-in-ESM
baseline**. With a limited number of proteins, expect high per-fold variance; a
larger, more diverse protein set is what makes unseen-protein generalization
work. Use [analyze_preds.py](analyze_preds.py) to check whether the gap to a
perfect score is a hard tail of near-constant TFs (a data ceiling) or something
the model can still learn.

python -u train.py --config config.yaml --folds 1 --epochs 80 \
  --out-dir runs/baseline_f0 > baseline_f0.log 2>&1