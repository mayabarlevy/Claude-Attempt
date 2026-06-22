"""Aggregate leave-proteins-out CV results and compare against baselines.

Reads the per-fold prediction files written by train.py (runs/preds_fold*.npz)
and reports mean +/- std per-protein Pearson/Spearman for:
  - the trained model
  - per-protein-mean baseline (trivial floor; correlation ~ 0 by construction)
  - kNN-in-ESM baseline: predict a held-out protein's profile by averaging the
    binding profiles of its nearest training proteins in ESM space.

The kNN baseline is the honest bar: the model is only useful if it beats
"copy the nearest known protein."

Usage:
    python evaluate.py
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from data import load_data, protein_group_folds
from metrics import per_protein_correlations
from utils import load_config


def model_results(run_dir: str) -> dict:
    """Concatenate all fold prediction files and compute per-protein metrics."""
    files = sorted(glob.glob(os.path.join(run_dir, "preds_fold*.npz")))
    if not files:
        raise FileNotFoundError(f"no preds_fold*.npz in {run_dir}; run train.py first")
    pidx, ytrue, ypred = [], [], []
    for f in files:
        z = np.load(f)
        pidx.append(z["prot_idx"])
        ytrue.append(z["y_true"])
        ypred.append(z["y_pred"])
    return per_protein_correlations(
        np.concatenate(pidx), np.concatenate(ytrue), np.concatenate(ypred)
    )


def knn_esm_baseline(data, esm_emb: np.ndarray, n_splits: int, seed: int, k: int = 1) -> dict:
    """Leave-proteins-out kNN baseline in ESM embedding space.

    For each held-out protein, find its k nearest training proteins (cosine
    distance in ESM space) and predict its per-probe affinity as their mean
    profile. Uses the SAME protein folds as training.
    """
    aff = data.affinity  # [N, P]
    # L2-normalize for cosine similarity
    emb = esm_emb / (np.linalg.norm(esm_emb, axis=1, keepdims=True) + 1e-8)

    all_pidx, all_true, all_pred = [], [], []
    for train_ids, val_ids in protein_group_folds(data.n_prot, n_splits=n_splits, seed=seed):
        sims = emb[val_ids] @ emb[train_ids].T  # [n_val, n_train]
        for row, vid in enumerate(val_ids):
            nn_local = np.argsort(-sims[row])[:k]
            nn_global = train_ids[nn_local]
            pred_profile = aff[:, nn_global].mean(axis=1)  # [N]
            all_pidx.append(np.full(data.n_dna, vid))
            all_true.append(aff[:, vid])
            all_pred.append(pred_profile)
    return per_protein_correlations(
        np.concatenate(all_pidx), np.concatenate(all_true), np.concatenate(all_pred)
    )


def maybe_plot(run_dir: str, out_png: str, n_examples: int = 3) -> None:
    """Scatter predicted vs true for a few held-out proteins (qualitative check)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping plot: {e})")
        return

    files = sorted(glob.glob(os.path.join(run_dir, "preds_fold*.npz")))
    z = np.load(files[0])
    ids = np.unique(z["prot_idx"])[:n_examples]
    fig, axes = plt.subplots(1, len(ids), figsize=(4 * len(ids), 4))
    if len(ids) == 1:
        axes = [axes]
    for ax, pid in zip(axes, ids):
        m = z["prot_idx"] == pid
        ax.scatter(z["y_true"][m], z["y_pred"][m], s=4, alpha=0.3)
        ax.set_title(f"held-out protein {pid}")
        ax.set_xlabel("true affinity")
        ax.set_ylabel("predicted")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved scatter plot -> {out_png}")


def fmt(name: str, r: dict) -> str:
    return (
        f"{name:<28} Pearson {r['pearson_mean']:.4f} +/- {r['pearson_std']:.4f}   "
        f"Spearman {r['spearman_mean']:.4f} +/- {r['spearman_std']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--knn-k", type=int, default=1)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(here, args.config))

    data = load_data(
        os.path.join(here, cfg["dbp_path"]),
        os.path.join(here, cfg["seq_path"]),
        os.path.join(here, cfg["data_path"]),
    )
    ckpt = torch.load(os.path.join(here, cfg["esm_cache"]), map_location="cpu")
    esm_emb = ckpt["embeddings"].float()

    if esm_emb.ndim == 3:
        lengths = ckpt.get("lengths", None)
        pooled = []
        for i in range(esm_emb.shape[0]):
            if lengths is not None:
                L = int(lengths[i])
                pooled.append(esm_emb[i, :L].mean(dim=0))
            else:
                pooled.append(esm_emb[i].mean(dim=0))
        esm_emb = torch.stack(pooled, dim=0)

    esm_emb = esm_emb.numpy()

    run_dir = os.path.join(here, args.run_dir)
    print("=== Leave-proteins-out evaluation (mean +/- std over held-out proteins) ===")
    print(fmt("Trained model", model_results(run_dir)))
    print(fmt(f"kNN-in-ESM (k={args.knn_k})", knn_esm_baseline(data, esm_emb, cfg["n_splits"], cfg["seed"], args.knn_k)))
    print("(per-protein-mean baseline has ~0 correlation by construction)")
    maybe_plot(run_dir, os.path.join(run_dir, "scatter_heldout.png"))


if __name__ == "__main__":
    main()
