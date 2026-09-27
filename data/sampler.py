#!/usr/bin/env python3
"""
sampler.py
==========

Weighted-sampler helper for the Stage-2 disease-classifier PyTorch
DataLoader, implementing B.6.C points 1-2:

    1. Oversample darker ITA bins ("dark", "very_dark") -- these are
       systematically underrepresented in every public dermatology
       dataset, and this is the model's single biggest fairness risk.
    2. Oversample the labeled-diverse sources: PAD-UFES-20 (Brazilian
       population, broader skin-tone range than ISIC-family sources) and
       the FST 4-6 slice of Fitzpatrick17k specifically (rather than
       Fitzpatrick17k as a whole, which spans FST 1-6).

Weights combine multiplicatively: a PAD-UFES-20 image in the "dark" ITA
bin gets both boosts. Final per-sample weight also includes the standard
inverse-class-frequency term so this sampler can fully replace (not just
supplement) class-balancing logic in the training loop.

This module is intentionally decoupled from the training script: it
consumes a manifest DataFrame (e.g. data/processed/train_manifest.csv, with
`source` and `ita_bin` columns from prepare_dataset.py) and returns a
ready-to-use `torch.utils.data.WeightedRandomSampler`.

Usage (inside a training script)
---------------------------------
    import pandas as pd
    from sampler import build_weighted_sampler

    train_df = pd.read_csv("data/processed/train_manifest.csv")
    sampler = build_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler,
        # NOTE: do not also pass shuffle=True -- a sampler replaces shuffling
    )

Standalone usage (prints resulting weight stats, no training)
---------------------------------------------------------------
    python sampler.py --manifest data/processed/train_manifest.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DARK_ITA_BINS = {"dark", "very_dark"}
DIVERSE_SOURCES = {"pad_ufes_20"}  # boosted regardless of ita_bin
FITZPATRICK_SOURCE_NAME = "fitzpatrick17k"
FITZPATRICK_DARK_BINS = {"tan", "dark", "very_dark"}  # proxy for FST 4-6 when
# an explicit fitzpatrick_type column isn't present in the manifest


def compute_sample_weights(
    df: pd.DataFrame,
    dark_ita_boost: float = 2.0,
    diverse_source_boost: float = 1.5,
    fitzpatrick_46_boost: float = 1.5,
    use_class_balancing: bool = True,
) -> np.ndarray:
    """Returns a 1D array of per-row sample weights, same length/order as df.

    `df` must have `label` and `ita_bin` columns; `source` and, optionally,
    `fitzpatrick_type` are used for the diverse-source boost when present.
    """
    n = len(df)
    weights = np.ones(n, dtype=np.float64)

    if use_class_balancing:
        class_counts = df["label"].value_counts()
        inv_freq = (1.0 / class_counts).to_dict()
        # normalize so the mean class weight is 1.0 (keeps overall scale stable)
        mean_inv = np.mean(list(inv_freq.values()))
        class_weight_map = {k: v / mean_inv for k, v in inv_freq.items()}
        weights *= df["label"].map(class_weight_map).fillna(1.0).to_numpy()

    is_dark_ita = df["ita_bin"].isin(DARK_ITA_BINS).to_numpy()
    weights[is_dark_ita] *= dark_ita_boost

    if "source" in df.columns:
        is_diverse_source = df["source"].isin(DIVERSE_SOURCES).to_numpy()
        weights[is_diverse_source] *= diverse_source_boost

        is_fitzpatrick = (df["source"] == FITZPATRICK_SOURCE_NAME).to_numpy()
        if "fitzpatrick_type" in df.columns:
            is_fst_46 = df["fitzpatrick_type"].astype(str).isin(
                {"4", "5", "6", "IV", "V", "VI"}
            ).to_numpy()
        else:
            # proxy: use ita_bin as a stand-in for Fitzpatrick skin type
            # when the manifest doesn't carry the original FST label
            is_fst_46 = df["ita_bin"].isin(FITZPATRICK_DARK_BINS).to_numpy()
        weights[is_fitzpatrick & is_fst_46] *= fitzpatrick_46_boost

    return weights


def build_weighted_sampler(df: pd.DataFrame, **weight_kwargs):
    """Returns a torch.utils.data.WeightedRandomSampler built from
    compute_sample_weights(df, **weight_kwargs). Imports torch lazily so
    this module stays importable in pure-data-tooling environments without
    a torch install (e.g. inside the other data/*.py scripts)."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    weights = compute_sample_weights(df, **weight_kwargs)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                         help="A processed manifest CSV, e.g. train_manifest.csv, "
                              "with label/ita_bin/source columns.")
    parser.add_argument("--dark-ita-boost", type=float, default=2.0)
    parser.add_argument("--diverse-source-boost", type=float, default=1.5)
    parser.add_argument("--fitzpatrick-46-boost", type=float, default=1.5)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    weights = compute_sample_weights(
        df,
        dark_ita_boost=args.dark_ita_boost,
        diverse_source_boost=args.diverse_source_boost,
        fitzpatrick_46_boost=args.fitzpatrick_46_boost,
    )

    print(f"Computed weights for {len(df)} rows.")
    print(f"  min={weights.min():.3f}  max={weights.max():.3f}  "
          f"mean={weights.mean():.3f}  std={weights.std():.3f}")

    df_w = df.copy()
    df_w["_weight"] = weights
    print("\nMean weight by ita_bin:")
    print(df_w.groupby("ita_bin")["_weight"].mean().sort_index())
    if "source" in df.columns:
        print("\nMean weight by source:")
        print(df_w.groupby("source")["_weight"].mean().sort_values(ascending=False))

    print("\nEffective sample count by ita_bin after weighting "
          "(sum of weights, i.e. expected draws per epoch):")
    print(df_w.groupby("ita_bin")["_weight"].sum().sort_index())


if __name__ == "__main__":
    main()
