#!/usr/bin/env python3
"""
inject_fitzpatrick_pad_ufes20.py
==================================

One-off patch script: PAD-UFES-20's own metadata.csv carries a REAL,
clinician-assigned Fitzpatrick skin type (column "fitspatrick", values
1-6, ~35% blank) -- a genuine fairness signal, unlike the pixel-based ITA
estimate ita_estimation.py falls back to for sources with no mask/bbox.

This script overwrites ita_bin in data/raw/pad_ufes_20/manifest_with_ita.csv
for every row that has a known Fitzpatrick type, using the approximate
Fitzpatrick-type -> ITA-bin correspondence below (documented as an
ESTIMATE, not a precise conversion -- Fitzpatrick type and ITA are related
but not identical scales). Rows with a blank Fitzpatrick type are left as
"unknown", same as before.

Usage
-----
    python inject_fitzpatrick_pad_ufes20.py \\
        --manifest-with-ita data/raw/pad_ufes_20/manifest_with_ita.csv \\
        --original-metadata "C:/Users/ANEEK/Downloads/PAD-UFES-20/metadata.csv"
"""
import argparse
from pathlib import Path

import pandas as pd

# ESTIMATE: approximate Fitzpatrick-type -> ITA-bin correspondence.
# Validate with a clinical partner before trusting this for anything beyond
# a first-pass fairness breakdown.
FITZPATRICK_TO_BIN = {
    1: "very_light",
    2: "light",
    3: "intermediate",
    4: "tan",
    5: "dark",
    6: "very_dark",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-with-ita", type=Path, required=True)
    p.add_argument("--original-metadata", type=Path, required=True)
    p.add_argument("--img-id-col", default="img_id")
    p.add_argument("--fitzpatrick-col", default="fitspatrick")
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest_with_ita)
    meta = pd.read_csv(args.original_metadata)

    # img_id in PAD-UFES-20's metadata includes the extension (e.g. "PAT_8_15_820.png");
    # manifest's image_path is "images/PAT_8_15_820.png". Join on the stem.
    meta["_stem"] = meta[args.img_id_col].astype(str).str.replace(
        r"\.(png|jpg|jpeg)$", "", regex=True, case=False
    )
    fst_by_stem = dict(zip(meta["_stem"], meta[args.fitzpatrick_col]))

    manifest["_stem"] = manifest["image_path"].apply(lambda p: Path(p).stem)

    def resolve(stem):
        fst = fst_by_stem.get(stem)
        if pd.isna(fst):
            return None
        try:
            return FITZPATRICK_TO_BIN.get(int(fst))
        except (ValueError, TypeError):
            return None

    resolved_bins = manifest["_stem"].apply(resolve)
    n_updated = resolved_bins.notna().sum()
    manifest.loc[resolved_bins.notna(), "ita_bin"] = resolved_bins[resolved_bins.notna()]
    manifest.drop(columns=["_stem"], inplace=True)

    manifest.to_csv(args.manifest_with_ita, index=False)

    print(f"Updated {n_updated}/{len(manifest)} rows with a real "
          "Fitzpatrick-derived ita_bin.")
    print("New ita_bin distribution:")
    print(manifest["ita_bin"].value_counts(dropna=False).to_string())
    print(f"\nWrote {args.manifest_with_ita}")


if __name__ == "__main__":
    main()
