#!/usr/bin/env python3
"""
prepare_dataset.py
===================

Main entry point for the Nirikshan data pipeline. Takes the per-source
manifests produced by download_sources.py (+ ita_estimation.py) and by
generate_stage1_negatives.py, and produces the final training-ready
artifacts:

    data/processed/images/<split>/*.jpg           -- deduped, resized 224x224
    data/processed/{train,val,test}_manifest.csv   -- final manifests
    data/processed/dataset_summary.json            -- per-class / per-skin-tone
                                                        counts (source for the
                                                        "honest framing" numbers)

Pipeline steps
--------------
1. Load and concatenate every input manifest (Stage-2 disease-class sources
   + the Stage-1 negatives manifest). All manifests must already have an
   `ita_bin` column (run ita_estimation.py on each source manifest first;
   Stage-1 negatives from strategy (a)/(b)/(c) inherit ita_bin='unknown'
   since ITA on cropped normal-skin patches / already-healthy images is
   less clinically meaningful -- they still get a fairness bucket so
   stratification doesn't break, but it's a fixed "unknown" bucket).
2. Deduplicate near-identical images via perceptual hashing (imagehash
   phash), since several sources (ISIC Archive, ISIC 2020, BCN20000) are
   known to overlap.
3. Resize every surviving image to 224x224 with Pillow LANCZOS and copy
   into data/processed/images/<split>/.
4. Stratified 70/15/15 train/val/test split, stratified jointly by
   (label, ita_bin) via a combined key, using sklearn's train_test_split
   twice (train vs rest, then val vs test). Rare (label, ita_bin)
   combinations that can't be split three ways (fewer than 3 examples)
   are all routed to `train` and a warning is printed, so fairness
   evaluation on val/test only ever sees strata with enough support.
5. Writes final manifests + dataset_summary.json with per-class and
   per-skin-tone-bin counts (overall and per split).

Usage
-----
    python prepare_dataset.py \\
        --input-manifests data/raw/isic_archive/manifest_with_ita.csv \\
                           data/raw/bcn20000/manifest_with_ita.csv \\
                           data/raw/ham10000/manifest_with_ita.csv \\
                           data/raw/pad_ufes_20/manifest_with_ita.csv \\
                           data/raw/fitzpatrick17k/manifest_with_ita.csv \\
                           data/raw/ddi/manifest_with_ita.csv \\
        --input-images-roots data/raw/isic_archive data/raw/bcn20000 \\
                              data/raw/ham10000 data/raw/pad_ufes_20 \\
                              data/raw/fitzpatrick17k data/raw/ddi \\
        --stage1-negatives-manifest data/raw/stage1_negatives/manifest.csv \\
        --stage1-negatives-images-root data/raw/stage1_negatives \\
        --output-dir data/processed
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = 224
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Raw source labels -> Stage-2's 4 disease classes. Anything NOT in this map
# (melanoma, BCC, scar, solar lentigo, dermatofibroma, blank/NaN, etc.) is
# outside Stage-2's scope and gets dropped here -- this is the label
# harmonization download_sources.py's docstring says happens in this file.
# Stage-1 negatives already carry label="no_lesion" and are NOT passed
# through this map (see harmonize_disease_labels below).
LABEL_MAP = {
    # BCN20000 -- raw values from its diagnosis_3 column
    "Squamous cell carcinoma, NOS": "squamous_cell_carcinoma",
    "Solar or actinic keratosis": "actinic_keratosis",
    "Nevus": "nevus",
    "Seborrheic keratosis": "seborrheic_keratosis",
    # PAD-UFES-20 -- raw values from its "diagnostic" column
    "SCC": "squamous_cell_carcinoma",
    "ACK": "actinic_keratosis",
    "NEV": "nevus",
    "SEK": "seborrheic_keratosis",
    # Add more sources' raw label strings here as you bring them in
    # (HAM10000: nv/mel/bkl/bcc/akiec/vasc/df -- 'akiec'->actinic_keratosis,
    # 'nv'->nevus, 'bkl'->seborrheic_keratosis are the relevant ones;
    # Fitzpatrick17k/DDI: check their own label vocab before mapping).
}


def harmonize_disease_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Maps each row's raw source label to one of Stage-2's 4 classes via
    LABEL_MAP, dropping every row whose label isn't in that map (e.g.
    melanoma, BCC -- out of this project's clinical scope per
    PROJECT_CONTEXT.md). Only call this on disease-class manifests, NOT
    on the Stage-1 negatives manifest (whose label is already the
    correct, final "no_lesion")."""
    before = len(df)
    df = df.copy()
    raw_counts = df["label"].value_counts(dropna=False)
    df["label"] = df["label"].map(LABEL_MAP)
    dropped = df["label"].isna().sum()
    df = df[df["label"].notna()].reset_index(drop=True)
    print(f"\nLabel harmonization: kept {len(df)}/{before} disease-class rows "
          f"({dropped} rows had a raw label outside the 4 Stage-2 classes "
          "and were dropped -- see LABEL_MAP in this script).")
    unmapped_raw = set(raw_counts.index) - set(LABEL_MAP.keys())
    if unmapped_raw:
        print("  Raw labels seen but NOT in LABEL_MAP (dropped): "
              f"{sorted(str(x) for x in unmapped_raw)}")
    return df


def load_and_concat_manifests(manifest_paths, images_roots):
    if len(manifest_paths) != len(images_roots):
        print("ERROR: --input-manifests and --input-images-roots must have "
              "the same length (one images-root per manifest).", file=sys.stderr)
        sys.exit(1)

    frames = []
    for manifest_path, images_root in zip(manifest_paths, images_roots):
        df = pd.read_csv(manifest_path)
        if "ita_bin" not in df.columns:
            print(f"ERROR: {manifest_path} has no ita_bin column -- run "
                  "ita_estimation.py on it first.", file=sys.stderr)
            sys.exit(1)
        df["_images_root"] = str(images_root)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_stage1_negatives(manifest_path, images_root):
    if manifest_path is None:
        return pd.DataFrame(columns=["image_path", "source", "label",
                                      "patient_id", "ita_bin", "_images_root"])
    df = pd.read_csv(manifest_path)
    df["ita_bin"] = "unknown"  # negatives skip per-tone ITA, see module docstring
    df["_images_root"] = str(images_root)
    return df


def dedupe_by_phash(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose image perceptual hash has already been seen.
    First occurrence (in current dataframe order) wins."""
    seen_hashes = set()
    keep_mask = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dedupe (phash)"):
        img_path = Path(row["_images_root"]) / row["image_path"]
        if not img_path.exists():
            keep_mask.append(False)
            continue
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img)
        except Exception:  # noqa: BLE001
            keep_mask.append(False)
            continue
        h_str = str(h)
        if h_str in seen_hashes:
            keep_mask.append(False)
        else:
            seen_hashes.add(h_str)
            keep_mask.append(True)
    df = df[pd.Series(keep_mask, index=df.index)].copy()
    return df


def resize_and_copy(df: pd.DataFrame, split_name: str, out_images_dir: Path) -> list:
    split_dir = out_images_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Resizing [{split_name}]"):
        src_path = Path(row["_images_root"]) / row["image_path"]
        out_name = f"{row['source']}_{Path(row['image_path']).stem}_{i}.jpg"
        out_path = split_dir / out_name
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(out_path, quality=95)
            new_paths.append(f"{split_name}/{out_name}")
        except Exception:  # noqa: BLE001
            new_paths.append(None)
    return new_paths


def stratified_three_way_split(df: pd.DataFrame, seed: int):
    """Stratify jointly by (label, ita_bin). Strata with < 3 members can't
    be split three ways -- they're routed entirely to train, with a
    warning, so val/test strata always have enough support to be
    meaningful for fairness evaluation.

    Degrades gracefully (never raises) on small/early-stage datasets:
    - If every stratum is "small" (e.g. an early pilot run with only a
      handful of images per class), everything goes to train and val/test
      come back empty -- a loud warning is printed rather than silently
      producing an unusable split.
    - If a large-enough stratum still leaves too few examples in `rest_df`
      to be split into val/test with stratification (can happen right at
      the 3-4 example boundary), that second split falls back to a
      non-stratified split rather than crashing."""
    df = df.copy()
    df["_strata_key"] = df["label"].astype(str) + "||" + df["ita_bin"].astype(str)

    counts = Counter(df["_strata_key"])
    small_strata = {k for k, v in counts.items() if v < 3}
    if small_strata:
        print(f"\nWARNING: {len(small_strata)} (label, ita_bin) strata have "
              "fewer than 3 examples and cannot be split across train/val/"
              "test. All examples in these strata are routed to `train`:")
        for k in sorted(small_strata):
            print(f"    {k}  (n={counts[k]})")

    small_df = df[df["_strata_key"].isin(small_strata)]
    large_df = df[~df["_strata_key"].isin(small_strata)]

    empty = df.iloc[0:0].drop(columns=["_strata_key"])

    if len(large_df) == 0:
        print("\nWARNING: every (label, ita_bin) stratum has fewer than 3 "
              "examples -- this looks like an early pilot run on a very "
              "small dataset. ALL rows are going to `train`; val and test "
              "are empty for this run. Re-run prepare_dataset.py once more "
              "images are available before trusting any val/test metrics.")
        return small_df.drop(columns=["_strata_key"]), empty, empty

    train_df, rest_df = train_test_split(
        large_df, train_size=SPLIT_RATIOS["train"], random_state=seed,
        stratify=large_df["_strata_key"],
    )

    val_frac_of_rest = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
    rest_counts = Counter(rest_df["_strata_key"])
    can_stratify_rest = len(rest_df) >= 2 and min(rest_counts.values()) >= 2
    if len(rest_df) == 0:
        val_df, test_df = rest_df, rest_df
    elif len(rest_df) == 1:
        # single leftover example: arbitrarily but deterministically put it
        # in test rather than crash on a split of size 1
        val_df, test_df = rest_df.iloc[0:0], rest_df
    elif can_stratify_rest:
        val_df, test_df = train_test_split(
            rest_df, train_size=val_frac_of_rest, random_state=seed,
            stratify=rest_df["_strata_key"],
        )
    else:
        print("\nWARNING: the val/test remainder has strata too small "
              "(fewer than 2 examples) to stratify further -- falling back "
              "to a plain random split for val/test on this remainder.")
        val_df, test_df = train_test_split(
            rest_df, train_size=val_frac_of_rest, random_state=seed,
        )

    train_df = pd.concat([train_df, small_df], ignore_index=False)

    for d in (train_df, val_df, test_df):
        d.drop(columns=["_strata_key"], inplace=True)
    return train_df, val_df, test_df


def build_summary(train_df, val_df, test_df) -> dict:
    def counts_for(d: pd.DataFrame) -> dict:
        return {
            "n_images": int(len(d)),
            "by_label": d["label"].value_counts().to_dict(),
            "by_ita_bin": d["ita_bin"].value_counts().to_dict(),
            "by_label_and_ita_bin": (
                d.groupby(["label", "ita_bin"]).size().to_dict()
                if len(d) else {}
            ),
        }

    # groupby tuple keys aren't JSON-serializable directly -- stringify them
    def stringify_tuple_keys(d: dict) -> dict:
        return {f"{k[0]}||{k[1]}": v for k, v in d.items()}

    summary = {
        "overall": counts_for(pd.concat([train_df, val_df, test_df])),
        "train": counts_for(train_df),
        "val": counts_for(val_df),
        "test": counts_for(test_df),
    }
    for split in summary:
        summary[split]["by_label_and_ita_bin"] = stringify_tuple_keys(
            summary[split]["by_label_and_ita_bin"]
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-manifests", nargs="+", required=True, type=Path,
                         help="Stage-2 disease-class source manifests (each must "
                              "already have an ita_bin column).")
    parser.add_argument("--input-images-roots", nargs="+", required=True, type=Path,
                         help="One images-root per --input-manifests entry, same order.")
    parser.add_argument("--stage1-negatives-manifest", type=Path, default=None,
                         help="Output of generate_stage1_negatives.py.")
    parser.add_argument("--stage1-negatives-images-root", type=Path, default=None,
                         help="Images root for the Stage-1 negatives manifest.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dedupe", action="store_true",
                         help="Skip perceptual-hash dedup (faster iteration during dev).")
    args = parser.parse_args()

    print("Loading manifests...")
    disease_df = load_and_concat_manifests(args.input_manifests, args.input_images_roots)
    disease_df = harmonize_disease_labels(disease_df)
    neg_df = load_stage1_negatives(args.stage1_negatives_manifest,
                                    args.stage1_negatives_images_root)
    df = pd.concat([disease_df, neg_df], ignore_index=True, sort=False)
    df = df[df["ita_bin"].astype(str) != ""].reset_index(drop=True)
    print(f"Loaded {len(df)} total rows across all sources "
          f"({len(disease_df)} disease-class + {len(neg_df)} stage1-negatives).")

    if not args.skip_dedupe:
        before = len(df)
        df = dedupe_by_phash(df)
        print(f"Deduped: {before} -> {len(df)} rows "
              f"({before - len(df)} duplicates/unreadable removed).")

    print("\nSplitting (stratified by label x ita_bin, 70/15/15)...")
    train_df, val_df, test_df = stratified_three_way_split(df, args.seed)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    out_images_dir = args.output_dir / "images"
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["image_path"] = resize_and_copy(train_df, "train", out_images_dir)
    val_df["image_path"] = resize_and_copy(val_df, "val", out_images_dir)
    test_df["image_path"] = resize_and_copy(test_df, "test", out_images_dir)

    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        before = len(d)
        d.dropna(subset=["image_path"], inplace=True)
        if len(d) != before:
            print(f"  [{name}] dropped {before - len(d)} rows that failed to resize/copy.")

    keep_cols = ["image_path", "source", "label", "patient_id", "ita_bin"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = args.output_dir / f"{name}_manifest.csv"
        d[keep_cols].to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(d)} rows)")

    summary = build_summary(train_df, val_df, test_df)
    summary_path = args.output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    print("\nOverall label distribution:")
    print(pd.Series(summary["overall"]["by_label"]))
    print("\nOverall ITA-bin distribution:")
    print(pd.Series(summary["overall"]["by_ita_bin"]))


if __name__ == "__main__":
    main()
