#!/usr/bin/env python3
"""
generate_stage1_negatives.py
=============================

Builds the "no lesion present" negative class for Stage 1 (the binary
lesion-presence detector), per B.3's strategy:

    (a) cropped normal-skin patches taken from OUTSIDE the annotated lesion
        region of wide-field ISIC / BCN20000 / ISIC-2020 source images
    (b) the MCSI healthy subset (~100 images), used as-is (already
        healthy-skin images, no cropping needed)
    (c) a placeholder folder for a team-run consented photo drive, so the
        rest of the pipeline has a stable path to point at once that drive
        happens, without blocking on it now

Output: a manifest CSV with the same schema as the source manifests
(image_path, source, label, patient_id), where label is always
"no_lesion", plus a `negative_strategy` column recording which of (a)/(b)/(c)
produced each row -- useful later for auditing Stage-1 training data
composition.

Crop strategy for (a)
----------------------
For each wide-field source image that has EITHER a lesion bounding box
(bbox_x, bbox_y, bbox_w, bbox_h, normalized 0-1) OR a segmentation mask
(mask_path), this script crops one or more square patches from the region
outside the lesion:
    - up to `--patches-per-image` patches (default 2) of size
      `--patch-size` (default 224x224 in source-image pixel space, scaled
      down if the source image is smaller than that)
    - patch placement is randomized among candidate positions that do not
      overlap the lesion bbox/mask (with a configurable safety margin),
      rejecting candidates after `--max-attempts` retries per patch.

Usage
-----
    python generate_stage1_negatives.py \\
        --wide-field-manifest data/raw/isic_archive/manifest_with_ita.csv \\
        --wide-field-images-root data/raw/isic_archive \\
        --mcsi-manifest data/raw/mcsi/manifest.csv \\
        --mcsi-images-root data/raw/mcsi \\
        --consented-photos-dir data/raw/team_consented_photos \\
        --output-dir data/raw/stage1_negatives \\
        --patches-per-image 2 --patch-size 224
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

RNG = random.Random(1234)  # fixed seed for reproducible negative sampling


def crop_negative_patches(image_path: Path, row: pd.Series, patch_size: int,
                           patches_per_image: int, max_attempts: int,
                           margin_frac: float = 0.02):
    """Yield up to `patches_per_image` PIL Image crops taken from outside
    the lesion region of a wide-field image. Returns [] if no localization
    info is available (image is skipped for strategy (a))."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return []
    w, h = img.size

    mask = None
    mask_path = row.get("mask_path", "")
    if isinstance(mask_path, str) and mask_path and Path(mask_path).exists():
        mask = np.array(Image.open(mask_path).convert("L").resize((w, h)))
        lesion_mask = mask > 128
    elif all(c in row.index for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")) \
            and not pd.isna(row.get("bbox_x")):
        bx, by, bw, bh = (float(row[c]) for c in
                           ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        lesion_mask = np.zeros((h, w), dtype=bool)
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = int((bx + bw) * w), int((by + bh) * h)
        # expand by safety margin
        mx, my = int(w * margin_frac), int(h * margin_frac)
        lesion_mask[max(y0 - my, 0):min(y1 + my, h),
                    max(x0 - mx, 0):min(x1 + mx, w)] = True
    else:
        return []  # no localization info -> can't safely crop negatives

    eff_patch = min(patch_size, w, h)
    if eff_patch < 32:
        return []

    patches = []
    attempts = 0
    while len(patches) < patches_per_image and attempts < max_attempts:
        attempts += 1
        x = RNG.randint(0, w - eff_patch)
        y = RNG.randint(0, h - eff_patch)
        candidate_overlap = lesion_mask[y:y + eff_patch, x:x + eff_patch].any()
        if candidate_overlap:
            continue
        crop = img.crop((x, y, x + eff_patch, y + eff_patch))
        if eff_patch != patch_size:
            crop = crop.resize((patch_size, patch_size), Image.LANCZOS)
        patches.append(crop)
    return patches


def strategy_a_wide_field_crops(args, rows_out):
    if not args.wide_field_manifest or not args.wide_field_manifest.exists():
        print("[strategy a] no wide-field manifest given/found; skipping.")
        return
    df = pd.read_csv(args.wide_field_manifest)
    out_images_dir = args.output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped_no_loc = 0, 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="[strategy a] cropping"):
        img_path = args.wide_field_images_root / row["image_path"]
        if not img_path.exists():
            continue
        patches = crop_negative_patches(
            img_path, row, args.patch_size, args.patches_per_image,
            args.max_attempts,
        )
        if not patches:
            n_skipped_no_loc += 1
            continue
        for j, patch in enumerate(patches):
            out_name = f"{Path(row['image_path']).stem}_neg{j}.jpg"
            patch.save(out_images_dir / out_name, quality=95)
            rows_out.append({
                "image_path": f"images/{out_name}",
                "source": row.get("source", "wide_field"),
                "label": "no_lesion",
                "patient_id": row.get("patient_id", ""),
                "negative_strategy": "a_wide_field_crop",
            })
            n_written += 1
    print(f"[strategy a] wrote {n_written} crops; "
          f"{n_skipped_no_loc} source images had no bbox/mask and were skipped.")


def strategy_b_mcsi(args, rows_out):
    if not args.mcsi_manifest or not args.mcsi_manifest.exists():
        print("[strategy b] no MCSI manifest given/found; skipping.")
        return
    df = pd.read_csv(args.mcsi_manifest)
    out_images_dir = args.output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[strategy b] MCSI"):
        img_path = args.mcsi_images_root / row["image_path"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        img = img.resize((args.patch_size, args.patch_size), Image.LANCZOS)
        out_name = f"mcsi_{Path(row['image_path']).stem}.jpg"
        img.save(out_images_dir / out_name, quality=95)
        rows_out.append({
            "image_path": f"images/{out_name}",
            "source": "mcsi",
            "label": "no_lesion",
            "patient_id": row.get("patient_id", ""),
            "negative_strategy": "b_mcsi_healthy",
        })
        n_written += 1
    print(f"[strategy b] wrote {n_written} MCSI healthy images.")


def strategy_c_consented_placeholder(args, rows_out):
    consented_dir = args.consented_photos_dir
    consented_dir.mkdir(parents=True, exist_ok=True)
    readme = consented_dir / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Placeholder for the team-run consented healthy-skin photo drive.\n"
            "Drop consented, deidentified healthy-skin photos directly into this "
            "folder (any of .jpg/.jpeg/.png) and re-run generate_stage1_negatives.py "
            "-- they will be picked up automatically as Stage-1 negatives on the "
            "next run, resized to the configured patch size, and merged into "
            "data/raw/stage1_negatives/manifest.csv with "
            "negative_strategy=c_team_consented.\n"
        )
        print(f"[strategy c] created placeholder + README at {consented_dir} "
              "(no photos yet -- 0 rows added this run).")

    out_images_dir = args.output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in consented_dir.glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    n_written = 0
    for img_path in tqdm(candidates, desc="[strategy c] consented photos"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        img = img.resize((args.patch_size, args.patch_size), Image.LANCZOS)
        out_name = f"consented_{img_path.stem}.jpg"
        img.save(out_images_dir / out_name, quality=95)
        rows_out.append({
            "image_path": f"images/{out_name}",
            "source": "team_consented",
            "label": "no_lesion",
            "patient_id": "",
            "negative_strategy": "c_team_consented",
        })
        n_written += 1
    print(f"[strategy c] wrote {n_written} team-consented photos.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wide-field-manifest", type=Path, default=None,
                         help="Manifest of wide-field source images with bbox/mask cols.")
    parser.add_argument("--wide-field-images-root", type=Path, default=None,
                         help="Root dir wide-field manifest's image_path is relative to.")
    parser.add_argument("--mcsi-manifest", type=Path, default=None,
                         help="Manifest of the MCSI healthy subset.")
    parser.add_argument("--mcsi-images-root", type=Path, default=None,
                         help="Root dir MCSI manifest's image_path is relative to.")
    parser.add_argument("--consented-photos-dir", type=Path,
                         default=Path("data/raw/team_consented_photos"),
                         help="Placeholder folder for the team consented-photo drive.")
    parser.add_argument("--output-dir", type=Path,
                         default=Path("data/raw/stage1_negatives"),
                         help="Where to write images/ and manifest.csv.")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patches-per-image", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=25,
                         help="Random-placement retries per patch before giving up.")
    args = parser.parse_args()

    if args.wide_field_manifest and not args.wide_field_images_root:
        print("ERROR: --wide-field-images-root is required when "
              "--wide-field-manifest is given.", file=sys.stderr)
        sys.exit(1)
    if args.mcsi_manifest and not args.mcsi_images_root:
        print("ERROR: --mcsi-images-root is required when --mcsi-manifest "
              "is given.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []

    strategy_a_wide_field_crops(args, rows_out)
    strategy_b_mcsi(args, rows_out)
    strategy_c_consented_placeholder(args, rows_out)

    manifest_path = args.output_dir / "manifest.csv"
    pd.DataFrame(rows_out, columns=[
        "image_path", "source", "label", "patient_id", "negative_strategy",
    ]).to_csv(manifest_path, index=False)

    print(f"\nTotal Stage-1 negatives written: {len(rows_out)}")
    if rows_out:
        print(pd.DataFrame(rows_out)["negative_strategy"].value_counts())
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
