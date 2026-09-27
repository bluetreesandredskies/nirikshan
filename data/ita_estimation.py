#!/usr/bin/env python3
"""
ita_estimation.py
==================

Estimates the Individual Typology Angle (ITA) for every image in a manifest
and buckets each image into one of the standard 6 ITA skin-tone bins. Adds
two columns to the manifest: `ita_value` (float, degrees) and `ita_bin`
(one of the 6 bin names below).

ITA formula (standard, pixel-based):
    ITA = atan( (L* - 50) / b* ) * 180 / pi
where L* and b* are computed by sampling non-lesion skin pixels from the
image and converting their mean RGB to CIELAB.

Standard 6-bin ITA convention used here (values in degrees):
    very_light     : ITA > 55
    light          : 41 < ITA <= 55
    intermediate   : 28 < ITA <= 41
    tan            : 10 < ITA <= 28
    dark           : -30 < ITA <= 10
    very_dark      : ITA <= -30

Sampling strategy for "non-lesion skin pixels":
    - If a lesion bounding box (columns `bbox_x`, `bbox_y`, `bbox_w`,
      `bbox_h`, all in [0,1] normalized coordinates) is present in the
      manifest, pixels are sampled from a border margin OUTSIDE that box.
    - If a segmentation mask path is present (column `mask_path`), pixels
      where the mask == 0 (non-lesion) are sampled instead -- this is more
      accurate than the bbox fallback and is used preferentially.
    - If neither is available (e.g. images that are already tightly
      cropped to a single lesion, as in PAD-UFES-20 or Fitzpatrick17k),
      the four image corners (10% margin) are sampled as a best-effort
      proxy for surrounding skin.
    - EVERY sampling path (mask, bbox-border, and corner fallback) filters
      out near-black and near-white pixels before computing ITA. This
      matters most for the corner fallback: dermoscopy images (BCN20000,
      ISIC-family) are near-circular photos with a black vignette ring
      in the corners, which without filtering reads as extremely dark
      ITA regardless of actual skin tone -- a real bug hit and fixed
      during this project (see git history / CHANGELOG). If filtering
      leaves too few pixels (<200), the row is left with an EMPTY ita_bin
      rather than a guessed value, and logged as
      "no_skin_pixels_after_vignette_filter" in the run summary. This is
      still a known limitation for tightly-cropped, heavily-vignetted
      images with no bbox/mask -- there may be nothing real to sample.
      These rows should be reviewed or excluded (or backfilled from a
      manifest that has bbox/mask info) before stratified splitting in
      prepare_dataset.py.

Usage
-----
    python ita_estimation.py \\
        --manifest data/raw/isic_archive/manifest.csv \\
        --images-root data/raw/isic_archive \\
        --output-manifest data/raw/isic_archive/manifest_with_ita.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from tqdm import tqdm

BIN_ORDER = ["very_light", "light", "intermediate", "tan", "dark", "very_dark"]

# Dermoscopy images are almost always a circular photo through a dermatoscope
# lens, with a solid black vignette ring (and sometimes white calibration
# rulers/stickers) outside the lesion circle. On sources with no bbox/mask
# (e.g. BCN20000), sample_skin_pixels() falls back to the four image corners
# -- which for dermoscopy images means it's very likely sampling that black
# ring, not skin, and will incorrectly read as extremely dark ITA. This
# filters those non-skin pixels out before ITA is computed.
VIGNETTE_MIN_MEAN = 25    # mean RGB below this ~= black vignette/ring, not skin
VIGNETTE_MAX_MEAN = 240   # mean RGB above this ~= white calibration marker/glare
MIN_VALID_PIXELS = 200    # below this, we don't trust the sample at all


def _filter_non_skin_pixels(pixels: np.ndarray) -> np.ndarray:
    """Drops near-black and near-white pixels (vignette ring / calibration
    stickers / blown-out glare) from a raw RGB pixel array."""
    if pixels.size == 0:
        return pixels
    means = pixels.astype(np.float64).mean(axis=1)
    keep = (means > VIGNETTE_MIN_MEAN) & (means < VIGNETTE_MAX_MEAN)
    return pixels[keep]


def ita_to_bin(ita: float) -> str:
    if ita > 55:
        return "very_light"
    elif ita > 41:
        return "light"
    elif ita > 28:
        return "intermediate"
    elif ita > 10:
        return "tan"
    elif ita > -30:
        return "dark"
    else:
        return "very_dark"


def has_localization(row: pd.Series) -> bool:
    """True if this row has a real mask or bbox to sample non-lesion skin
    from. False means there is no trustworthy way to isolate skin pixels
    from lesion/vignette/background pixels for this image."""
    mask_path = row.get("mask_path", "")
    if isinstance(mask_path, str) and mask_path and Path(mask_path).exists():
        return True
    bbox_cols = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    if all(c in row.index for c in bbox_cols) and not pd.isna(row.get("bbox_x")):
        return True
    return False


def sample_skin_pixels(img: np.ndarray, row: pd.Series, margin_frac: float = 0.08) -> np.ndarray:
    """Return an (N, 3) array of RGB skin pixels sampled from outside the
    lesion, using whichever localization info the row provides.

    NOTE: this is only called when has_localization(row) is True. There is
    intentionally NO corner-patch fallback for rows with neither a mask nor
    a bbox: dermoscopy images (BCN20000, ISIC-family) are near-circular
    photos with a black vignette ring in the corners, and non-dermoscopy
    corners often hold rulers/calibration charts/background -- none of
    that is skin, and "not black and not white" is not the same test as
    "is skin". Guessing from those corners produced a bimodal, physically
    implausible ITA distribution during this project (see CHANGELOG) even
    after filtering out pure black/white pixels. Rows with no
    localization info get ita_bin="unknown" instead (same bucket
    prepare_dataset.py already uses for Stage-1 negatives) -- this is an
    honest gap, not a guess, and should be backfilled from sources that
    ship real Fitzpatrick-type labels (e.g. Fitzpatrick17k, DDI) rather
    than by re-tuning pixel thresholds here."""
    h, w = img.shape[:2]

    mask_path = row.get("mask_path", "")
    if isinstance(mask_path, str) and mask_path and Path(mask_path).exists():
        mask = np.array(Image.open(mask_path).convert("L").resize((w, h)))
        skin_pixels = img[mask < 128]  # mask==0/low => non-lesion skin
        skin_pixels = _filter_non_skin_pixels(skin_pixels)
        if skin_pixels.shape[0] >= MIN_VALID_PIXELS:
            return skin_pixels

    bbox_cols = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    if all(c in row.index for c in bbox_cols) and not pd.isna(row["bbox_x"]):
        bx, by, bw, bh = (float(row[c]) for c in bbox_cols)
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = int((bx + bw) * w), int((by + bh) * h)
        border = np.ones((h, w), dtype=bool)
        border[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = False
        # also exclude the outermost image edge (vignetting / dermoscope ring)
        m = int(min(h, w) * margin_frac)
        edge_mask = np.zeros((h, w), dtype=bool)
        edge_mask[m:h - m, m:w - m] = True
        skin_pixels = img[border & edge_mask]
        skin_pixels = _filter_non_skin_pixels(skin_pixels)
        if skin_pixels.shape[0] >= MIN_VALID_PIXELS:
            return skin_pixels

    # No corner-patch fallback: see the docstring above for why. If mask/bbox
    # sampling above didn't return (either absent, or too few valid pixels
    # after filtering), there is nothing trustworthy left to sample.
    return np.empty((0, 3), dtype=img.dtype)


def compute_ita_for_image(image_path: Path, row: pd.Series):
    if not has_localization(row):
        # No mask, no bbox: there is no trustworthy way to isolate skin
        # pixels for this image. Tag it "unknown" (not a guessed bin) --
        # same bucket prepare_dataset.py uses for Stage-1 negatives, so
        # stratification still works, it just isn't used for the
        # per-skin-tone fairness breakdown.
        return None, "unknown", "no_localization_info"

    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        return None, None, f"read_error: {e}"

    pixels = sample_skin_pixels(img, row)
    if pixels.size == 0:
        return None, None, "no_skin_pixels_after_vignette_filter"

    # Downsample for speed if huge
    if pixels.shape[0] > 20000:
        idx = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[idx]

    lab = rgb2lab(pixels.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)
    L_mean, b_mean = lab[:, 0].mean(), lab[:, 2].mean()

    if abs(b_mean) < 1e-6:
        return None, None, "degenerate_b_channel"

    ita = float(np.degrees(np.arctan((L_mean - 50.0) / b_mean)))
    return ita, ita_to_bin(ita), None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                         help="Input manifest CSV (must have image_path column, "
                              "optionally bbox_x/bbox_y/bbox_w/bbox_h or mask_path).")
    parser.add_argument("--images-root", type=Path, required=True,
                         help="Root directory image_path values are relative to.")
    parser.add_argument("--output-manifest", type=Path, required=True,
                         help="Where to write the manifest with ita_value/ita_bin added.")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "image_path" not in df.columns:
        print("ERROR: manifest must have an image_path column.", file=sys.stderr)
        sys.exit(1)

    ita_values, ita_bins, errors = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Estimating ITA"):
        img_path = args.images_root / row["image_path"]
        if not img_path.exists():
            ita_values.append(np.nan)
            ita_bins.append("")
            errors.append("file_missing")
            continue
        ita, bin_name, err = compute_ita_for_image(img_path, row)
        ita_values.append(ita if ita is not None else np.nan)
        ita_bins.append(bin_name if bin_name is not None else "")
        errors.append(err or "")

    df["ita_value"] = ita_values
    df["ita_bin"] = ita_bins

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_manifest, index=False)

    n_real_bins = df["ita_bin"].isin(BIN_ORDER).sum()
    n_unknown = (df["ita_bin"] == "unknown").sum()
    n_missing = (df["ita_bin"] == "").sum()
    print(f"\n{n_real_bins}/{len(df)} images got a real ITA-based skin-tone bin.")
    print(f"{n_unknown}/{len(df)} images had no mask/bbox to sample from and "
          "are tagged 'unknown' (honest gap, not a guess -- see module "
          "docstring; these still count for class stratification, just not "
          "for the per-skin-tone fairness breakdown).")
    print(f"{n_missing}/{len(df)} images had a real error (missing file, "
          "read error, or a mask/bbox that yielded too few real skin pixels "
          "after vignette filtering) and were left with an empty ita_bin -- "
          "these should be reviewed or excluded before stratified splitting "
          "in prepare_dataset.py.")
    print("\nReal-bin distribution (excludes 'unknown' and empty rows):")
    print(df["ita_bin"].value_counts().reindex(BIN_ORDER, fill_value=0))
    print(f"\nWrote {args.output_manifest}")


if __name__ == "__main__":
    main()
