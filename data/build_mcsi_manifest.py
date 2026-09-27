"""
data/build_mcsi_manifest.py

Builds Stage 1's MCSI-sourced negative-class manifest rows, filtered to the
REAL per-image label in MCSI's own metadata.csv -- not by folder name.

Why this version exists: an earlier version of this script assumed every
image file found under a folder named "healthy" was a healthy/no_lesion
image. That assumption was wrong for this extraction -- the folder (locally
named "healthy" during setup, but actually the root of the full MCSI.zip
extraction) contains all 400 images across all 4 MCSI classes (monkeypox,
chickenpox, normal, acne) mixed together in one images/ folder, with the
real per-image class living in metadata.csv's `diagnostic` column, not in
any folder path. The earlier version silently labeled all 400 -- including
100 real monkeypox lesion photos and 100 chickenpox lesion photos -- as
"no_lesion". This version fixes that by reading metadata.csv directly.

Per MCSI's own README.md, `diagnostic` takes these values (100 rows each):
    monkeypox, chickenpox, normal, acne_l1/acne_l2/acne_l3 (acne, 3 levels)
Only "normal" is a true negative for Stage 1 -- monkeypox and chickenpox are
themselves visible skin lesions/rashes and must NOT be labeled no_lesion.
Acne is also excluded here (it's a lesion of sorts, and PROJECT_CONTEXT's
Stage 1 negative-sourcing strategy doesn't call for it) but you can widen
`NEGATIVE_DIAGNOSTIC_VALUES` below if the team decides otherwise.
"""

import argparse
from pathlib import Path

import pandas as pd

# Only this MCSI diagnostic value is a real Stage-1 negative.
NEGATIVE_DIAGNOSTIC_VALUES = {"normal"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcsi-extraction-dir",
        type=str,
        default="data/raw/mcsi/healthy",
        help=(
            "The folder containing MCSI's own metadata.csv, README.md, and "
            "images/ subfolder -- i.e. wherever the MCSI.zip was actually "
            "extracted to, regardless of what it's locally named."
        ),
    )
    parser.add_argument(
        "--mcsi-root",
        type=str,
        default="data/raw/mcsi",
        help="Root that image_path in the output manifest will be relative to.",
    )
    parser.add_argument("--output-manifest", type=str, default="data/raw/mcsi/manifest.csv")
    args = parser.parse_args()

    extraction_dir = Path(args.mcsi_extraction_dir)
    mcsi_root = Path(args.mcsi_root)
    images_dir = extraction_dir / "images"
    metadata_path = extraction_dir / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Pass --mcsi-extraction-dir "
            f"pointing at the folder that directly contains metadata.csv."
        )

    meta = pd.read_csv(metadata_path)
    for col in ("diagnostic", "img_id"):
        if col not in meta.columns:
            raise ValueError(
                f"{metadata_path} is missing expected column '{col}'. "
                f"Found columns: {meta.columns.tolist()}"
            )

    print(f"{metadata_path}: {len(meta)} total rows")
    print("diagnostic value counts:\n", meta["diagnostic"].value_counts().to_string())

    negative_rows = meta[meta["diagnostic"].isin(NEGATIVE_DIAGNOSTIC_VALUES)].copy()
    print(f"\n{len(negative_rows)} rows match negative diagnostic value(s) {NEGATIVE_DIAGNOSTIC_VALUES}")

    out_rows = []
    missing = []
    for img_id in negative_rows["img_id"]:
        candidate = images_dir / str(img_id)
        if not candidate.exists():
            missing.append(img_id)
            continue
        out_rows.append(
            {
                "image_path": str(candidate.relative_to(mcsi_root)).replace("\\", "/"),
                "source": "mcsi",
                "label": "no_lesion",
                "patient_id": None,
            }
        )

    if missing:
        preview = missing[:10]
        print(
            f"\nWARNING: {len(missing)} 'normal' img_id(s) from metadata.csv "
            f"had no matching file under {images_dir}: {preview}"
            f"{' ...' if len(missing) > len(preview) else ''}"
        )

    out_df = pd.DataFrame(out_rows)
    output_path = Path(args.output_manifest)
    out_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {output_path}")

    if len(out_df) != 100:
        print(
            f"NOTE: expected 100 rows per MCSI's own README (100 samples per "
            f"label), got {len(out_df)}. Worth a quick look at the WARNING "
            f"above (if any) or the raw file count under {images_dir} before "
            f"trusting this manifest."
        )


if __name__ == "__main__":
    main()
