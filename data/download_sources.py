#!/usr/bin/env python3
"""
download_sources.py
====================

Pulls / organizes every raw data source used by the Nirikshan two-stage
lesion classifier into a consistent on-disk layout:

    data/raw/<source_name>/images/*.jpg
    data/raw/<source_name>/manifest.csv   (image_path, source, label, patient_id)

Two families of sources are handled:

1. ISIC-family sources that are pullable programmatically via the official
   `isic-cli` tool: ISIC Archive, ISIC 2020 Challenge, BCN20000 (BCN20000 is
   distributed as an ISIC-Archive collection, so it is also isic-cli-pullable).

2. Sources that do NOT have a public API / CLI and must be downloaded
   manually by a human (license click-through, gated request form, etc.):
   HAM10000, PAD-UFES-20, Fitzpatrick17k, DDI, MCSI. For these, this script
   does NOT attempt any network calls -- it prints step-by-step manual
   instructions and then, if the expected raw files are already present
   (because a human followed the instructions), organizes them into the
   consistent layout above and builds the manifest.

Consistent manifest schema (all sources write to this schema):
    image_path   -- path to image file, relative to data/raw/<source>/images/
    source       -- string source name, e.g. "isic_archive", "ham10000"
    label        -- diagnosis label string as provided by the source
                     (label harmonization to the 5 Stage-2 classes happens
                     later in prepare_dataset.py, not here -- this script
                     preserves the source's own label vocabulary)
    patient_id   -- patient/lesion id if the source provides one, else ""

Usage
-----
    # Sanity-check args/paths for every source without touching the network
    # or filesystem at all (recommended first run):
    python download_sources.py --dry-run

    # Pull everything isic-cli can pull:
    python download_sources.py --sources isic_archive isic2020 bcn20000 \
        --output-dir data/raw

    # Organize a manually-downloaded source that's already on disk:
    python download_sources.py --sources ham10000 \
        --manual-source-dir /path/to/already/downloaded/HAM10000 \
        --output-dir data/raw

    # Just print manual download instructions for every non-CLI source:
    python download_sources.py --sources ham10000 pad_ufes_20 \
        fitzpatrick17k ddi mcsi --instructions-only

--dry-run vs --instructions-only
---------------------------------
    --instructions-only prints manual-download instructions for manual
    sources and does nothing at all for CLI sources (not even a preview).

    --dry-run covers every source, CLI and manual alike, and never touches
    the network or filesystem: for CLI sources it prints the exact `isic`
    commands that would be run (without running them); for manual sources
    it prints the same instructions --instructions-only would, plus, if
    --manual-source-dir is also given, a preview of what would be copied/
    manifested (file count found, no copying performed). Use --dry-run to
    sanity-check the whole invocation -- flags, paths, which sources are
    selected -- before committing to a real, possibly multi-hour pull.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------
# "cli" sources are pulled via isic-cli. "manual" sources require a human
# to download first (license gate, request form, etc.) -- this script only
# organizes them into the consistent layout once the raw files exist.

CLI_SOURCES = {"isic_archive", "isic2020", "bcn20000"}

MANUAL_SOURCES = {"ham10000", "pad_ufes_20", "fitzpatrick17k", "ddi", "mcsi"}

ALL_SOURCES = CLI_SOURCES | MANUAL_SOURCES

# isic-cli collection IDs per source. Collections are selected with the
# `--collections <id>` flag -- NOT with a `collections:"..."` term inside
# `--search` (that is not a valid search field and isic-cli rejects it with
# "Invalid search query string"). Confirm/refresh IDs any time with:
#     isic collection list
ISIC_CLI_COLLECTION_IDS = {
    "isic_archive": None,  # no collection; REQUIRES --isic-search (see below)
    "isic2020": 70,        # ISIC 2020 Challenge training set
    "bcn20000": 249,       # BCN20000
}

MANUAL_INSTRUCTIONS = {
    "ham10000": """
    HAM10000 ("Human Against Machine with 10000 training images")
    ----------------------------------------------------------------
    1. Go to the Harvard Dataverse listing:
       https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    2. Click "Access Dataset" -> "Original Format ZIP" (requires free
       Harvard Dataverse account + accepting the dataset's terms of use).
    3. Download HAM10000_images_part_1.zip, HAM10000_images_part_2.zip,
       and HAM10000_metadata.csv (or HAM10000_metadata.tab).
    4. Unzip both image parts into a single folder, e.g.:
           /path/to/HAM10000/images/*.jpg
           /path/to/HAM10000/HAM10000_metadata.csv
    5. Re-run this script with:
           --sources ham10000 --manual-source-dir /path/to/HAM10000
    """,
    "pad_ufes_20": """
    PAD-UFES-20
    ----------------------------------------------------------------
    1. Go to the Mendeley Data listing:
       https://data.mendeley.com/datasets/zr7vgbcyr2
    2. Click "Download All" (no login required, CC BY 4.0 license).
    3. Unzip. Expected layout after unzip:
           /path/to/PAD-UFES-20/images/*.png
           /path/to/PAD-UFES-20/metadata.csv
    4. Re-run this script with:
           --sources pad_ufes_20 --manual-source-dir /path/to/PAD-UFES-20
    """,
    "fitzpatrick17k": """
    Fitzpatrick17k
    ----------------------------------------------------------------
    1. Clone the official metadata repo (images are hotlinked from
       multiple public atlases, not hosted directly):
           git clone https://github.com/mattgroh/fitzpatrick17k.git
    2. Use the CSV in that repo (fitzpatrick17k.csv) which contains
       image URLs + Fitzpatrick skin-type labels + diagnosis labels.
    3. Download the actual images from the URLs in that CSV (a small
       fraction are dead links -- that's expected and documented
       upstream; skip failures).
       Save all images into:
           /path/to/Fitzpatrick17k/images/*.jpg
       and keep fitzpatrick17k.csv alongside it as:
           /path/to/Fitzpatrick17k/fitzpatrick17k.csv
    4. Re-run this script with:
           --sources fitzpatrick17k --manual-source-dir /path/to/Fitzpatrick17k
    """,
    "ddi": """
    DDI (Diverse Dermatology Images)
    ----------------------------------------------------------------
    1. Go to the Stanford AIMI listing:
       https://ddi-dataset.github.io/  (or the Stanford AIMI Shared
       Datasets page) and request/accept the data use agreement.
    2. Download the DDI images + ddi_metadata.csv.
    3. Expected layout:
           /path/to/DDI/images/*.png
           /path/to/DDI/ddi_metadata.csv
    4. Re-run this script with:
           --sources ddi --manual-source-dir /path/to/DDI
    """,
    "mcsi": """
    MCSI (healthy-skin subset used for Stage-1 negatives)
    ----------------------------------------------------------------
    1. Obtain the MCSI dataset per your team's existing data-sharing
       agreement / source (this is the same source already used by the
       proven prototype -- reuse that download, do not re-source it).
    2. Expected layout:
           /path/to/MCSI/images/*.jpg
           /path/to/MCSI/metadata.csv   (optional; if absent, all images
                                          are treated as label="healthy")
    3. Re-run this script with:
           --sources mcsi --manual-source-dir /path/to/MCSI
    """,
}


def write_manifest(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "source", "label", "patient_id"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  -> wrote {len(rows)} rows to {out_path}")


def pull_isic_source(source: str, output_dir: Path, dry_run: bool = False,
                     search: str = None):
    """Pull an ISIC-family source via isic-cli and build its manifest.

    `search` is an optional ISIC search string (e.g.
    'diagnosis_3:"Nevus" OR diagnosis_3:"Solar or actinic keratosis"')
    applied on top of the source's collection filter."""
    dest = output_dir / source
    images_dir = dest / "images"

    filters = []
    coll_id = ISIC_CLI_COLLECTION_IDS[source]
    if coll_id is not None:
        filters += ["--collections", str(coll_id)]
    if search:
        filters += ["--search", search]

    if not filters:
        # No collection and no search == "download the entire ISIC Archive"
        # (hundreds of thousands of images, tens of GB). Never do that by
        # accident -- ISIC2020 and BCN20000 are already subsets of it anyway.
        print(f"  !! [{source}] refusing to pull the ENTIRE ISIC Archive. "
              "Pass --isic-search '<query>' to pull a filtered slice, e.g.\n"
              "       --sources isic_archive --isic-search "
              "'diagnosis_3:\"Squamous cell carcinoma\"'\n"
              "     (find exact diagnosis strings by running "
              "`isic metadata download` and inspecting the diagnosis_3 column). "
              "Skipping.")
        return

    cmd = ["isic", "image", "download", *filters, str(images_dir)]

    meta_path = dest / "isic_metadata.csv"
    meta_cmd = ["isic", "metadata", "download", *filters, "-o", str(meta_path)]

    if dry_run:
        print(f"[{source}] (dry-run) would create: {images_dir}")
        print(f"[{source}] (dry-run) would run: {' '.join(cmd)}")
        print(f"[{source}] (dry-run) would run: {' '.join(meta_cmd)}")
        existing = list(images_dir.glob("*")) if images_dir.exists() else []
        if existing:
            print(f"[{source}] (dry-run) {len(existing)} images already "
                  f"present at {images_dir} -- would be re-manifested, not "
                  "re-downloaded (isic-cli skips existing files by default).")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{source}] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "  !! isic-cli not found. Install with: pip install isic-cli\n"
            "  Skipping download; if images already exist under "
            f"{images_dir} they will still be manifested below."
        )
    except subprocess.CalledProcessError as e:
        print(f"  !! isic-cli exited with error ({e}); continuing to manifest "
              "whatever was already downloaded.")

    # isic-cli also writes a metadata.csv alongside the images when given
    # a `-m` flag; to keep this script self-contained we additionally
    # pull metadata explicitly so labels are available.
    try:
        subprocess.run(meta_cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  !! could not fetch isic metadata CSV; labels will be blank "
              "for this source until it's fetched manually.")

    # Build manifest: join image files present on disk against metadata
    # (isic_id -> diagnosis) when available. A metadata pull can "succeed"
    # at the subprocess level but still produce an empty/zero-byte file
    # (isic-cli version differences, an empty result set, a transient API
    # hiccup) -- treat that the same as "no metadata available" rather
    # than crashing the whole run over it.
    labels_by_id = {}
    if meta_path.exists() and meta_path.stat().st_size > 0:
        import pandas as pd  # local import: only needed on this path
        try:
            meta_df = pd.read_csv(meta_path)
            if meta_df.empty:
                raise pd.errors.EmptyDataError("metadata CSV has a header but no rows")
            id_col = "isic_id" if "isic_id" in meta_df.columns else meta_df.columns[0]
            label_col = None
            # Newer ISIC taxonomy: diagnosis_1 is only Benign/Malignant, so
            # prefer the granular levels first (diagnosis_3 holds names like
            # "Nevus"; check value_counts() on your download to confirm).
            for cand in ("diagnosis_3", "diagnosis", "diagnosis_2",
                         "diagnosis_1", "benign_malignant"):
                if cand in meta_df.columns:
                    label_col = cand
                    break
            if label_col:
                labels_by_id = dict(zip(meta_df[id_col], meta_df[label_col]))
            else:
                print(f"  !! {meta_path.name} has no recognized diagnosis "
                      "column; labels will be blank for this source.")
        except pd.errors.EmptyDataError:
            print(f"  !! {meta_path.name} is empty; labels will be blank "
                  "for this source until metadata is fetched successfully.")
        except Exception as e:  # noqa: BLE001
            print(f"  !! could not parse {meta_path.name} ({e}); labels "
                  "will be blank for this source.")
    elif meta_path.exists():
        print(f"  !! {meta_path.name} is a zero-byte file; labels will be "
              "blank for this source until metadata is fetched successfully.")
    else:
        print(f"  !! no metadata file at {meta_path}; labels will be blank "
              "for this source.")

    rows = []
    image_files = sorted(
        p for p in images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for img in tqdm(image_files, desc=f"[{source}] manifesting"):
        image_id = img.stem
        rows.append(
            {
                "image_path": str(img.relative_to(dest)),
                "source": source,
                "label": labels_by_id.get(image_id, ""),
                "patient_id": "",
            }
        )
    write_manifest(rows, dest / "manifest.csv")


def organize_manual_source(source: str, manual_source_dir: Path, output_dir: Path,
                            dry_run: bool = False):
    """Copy a manually-downloaded source into the consistent raw layout
    and build its manifest, using source-specific metadata parsing."""
    import pandas as pd  # local import: only needed on this path

    dest = output_dir / source
    images_dir = dest / "images"

    src_images_dir = manual_source_dir / "images"
    if not src_images_dir.exists():
        # some sources (e.g. HAM10000's two zip parts) may already be
        # extracted directly under manual_source_dir
        src_images_dir = manual_source_dir

    image_files = [
        p
        for p in src_images_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    ]
    if not image_files:
        print(f"  !! no images found under {src_images_dir}; "
              "did you follow the manual download instructions?")
        return

    if dry_run:
        print(f"[{source}] (dry-run) found {len(image_files)} images under "
              f"{src_images_dir}")
        print(f"[{source}] (dry-run) would copy them into: {images_dir}")
        print(f"[{source}] (dry-run) would write manifest to: {dest / 'manifest.csv'}")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{source}] copying {len(image_files)} images into {images_dir}")
    for img in tqdm(image_files, desc=f"[{source}] copying"):
        target = images_dir / img.name
        if not target.exists():
            shutil.copy2(img, target)

    # Source-specific metadata parsing
    label_map, patient_map = {}, {}
    if source == "ham10000":
        meta_csv = next(manual_source_dir.glob("HAM10000_metadata*"), None)
        if meta_csv:
            df = pd.read_csv(meta_csv)
            label_map = dict(zip(df["image_id"], df["dx"]))
            patient_map = dict(zip(df["image_id"], df.get("lesion_id", "")))
    elif source == "pad_ufes_20":
        meta_csv = manual_source_dir / "metadata.csv"
        if meta_csv.exists():
            df = pd.read_csv(meta_csv)
            key_col = "img_id" if "img_id" in df.columns else df.columns[0]
            label_map = dict(zip(df[key_col].astype(str).str.replace(
                r"\.(png|jpg|jpeg)$", "", regex=True), df.get("diagnostic", "")))
            patient_map = dict(zip(df[key_col].astype(str).str.replace(
                r"\.(png|jpg|jpeg)$", "", regex=True), df.get("patient_id", "")))
    elif source == "fitzpatrick17k":
        meta_csv = manual_source_dir / "fitzpatrick17k.csv"
        if meta_csv.exists():
            df = pd.read_csv(meta_csv)
            # md5hash is the conventional filename key used by this dataset
            key_col = "md5hash" if "md5hash" in df.columns else df.columns[0]
            label_map = dict(zip(df[key_col], df.get("label", "")))
    elif source == "ddi":
        meta_csv = manual_source_dir / "ddi_metadata.csv"
        if meta_csv.exists():
            df = pd.read_csv(meta_csv)
            key_col = "DDI_file" if "DDI_file" in df.columns else df.columns[0]
            label_map = dict(zip(
                df[key_col].astype(str).str.replace(r"\.(png|jpg|jpeg)$", "", regex=True),
                df.get("disease", ""),
            ))
    elif source == "mcsi":
        meta_csv = manual_source_dir / "metadata.csv"
        if meta_csv.exists():
            df = pd.read_csv(meta_csv)
            key_col = df.columns[0]
            label_map = dict(zip(df[key_col].astype(str), df.get("label", "healthy")))

    rows = []
    for img in sorted(images_dir.glob("*")):
        stem = img.stem
        label = label_map.get(stem, "healthy" if source == "mcsi" else "")
        patient_id = patient_map.get(stem, "")
        rows.append(
            {
                "image_path": str(img.relative_to(dest)),
                "source": source,
                "label": label,
                "patient_id": patient_id,
            }
        )
    write_manifest(rows, dest / "manifest.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", choices=sorted(ALL_SOURCES),
                         default=sorted(ALL_SOURCES),
                         help="Which sources to process.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"),
                         help="Root dir to write data/raw/<source>/ into.")
    parser.add_argument("--manual-source-dir", type=Path, default=None,
                         help="Path to an already-downloaded manual source "
                              "(required for exactly one manual source per run).")
    parser.add_argument("--isic-search", type=str, default=None,
                         help="Optional ISIC search query applied to the CLI "
                              "source(s) being pulled, on top of their collection "
                              "filter. REQUIRED for --sources isic_archive. "
                              "Best used with ONE source per run. Example: "
                              "--isic-search 'diagnosis_3:\"Nevus\"'")
    parser.add_argument("--instructions-only", action="store_true",
                         help="Print manual-download instructions for manual "
                              "sources only; does nothing for CLI sources. "
                              "See --dry-run for a preview that covers both.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview what every selected source would do "
                              "(commands that would run, files that would be "
                              "copied/manifested) without touching the "
                              "network or filesystem. Safe to run anytime.")
    args = parser.parse_args()

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        print(f"\n=== {source} ===")
        if source in MANUAL_SOURCES:
            print(MANUAL_INSTRUCTIONS[source])
            if args.instructions_only and not args.dry_run:
                continue
            if args.manual_source_dir is None:
                print("  (skipping organize step: pass --manual-source-dir "
                      "once you've downloaded this source)")
                continue
            organize_manual_source(source, args.manual_source_dir, args.output_dir,
                                    dry_run=args.dry_run)
        elif source in CLI_SOURCES:
            if args.instructions_only and not args.dry_run:
                print("  (CLI-pullable source; nothing to print)")
                continue
            pull_isic_source(source, args.output_dir, dry_run=args.dry_run,
                             search=args.isic_search)

    if args.dry_run:
        print("\nDry run complete. Nothing was downloaded, copied, or written.")
        print("Re-run without --dry-run to actually pull/organize the data.")
    else:
        print("\nDone. Per-source manifests are at data/raw/<source>/manifest.csv")
        print("Run prepare_dataset.py next to merge, dedupe, and split them.")


if __name__ == "__main__":
    sys.exit(main())
