"""
model/stage1_lesion_presence.py

Nirikshan — Stage 1: Lesion Presence Detector.

Binary classifier: "is there a lesion in this image at all?" (label 1) vs.
"normal / healthy skin" (label 0). Trained on a heterogeneous mix of true
positives (any lesion, any disease class) and true negatives (cropped
normal-skin patches from wide-field source images, the MCSI healthy subset,
and team-collected consented photos) — see PROJECT_CONTEXT.md §5.

This exists as its own stage (rather than folding "healthy" into a single
5-way softmax) because a single softmax forced with too few true negatives
learns spurious cues (framing/lighting/dataset fingerprints) instead of
actual skin texture. Stage 2 (disease classification) only ever runs on
images this stage flags as "lesion present."

Expected input: a CSV manifest (as produced by data/prepare_dataset.py) with
at minimum these columns:
    image_path   -- path to a 224x224 (or larger; we resize) RGB image
    label        -- the disease/class string for that row, e.g.
                     "squamous_cell_carcinoma", "actinic_keratosis", "nevus",
                     "seborrheic_keratosis", or (once added) a negative-class
                     label such as "no_lesion"
    ita_bin      -- (optional) Individual Typology Angle skin-tone bucket,
                     one of the 6 standard ITA bins, or the literal string
                     "unknown" for rows without a usable ITA estimate (per
                     the data pipeline notes: PAD-UFES-20 gives this for
                     ~6% of rows via Fitzpatrick type, BCN20000 has none).
                     "unknown" is a normal, valid bucket here, not an error
                     or a row to drop.

IMPORTANT -- Stage 1 is a binary task (lesion present / not present), but
the manifest's `label` column is NOT already binary: it holds whatever
disease class string prepare_dataset.py assigned to that row, and there is
currently no single fixed literal that always means "negative." This module
therefore does NOT hardcode which strings are negative. Instead:
  - `--negative-labels` (default: "no_lesion") names the label string(s)
    that should be treated as class 0. Everything else in the manifest is
    treated as class 1 (lesion present).
  - At dataset-construction time, the actual label strings present in the
    manifest are inspected and recorded (not assumed), and stored under
    `metadata["source_negative_labels"]` / `metadata["source_positive_labels"]`
    in every checkpoint, so there's a real audit trail of what this specific
    training run actually saw -- this is the "store class_names from the
    data itself" requirement.

KNOWN CURRENT DATA GAP (as of the team's latest pipeline notes): the
training/val/test manifests presently contain ZERO negative-class rows --
`label` is only ever one of the four disease classes, because the Stage-1
negative sources (MCSI healthy subset, team-consented photo drive,
cropped-patch generation) haven't been run/merged into prepare_dataset.py's
output yet. A binary classifier cannot be trained with only one class
present. Rather than silently training a degenerate single-class model (or
crashing on an unrelated error deep in the training loop), this script
FAILS LOUDLY at dataset-construction time with a specific, actionable error
message if no row's label is in `--negative-labels`. Do not remove or
weaken that check -- see `LesionPresenceDataset.__init__`.

Sampling: uses the weighted sampler from data/sampler.py so that the True
Negative sources (which are structurally smaller / more homogeneous than the
positive pool) don't get drowned out. If data/sampler.py is not importable
in this environment, a local fallback is used so this file remains runnable
standalone; the fallback also incorporates `ita_bin` (treating "unknown" as
an ordinary bucket) when that column is present, so skin-tone balancing
isn't silently dropped just because the shared sampler module is missing.

Checkpointing: every epoch, unconditionally, to
    model/checkpoints/stage1_epoch{N}.pt
Each checkpoint stores BOTH `model_state_dict` and a `metadata` dict
(class name order, image size, normalization stats, epoch, val macro-F1).
Downstream code (model/inference.py) must always read class order from this
metadata and never hardcode it — see PROJECT_CONTEXT.md Part J / §8.

Early stopping: patience=4, monitored on validation macro-F1 (NOT accuracy),
because with a small "healthy" pool accuracy is easy to game by predicting
the majority class.

Augmentation: horizontal flip, +/-20 degree rotation, brightness/contrast
jitter ONLY. No hue/saturation jitter, ever — it distorts diagnostically
relevant skin-tone and lesion-color information (PROJECT_CONTEXT.md §5, §8).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required for stage1_lesion_presence.py. "
        "pip install -r requirements-model.txt"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [stage1] %(levelname)s %(message)s",
)
logger = logging.getLogger("stage1_lesion_presence")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["no_lesion", "lesion_present"]  # index 0 / 1, fixed order
BACKBONE_NAME = "efficientnet_b0"


# ---------------------------------------------------------------------------
# Weighted sampler: prefer data/sampler.py, fall back to a local equivalent
# ---------------------------------------------------------------------------

def _get_weighted_sampler(
    labels: np.ndarray,
    ita_bins: Optional[np.ndarray] = None,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that inversely weights samples by class
    frequency (and, when available, by ITA skin-tone bin) so the minority
    class/bin is not drowned out during training.

    Tries to reuse the shared implementation in data/sampler.py first
    (expected signature: get_weighted_sampler(labels: np.ndarray) -> Sampler
    -- if that module's real signature also takes ITA bins or source names
    once it's written in Session 3, update this call site to match it).
    Falls back to an equivalent local implementation if that module isn't
    importable in this environment, so this script stays runnable standalone.

    The fallback treats every distinct value in `ita_bins` -- including the
    literal string "unknown" -- as its own ordinary stratification bucket.
    "unknown" is expected (BCN20000 rows have no ITA estimate at all) and is
    never filtered out or special-cased as an error.
    """
    try:
        from data.sampler import get_weighted_sampler  # type: ignore

        logger.info("Using shared weighted sampler from data/sampler.py")
        return get_weighted_sampler(labels)
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        logger.warning(
            "Could not import data.sampler.get_weighted_sampler (%s). "
            "Falling back to a local equivalent weighted sampler.",
            exc,
        )
        labels_int = labels.astype(int)

        if ita_bins is None:
            # Class-frequency-only weighting.
            class_counts = np.bincount(labels_int)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[labels_int]
        else:
            # Joint (class, ita_bin) weighting -- e.g. a "lesion_present"
            # sample from a well-represented ITA bin gets downweighted
            # relative to a "lesion_present" sample from a rare/"unknown"
            # bin, and likewise across classes.
            strata = list(zip(labels_int.tolist(), ita_bins.tolist()))
            unique_strata, inverse, counts = np.unique(
                strata, axis=0, return_inverse=True, return_counts=True
            )
            stratum_weights = 1.0 / np.maximum(counts, 1)
            sample_weights = stratum_weights[inverse]
            logger.info(
                "Weighted sampler strata (class, ita_bin) -> count: %s",
                {tuple(s): int(c) for s, c in zip(unique_strata.tolist(), counts.tolist())},
            )

        return torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LesionPresenceDataset(Dataset):
    """
    Reads a manifest CSV with a raw `label` column (disease class string, or
    a negative-class string once negatives exist) and binarizes it on the
    fly: rows whose label is in `negative_labels` -> 0 ("no lesion"),
    everything else -> 1 ("lesion present"). See module docstring for why
    this can't just read a pre-existing 0/1 column.

    Fails loudly at construction time (ValueError, not a silent no-op) if
    `negative_labels` matches zero rows in the manifest -- see the module
    docstring's "KNOWN CURRENT DATA GAP" section. This is deliberate: a
    binary classifier trained on one class only would either crash
    unhelpfully mid-training or, worse, "succeed" at producing a useless
    always-positive model. Better to stop immediately with a clear message.
    """

    def __init__(
        self,
        manifest_csv: str,
        transform: T.Compose,
        image_path_col: str = "image_path",
        label_col: str = "label",
        negative_labels: tuple[str, ...] = ("no_lesion",),
        ita_bin_col: str = "ita_bin",
    ):
        self.df = pd.read_csv(manifest_csv)
        for col in (image_path_col, label_col):
            if col not in self.df.columns:
                raise ValueError(
                    f"Manifest {manifest_csv} is missing required column '{col}'. "
                    f"Found columns: {list(self.df.columns)}"
                )
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.ita_bin_col = ita_bin_col if ita_bin_col in self.df.columns else None
        self.negative_labels = tuple(negative_labels)
        self.transform = transform

        raw_labels = self.df[label_col].astype(str)
        negative_set = set(self.negative_labels)
        is_negative = raw_labels.isin(negative_set)

        self.source_negative_labels = sorted(raw_labels[is_negative].unique().tolist())
        self.source_positive_labels = sorted(raw_labels[~is_negative].unique().tolist())

        if len(self.source_negative_labels) == 0:
            raise ValueError(
                f"No negative-class ('no lesion') rows found in {manifest_csv}. "
                f"Looked for label(s) {sorted(negative_set)} in column "
                f"'{label_col}', but every row's label is one of: "
                f"{self.source_positive_labels}. Stage 1 is a binary "
                f"classifier and cannot train with only one class present. "
                f"This is a known, expected gap until the healthy-skin "
                f"negative sources (MCSI subset, team-consented photo "
                f"drive, cropped-patch generation) are merged into this "
                f"manifest by prepare_dataset.py -- see the module "
                f"docstring's 'KNOWN CURRENT DATA GAP' note and the "
                f"negative-sourcing plan in the project docs. Re-run once "
                f"negatives exist, or pass --negative-labels matching "
                f"whatever label string your negative rows actually use."
            )
        if len(self.source_positive_labels) == 0:
            raise ValueError(
                f"Every row in {manifest_csv} matched a negative label "
                f"{sorted(negative_set)} -- there are no lesion-positive "
                f"rows at all. Stage 1 cannot train on a single class."
            )

        self._binary_labels = (~is_negative).astype(int).to_numpy()

        logger.info(
            "Stage 1 manifest %s: %d negative rows (labels: %s), %d positive "
            "rows (labels: %s)",
            manifest_csv,
            int(is_negative.sum()),
            self.source_negative_labels,
            int((~is_negative).sum()),
            self.source_positive_labels,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_path_col]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load image at '{img_path}'") from exc
        image = self.transform(image)
        label = int(self._binary_labels[idx])
        return image, label

    def labels_array(self) -> np.ndarray:
        return self._binary_labels

    def ita_bins_array(self) -> Optional[np.ndarray]:
        """
        Returns the raw `ita_bin` column as a string array (with "unknown"
        left exactly as-is, as a normal bucket value), or None if the
        manifest didn't have that column at all.
        """
        if self.ita_bin_col is None:
            return None
        return self.df[self.ita_bin_col].astype(str).to_numpy()


def build_transforms() -> tuple[T.Compose, T.Compose]:
    """
    Train transform: flip / rotation / brightness-contrast jitter ONLY.
    No hue or saturation jitter — see module docstring and PROJECT_CONTEXT.md §5/§8.
    Val transform: deterministic resize + normalize only.
    """
    train_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2),  # brightness/contrast only
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 via timm, binary output (1 logit, trained with BCE).
    Transfer learning: freeze everything except the last 2 blocks + the
    classifier head.
    """
    model = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=1)
    freeze_backbone_except_last_n_blocks(model, n_blocks=2)
    return model


def freeze_backbone_except_last_n_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """
    Freezes all parameters, then unfreezes:
      - the last `n_blocks` entries of model.blocks (timm EfficientNet's
        MBConv stage list), and
      - the classifier head (model.classifier), plus the final conv/bn
        (conv_head / bn2) that feed it, since those are architecturally
        part of "the head" for an EfficientNet.
    """
    for param in model.parameters():
        param.requires_grad = False

    if not hasattr(model, "blocks"):
        raise AttributeError(
            "Expected a timm EfficientNet-style model with a `.blocks` "
            "ModuleList; got a model without one. Check the timm version / "
            "backbone name."
        )

    for block in model.blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for attr_name in ("conv_head", "bn2", "classifier"):
        module = getattr(model, attr_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Unfroze last %d block(s) + head: %d / %d params trainable",
        n_blocks,
        n_trainable,
        n_total,
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    """
    Runs one epoch. If `optimizer` is provided, trains; otherwise evaluates
    with no_grad. Returns (mean_loss, macro_f1).
    """
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.long().cpu().numpy())

    mean_loss = total_loss / max(n_samples, 1)
    all_preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels_np = np.concatenate(all_labels) if all_labels else np.array([])
    macro_f1 = (
        f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
        if len(all_labels_np)
        else 0.0
    )
    return mean_loss, macro_f1


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    epoch: int,
    val_macro_f1: float,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """
    `extra_metadata` is expected to include `source_negative_labels` and
    `source_positive_labels` (from the training dataset's discovered label
    provenance) so every checkpoint carries an honest record of what this
    specific run actually saw in the data -- not an assumption baked in at
    import time. See LesionPresenceDataset and the module docstring.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": "stage1_lesion_presence",
        "class_names": CLASS_NAMES,
        "backbone": BACKBONE_NAME,
        "img_size": IMG_SIZE,
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
        "epoch": epoch,
        "val_macro_f1": val_macro_f1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    ckpt_path = checkpoint_dir / f"stage1_epoch{epoch}.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, ckpt_path)
    logger.info("Saved checkpoint: %s (val_macro_f1=%.4f)", ckpt_path, val_macro_f1)
    return ckpt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 lesion-presence detector")
    parser.add_argument("--train-manifest", type=str, default="data/processed/train_manifest.csv")
    parser.add_argument("--val-manifest", type=str, default="data/processed/val_manifest.csv")
    parser.add_argument("--image-path-col", type=str, default="image_path")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument(
        "--negative-labels",
        type=str,
        default="no_lesion",
        help=(
            "Comma-separated label string(s) in --label-col that count as "
            "'no lesion' (class 0). Everything else is 'lesion present' "
            "(class 1). Default: 'no_lesion'. As of the current data "
            "pipeline, zero rows match this -- see module docstring."
        ),
    )
    parser.add_argument("--ita-bin-col", type=str, default="ita_bin")
    parser.add_argument("--checkpoint-dir", type=str, default="model/checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    negative_labels = tuple(s.strip() for s in args.negative_labels.split(",") if s.strip())

    train_tf, val_tf = build_transforms()
    # NOTE: dataset construction is where the "no negative rows in this
    # manifest" check fires (see LesionPresenceDataset.__init__). With the
    # current data pipeline, this line is expected to raise a clear
    # ValueError until negatives are merged into train_manifest.csv -- that
    # is intentional, not a bug in this script.
    train_ds = LesionPresenceDataset(
        args.train_manifest,
        train_tf,
        args.image_path_col,
        args.label_col,
        negative_labels,
        args.ita_bin_col,
    )
    val_ds = LesionPresenceDataset(
        args.val_manifest,
        val_tf,
        args.image_path_col,
        args.label_col,
        negative_labels,
        args.ita_bin_col,
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # Recorded once, attached to every checkpoint saved below, so each
    # checkpoint carries an honest record of what labels this run actually
    # trained on -- never assumed downstream.
    label_provenance = {
        "negative_labels_config": list(negative_labels),
        "source_negative_labels": train_ds.source_negative_labels,
        "source_positive_labels": train_ds.source_positive_labels,
        "label_source_column": args.label_col,
    }

    sampler = _get_weighted_sampler(train_ds.labels_array(), train_ds.ita_bins_array())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_f1 = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, device, criterion, optimizer=None)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_macro_f1=%.4f | "
            "val_loss=%.4f val_macro_f1=%.4f | %.1fs",
            epoch,
            args.epochs,
            train_loss,
            train_f1,
            val_loss,
            val_f1,
            elapsed,
        )

        # Checkpoint every epoch, unconditionally (§8 non-negotiable:
        # never assume an uninterrupted session).
        save_checkpoint(model, checkpoint_dir, epoch, val_f1, extra_metadata=label_provenance)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "stage1_best.pt"
            save_checkpoint(
                model,
                checkpoint_dir,
                epoch,
                val_f1,
                extra_metadata={**label_provenance, "is_best": True},
            )
            # Also write a stable "best" copy for downstream consumers.
            torch.save(torch.load(checkpoint_dir / f"stage1_epoch{epoch}.pt"), best_path)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No val_macro_f1 improvement for %d epoch(s) (best=%.4f)",
                epochs_without_improvement,
                best_val_macro_f1,
            )

        if epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping: no improvement in val_macro_f1 for %d epochs "
                "(patience=%d).",
                epochs_without_improvement,
                args.patience,
            )
            break

    logger.info("Training complete. Best val_macro_f1=%.4f", best_val_macro_f1)


if __name__ == "__main__":
    main()
