"""
model/stage2_disease_classifier.py

Nirikshan — Stage 2: Disease Classifier.

4-way classifier, only ever run (at inference time) on images Stage 1 has
flagged as "lesion present" (see model/stage1_lesion_presence.py and
model/inference.py). Classes, fixed order:

    0: squamous_cell_carcinoma   -- malignant end-state
    1: actinic_keratosis         -- precancerous precursor
    2: nevus                     -- hard negative control
    3: seborrheic_keratosis      -- hard negative control

IMPORTANT (per the task spec): for TRAINING and evaluation, this script uses
ground-truth lesion labels from the manifest, NOT Stage 1's predictions —
using Stage 1's predicted lesion-present flag to filter/train Stage 2 would
compound Stage 1's errors into Stage 2's training distribution. Only at
serving time (model/inference.py) does Stage 1's prediction gate whether
Stage 2 runs at all.

Expected input: a CSV manifest with at minimum these columns:
    image_path   -- path to the image
    label        -- the same raw label column model/stage1_lesion_presence.py
                     reads. Rows whose label is one of the 4 class name
                     strings above are used for Stage 2 training; any other
                     row (e.g. a negative-class "no_lesion" row once those
                     exist) is simply not one of Stage 2's classes and is
                     excluded -- there's no separate lesion_present flag to
                     check. See DiseaseClassifierDataset.

Class-weighted loss: CrossEntropyLoss weights computed as inverse class
frequency over the training manifest's disease_label column, so the
minority classes (typically actinic_keratosis and the rarer control class)
are not swamped.

Checkpoint format matches Stage 1: `model_state_dict` + `metadata` dict
(class_names list + everything else), saved every epoch to
model/checkpoints/stage2_epoch{N}.pt. Never hardcode label order downstream
— always read `class_names` from the checkpoint metadata (§8).

Early stopping: patience=4 on validation macro-F1.

Augmentation: identical policy to Stage 1 — horizontal flip, +/-20 degree
rotation, brightness/contrast jitter only. No hue/saturation jitter, ever.
"""

from __future__ import annotations

import argparse
import logging
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
        "timm is required for stage2_disease_classifier.py. "
        "pip install -r requirements-model.txt"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [stage2] %(levelname)s %(message)s",
)
logger = logging.getLogger("stage2_disease_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BACKBONE_NAME = "efficientnet_b0"

# Fixed class order. This exact list (and this exact order) is what gets
# written into every checkpoint's metadata["class_names"], and is the only
# source of truth downstream code should ever use for index -> label lookup.
CLASS_NAMES = [
    "squamous_cell_carcinoma",
    "actinic_keratosis",
    "nevus",
    "seborrheic_keratosis",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DiseaseClassifierDataset(Dataset):
    """
    Reads a manifest CSV that has a single raw `label` column (the same
    column model/stage1_lesion_presence.py reads) and filters to rows whose
    label is one of the four fixed CLASS_NAMES. Ground-truth labels only --
    never Stage 1's predictions -- so Stage 2 training never compounds
    Stage 1's errors (see module docstring). Any row whose label is
    something else (a negative-class label like "no_lesion" once those
    exist, or anything unrecognized) is naturally excluded by this filter
    without needing a separate lesion_present flag column at all -- Stage 2
    only ever cares "is this one of my four known disease classes."

    Unlike Stage 1's LesionPresenceDataset, CLASS_NAMES here is NOT
    discovered from the data: it's the fixed 4-class clinical taxonomy from
    PROJECT_CONTEXT.md §5 (SCC / AK / nevus / seborrheic keratosis), which
    doesn't change based on what happens to be in a given manifest. What
    IS validated against the data is that each of those 4 classes actually
    has training examples -- a class with zero rows gets a loud warning
    (not a silent divide-by-a-floor-of-1 in class_weights()), since per-bin
    counts feed the fairness story this whole project is built around.
    """

    def __init__(
        self,
        manifest_csv: str,
        transform: T.Compose,
        image_path_col: str = "image_path",
        label_col: str = "label",
    ):
        raw_df = pd.read_csv(manifest_csv)
        for col in (image_path_col, label_col):
            if col not in raw_df.columns:
                raise ValueError(
                    f"Manifest {manifest_csv} is missing required column '{col}'. "
                    f"Found columns: {list(raw_df.columns)}"
                )

        labels_str = raw_df[label_col].astype(str)
        mask_known_class = labels_str.isin(CLASS_NAMES)
        dropped = (~mask_known_class).sum()
        if dropped:
            dropped_labels = sorted(labels_str[~mask_known_class].unique().tolist())
            logger.info(
                "Excluding %d rows whose label isn't one of Stage 2's 4 "
                "disease classes (found: %s) -- expected for negative-class "
                "('no_lesion') and any other non-disease rows.",
                dropped,
                dropped_labels,
            )

        self.df = raw_df[mask_known_class].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"No usable rows found in {manifest_csv} after filtering "
                f"'{label_col}' to Stage 2's known classes {CLASS_NAMES}. "
                f"Labels actually present: {sorted(labels_str.unique().tolist())}"
            )

        counts_by_class = self.df[label_col].value_counts()
        missing_classes = [c for c in CLASS_NAMES if c not in counts_by_class.index]
        if missing_classes:
            logger.warning(
                "Manifest %s has ZERO rows for class(es) %s. Stage 2 will "
                "still 'train' (class_weights() floors the count to 1 to "
                "avoid a divide-by-zero) but that class will be learned "
                "from nothing -- treat any reported accuracy on it as "
                "meaningless until real examples are added.",
                manifest_csv,
                missing_classes,
            )
        logger.info("Stage 2 manifest %s class counts: %s", manifest_csv, counts_by_class.to_dict())

        self.image_path_col = image_path_col
        self.label_col = label_col
        self.transform = transform

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
        label_idx = CLASS_TO_IDX[row[self.label_col]]
        return image, label_idx

    def labels_array(self) -> np.ndarray:
        return self.df[self.label_col].map(CLASS_TO_IDX).to_numpy()

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-class-frequency weights, in CLASS_NAMES order, for use with
        nn.CrossEntropyLoss(weight=...). A class with 0 real rows is floored
        to a count of 1 purely to avoid a division by zero -- see the loud
        warning logged in __init__ for that case, since the resulting
        weight is not meaningful.
        """
        counts = np.array(
            [max((self.labels_array() == i).sum(), 1) for i in range(len(CLASS_NAMES))],
            dtype=np.float64,
        )
        weights = counts.sum() / (len(CLASS_NAMES) * counts)
        return torch.as_tensor(weights, dtype=torch.float32)


def build_transforms() -> tuple[T.Compose, T.Compose]:
    """Same augmentation policy as Stage 1: flip / rotation / brightness-contrast only."""
    train_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2),
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

def build_model(pretrained: bool = True, num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    model = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=num_classes)
    freeze_backbone_except_last_n_blocks(model, n_blocks=2)
    return model


def freeze_backbone_except_last_n_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """Identical freezing policy to Stage 1 — see that module for rationale."""
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
            labels = labels.to(device).long()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": "stage2_disease_classifier",
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

    ckpt_path = checkpoint_dir / f"stage2_epoch{epoch}.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, ckpt_path)
    logger.info("Saved checkpoint: %s (val_macro_f1=%.4f)", ckpt_path, val_macro_f1)
    return ckpt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 disease classifier")
    parser.add_argument("--train-manifest", type=str, default="data/processed/train_manifest.csv")
    parser.add_argument("--val-manifest", type=str, default="data/processed/val_manifest.csv")
    parser.add_argument("--image-path-col", type=str, default="image_path")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--checkpoint-dir", type=str, default="model/checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
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

    train_tf, val_tf = build_transforms()
    train_ds = DiseaseClassifierDataset(
        args.train_manifest, train_tf, args.image_path_col, args.label_col
    )
    val_ds = DiseaseClassifierDataset(
        args.val_manifest, val_tf, args.image_path_col, args.label_col
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    logger.info("Class order: %s", CLASS_NAMES)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    class_weights = train_ds.class_weights().to(device)
    logger.info(
        "Class-weighted CE loss weights (inverse frequency), %s: %s",
        CLASS_NAMES,
        [round(w, 4) for w in class_weights.cpu().tolist()],
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

        # Checkpoint every epoch, unconditionally (§8: never assume an
        # uninterrupted session).
        save_checkpoint(model, checkpoint_dir, epoch, val_f1)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "stage2_best.pt"
            torch.save(torch.load(checkpoint_dir / f"stage2_epoch{epoch}.pt"), best_path)
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
