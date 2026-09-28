"""
backend/app/models_io.py

Model loading for the Nirikshan API. Checkpoints are loaded ONCE at startup (see
`load_model_backend`, called from main.py's lifespan) and reused for every request.

>>> SESSION 9 NOTE (Render, root dir = backend/) <<<
    Locally the app runs as `uvicorn backend.app.main:app` from the repo root, where `model/`
    is a sibling of `backend/`. All sys.path handling for that lives in
    `_ensure_model_package_importable()` below and NOWHERE else. On Render with root dir
    `backend/`, `model/safety_net.py` will not be importable unless it is copied/relocated;
    the only place that needs to change is `_load_apply_safety_net()` (the one function that
    imports it). main.py never touches model/ directly.

Design
------
* `ModelBackend` (abstract) owns everything that is NOT tied to a runtime: Stage-2 class-name
  lookup by NAME, the Stage-1 threshold, the base-risk mapping, the B.4 safety net, and the
  shape of the result dict. Its public method is `predict(image) -> dict`.
* A concrete backend only implements `infer_raw(image) -> RawInference` (Stage-1 lesion
  probability + Stage-2 softmax probabilities in checkpoint class order).
  `TorchModelBackend` below is the PyTorch one; Session 9's `OnnxModelBackend` implements the
  same single method and `predict()` / main.py stay untouched.
* Label order and preprocessing values are always read from `ckpt["metadata"]`, never hardcoded.
* Stage 2 ALWAYS runs on every image (Stage 1 is a placeholder and unreliable). The safety net
  runs on Stage 2's probabilities regardless of Stage 1's verdict.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image

logger = logging.getLogger("models_io")

# ---------------------------------------------------------------------------
# Constants (single source of truth)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = "model/checkpoints"
STAGE1_FILENAME = "stage1_final.pt"
STAGE2_FILENAME = "stage2_final.pt"
STAGE1_MODULE = "model.stage1_lesion_presence"
STAGE2_MODULE = "model.stage2_disease_classifier"

# Stage 1 was trained on only 69 unique negative images -> placeholder. Surfaced to the
# frontend in every prediction response. Change it here and nowhere else.
MODEL_STATUS = "stage1_placeholder"

LESION_PRESENT_THRESHOLD = 0.5
LESION_CLASS_NAME = "lesion_present"  # Stage-1 class name (index 1 in the current checkpoint)
SCC_CLASS_NAME = "squamous_cell_carcinoma"
AK_CLASS_NAME = "actinic_keratosis"

# Risk BEFORE the safety net, from Stage 2's top-1 class. (Same mapping as model/inference.py;
# duplicated here only so the deployed backend does not need to import torch-heavy inference.py.)
BASE_RISK_BY_TOP1_CLASS = {
    "squamous_cell_carcinoma": "high",
    "actinic_keratosis": "moderate",
    "nevus": "low",
    "seborrheic_keratosis": "low",
}
# Stage 1 said "no lesion": base risk is "low" (never "none" -- a photo is not a clearance).
# The safety net can still lift it if Stage 2 sees p_scc / p_ak above threshold.
NO_LESION_BASE_RISK = "low"


# ---------------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawInference:
    """Runtime-independent model outputs. Stage-2 probs are in Stage-2 checkpoint class order."""

    p_lesion: float
    stage2_probs: Sequence[float]


def _ensure_model_package_importable() -> None:
    """The ONLY place that touches sys.path for model/ (see Session 9 note at top)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_apply_safety_net() -> Callable[..., Any]:
    """The ONLY place that imports model/safety_net.py (see Session 9 note at top)."""
    _ensure_model_package_importable()
    from model.safety_net import apply_safety_net  # noqa: WPS433 (deliberately local)

    return apply_safety_net


class ModelBackend(ABC):
    """
    Interface main.py depends on. Public surface: `predict(image)`, `info()`.
    Subclasses implement `infer_raw` only.
    """

    def __init__(self, stage2_class_names: Sequence[str], model_versions: dict):
        names = list(stage2_class_names)
        for required in (SCC_CLASS_NAME, AK_CLASS_NAME):
            if required not in names:
                raise ValueError(
                    f"Stage 2 class_names {names} is missing '{required}', which the safety net needs."
                )
        self.stage2_class_names: list[str] = names
        self.model_versions: dict = dict(model_versions)
        # Look up by NAME, never by hardcoded index.
        self._scc_idx = names.index(SCC_CLASS_NAME)
        self._ak_idx = names.index(AK_CLASS_NAME)
        self._apply_safety_net = _load_apply_safety_net()

    @abstractmethod
    def infer_raw(self, image: Image.Image) -> RawInference:
        """Run Stage 1 and Stage 2 on an RGB PIL image. Stage 2 must run unconditionally."""

    def info(self) -> dict:
        return {
            "backend": type(self).__name__,
            "model_status": MODEL_STATUS,
            "stage2_class_names": list(self.stage2_class_names),
            "model_versions": dict(self.model_versions),
        }

    def predict(self, image: Image.Image) -> dict:
        """
        Returns the image-side result. Keys `image_risk_level`, `p_scc`, `p_ak` are exactly
        what backend/app/fusion.py expects; the rest is additive.
        """
        raw = self.infer_raw(image)
        probs = [float(p) for p in raw.stage2_probs]
        if len(probs) != len(self.stage2_class_names):
            raise ValueError(
                f"Backend returned {len(probs)} Stage-2 probabilities for "
                f"{len(self.stage2_class_names)} classes."
            )
        p_lesion = float(raw.p_lesion)
        lesion_detected = p_lesion >= LESION_PRESENT_THRESHOLD

        top1_idx = max(range(len(probs)), key=probs.__getitem__)
        top1_class = self.stage2_class_names[top1_idx]
        p_scc, p_ak = probs[self._scc_idx], probs[self._ak_idx]

        base_risk = (
            BASE_RISK_BY_TOP1_CLASS.get(top1_class, "low") if lesion_detected else NO_LESION_BASE_RISK
        )
        # B.4 safety net on Stage 2's probabilities, regardless of Stage 1's verdict.
        net = self._apply_safety_net(p_scc=p_scc, p_ak=p_ak, base_risk=base_risk)

        return {
            "image_risk_level": net.risk_level,  # POST-escalation
            "p_scc": p_scc,
            "p_ak": p_ak,
            "safety_escalated": bool(net.safety_escalated),
            "safety_explanation": net.explanation,
            "base_risk": base_risk,  # PRE-escalation, for transparency
            "lesion_detected": lesion_detected,  # Stage 1 verdict only
            "stage1_lesion_prob": p_lesion,
            "top1_class": top1_class,
            "probabilities": dict(zip(self.stage2_class_names, probs)),
            "gradcam_png_base64": None,  # Session 8 fills this in
            "model_status": MODEL_STATUS,
            "model_versions": dict(self.model_versions),
        }


# ---------------------------------------------------------------------------
# Checkpoint + transform helpers (PyTorch path)
# ---------------------------------------------------------------------------
_REQUIRED_METADATA = ("class_names", "img_size", "normalize_mean", "normalize_std")


def _load_checkpoint(torch_module: Any, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Set MODEL_CHECKPOINT_DIR to a directory containing "
            f"{STAGE1_FILENAME} and {STAGE2_FILENAME}."
        )
    ckpt = torch_module.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "metadata" not in ckpt:
        raise ValueError(f"{path} must be a dict with 'model_state_dict' and 'metadata'.")
    missing = [k for k in _REQUIRED_METADATA if k not in ckpt["metadata"]]
    if missing:
        raise ValueError(f"{path}: metadata is missing {missing}. Label order/preprocessing are never hardcoded.")
    return ckpt


def _head_out_features(state_dict: dict) -> Optional[int]:
    for key in ("classifier.weight", "head.fc.weight", "fc.weight", "head.weight"):
        weight = state_dict.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            return int(weight.shape[0])
    return None


def _is_deterministic(transform: Any) -> bool:
    for part in getattr(transform, "transforms", []):
        name = type(part).__name__
        if name.startswith("Random") or name == "ColorJitter":
            return False
    return True


def _produces_expected_tensor(transform: Any, img_size: int) -> bool:
    try:
        out = transform(Image.new("RGB", (300, 200)))
        return tuple(out.shape) == (3, img_size, img_size)
    except Exception:  # noqa: BLE001
        return False


def _fallback_eval_transform(img_size: int, mean: list, std: list) -> Any:
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _resolve_eval_transform(stage_module_name: str, metadata: dict) -> tuple[Any, str]:
    """
    Prefer the stage module's own `build_transforms()` eval transform (so inference matches
    training by construction). If it cannot be resolved unambiguously, fall back to the
    metadata-driven eval transform (224x224 LANCZOS + ImageNet mean/std, no augmentation) and
    log a WARNING. Returns (transform, source_label).
    """
    img_size = int(metadata["img_size"])
    mean, std = list(metadata["normalize_mean"]), list(metadata["normalize_std"])
    try:
        _ensure_model_package_importable()
        build = getattr(importlib.import_module(stage_module_name), "build_transforms")
        kwargs: dict = {}
        for pname, param in inspect.signature(build).parameters.items():
            if pname in ("img_size", "image_size", "size"):
                kwargs[pname] = img_size
            elif pname in ("mean", "normalize_mean"):
                kwargs[pname] = mean
            elif pname in ("std", "normalize_std"):
                kwargs[pname] = std
            elif pname in ("train", "is_train", "training", "augment", "augmentation"):
                kwargs[pname] = False
            elif pname in ("split", "mode", "phase"):
                kwargs[pname] = "val"
            elif param.default is inspect.Parameter.empty and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(f"unrecognised required parameter '{pname}'")
        out = build(**kwargs)
        if isinstance(out, dict):
            candidates = [out.get(k) for k in ("val", "valid", "eval", "test")]
        elif isinstance(out, (tuple, list)):
            candidates = list(reversed(out))  # conventional (train, val): try val first
        else:
            candidates = [out]
        for cand in candidates:
            if callable(cand) and _is_deterministic(cand) and _produces_expected_tensor(cand, img_size):
                return cand, f"{stage_module_name}.build_transforms"
        raise ValueError("no deterministic eval transform among build_transforms() outputs")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not reuse %s.build_transforms() (%s: %s); using metadata-driven eval "
            "transform (%dx%d, LANCZOS, ImageNet mean/std). Verify it matches training.",
            stage_module_name, type(exc).__name__, exc, img_size, img_size,
        )
        return _fallback_eval_transform(img_size, mean, std), "metadata-fallback"


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------
class TorchModelBackend(ModelBackend):
    """Loads stage1_final.pt / stage2_final.pt once; CPU inference."""

    def __init__(self, checkpoint_dir: Path):
        import timm
        import torch

        t0 = time.perf_counter()
        self._torch = torch
        s1 = _load_checkpoint(torch, checkpoint_dir / STAGE1_FILENAME)
        s2 = _load_checkpoint(torch, checkpoint_dir / STAGE2_FILENAME)
        s1_meta, s2_meta = s1["metadata"], s2["metadata"]
        s1_names, s2_names = list(s1_meta["class_names"]), list(s2_meta["class_names"])
        if LESION_CLASS_NAME not in s1_names:
            raise ValueError(f"Stage 1 class_names {s1_names} has no '{LESION_CLASS_NAME}'.")

        super().__init__(
            s2_names,
            {"stage1_epoch": s1_meta.get("epoch"), "stage2_epoch": s2_meta.get("epoch")},
        )
        self._s1_names = s1_names
        self._s1_lesion_idx = s1_names.index(LESION_CLASS_NAME)

        self._s1_model, s1_out = self._build_model(timm, s1, len(s1_names))
        self._s2_model, s2_out = self._build_model(timm, s2, len(s2_names))
        # Stage 1 may be a 1-logit BCE head or a 2-logit softmax head; support both.
        if s1_out not in (1, len(s1_names)):
            raise ValueError(f"Stage 1 head has {s1_out} outputs; expected 1 or {len(s1_names)}.")
        if s2_out != len(s2_names):
            raise ValueError(f"Stage 2 head has {s2_out} outputs but class_names has {len(s2_names)}.")
        self._s1_single_logit = s1_out == 1

        self._s1_tf, s1_src = _resolve_eval_transform(STAGE1_MODULE, s1_meta)
        self._s2_tf, s2_src = _resolve_eval_transform(STAGE2_MODULE, s2_meta)
        self.transform_sources = {"stage1": s1_src, "stage2": s2_src}

        logger.info(
            "Loaded checkpoints from %s in %.2fs (stage1 classes=%s epoch=%s, stage2 classes=%s "
            "epoch=%s, transforms=%s)",
            checkpoint_dir, time.perf_counter() - t0, s1_names, s1_meta.get("epoch"),
            s2_names, s2_meta.get("epoch"), self.transform_sources,
        )

    @staticmethod
    def _build_model(timm_module: Any, ckpt: dict, n_names: int) -> tuple[Any, int]:
        state_dict, meta = ckpt["model_state_dict"], ckpt["metadata"]
        out_features = _head_out_features(state_dict) or n_names
        model = timm_module.create_model(
            meta.get("backbone", "efficientnet_b0"), pretrained=False, num_classes=out_features
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model, out_features

    def infer_raw(self, image: Image.Image) -> RawInference:
        torch = self._torch
        with torch.inference_mode():
            logits1 = self._s1_model(self._s1_tf(image).unsqueeze(0))
            if self._s1_single_logit:
                p_lesion = torch.sigmoid(logits1.reshape(-1)[0]).item()
            else:
                p_lesion = torch.softmax(logits1, dim=1)[0, self._s1_lesion_idx].item()
            # Stage 2 runs on EVERY image, whatever Stage 1 said.
            probs = torch.softmax(self._s2_model(self._s2_tf(image).unsqueeze(0)), dim=1)[0].tolist()
        return RawInference(p_lesion=p_lesion, stage2_probs=probs)

    def info(self) -> dict:
        return {**super().info(), "transform_sources": dict(self.transform_sources)}


# ---------------------------------------------------------------------------
# Factory (Session 9: switch the default here to the ONNX backend)
# ---------------------------------------------------------------------------
def resolve_checkpoint_dir(checkpoint_dir: Optional[str] = None) -> Path:
    path = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    return path if path.is_absolute() else REPO_ROOT / path


def load_model_backend(checkpoint_dir: Optional[str] = None) -> ModelBackend:
    """Called once at startup. Returns the concrete ModelBackend for this deployment."""
    return TorchModelBackend(resolve_checkpoint_dir(checkpoint_dir))
