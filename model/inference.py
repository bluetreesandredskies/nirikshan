"""
model/inference.py

Nirikshan — end-to-end single-image inference.

Loads the Stage 1 (lesion presence) and Stage 2 (disease classifier)
checkpoints, and on a single image runs:

    Stage 1 -> (if lesion present) Stage 2 -> safety_net.apply_safety_net()

...returning one structured result dict. This is the "image_risk" side of
the fusion the backend performs in backend/app/fusion.py (PROJECT_CONTEXT.md
§3, §7): the exposure-history side is computed separately and the two are
combined there, NOT here -- this module only ever produces the image-based
half of the final risk, per the §8 non-negotiable that image-based and
exposure-based risk must always be shown/kept separately, never pre-merged.

Class order is NEVER hardcoded here: both stages' class names are read from
each checkpoint's `metadata["class_names"]` at load time (§8 non-negotiable).

Output shape. Per the Build Execution Guide's Session 2 prompt,
backend/app/fusion.py is written against an image-risk dict stubbed as
exactly `{image_risk_level, p_scc, p_ak}`, and Session 6's /predict
endpoint additionally needs the `safety_escalated` flag. Those four keys
are therefore kept flat at the top level so fusion.py and main.py can read
them directly without reaching into nested structures; everything else
(stage-1 probability, full stage-2 distribution, model versions, the
pre-safety-net base risk) is included alongside for anyone who wants more
detail, but is additive -- removing it wouldn't break the fusion.py /
main.py contract.

    {
        "image_risk_level": str,        # POST-safety-net risk; "none"/"low"/"moderate"/"high"
        "p_scc": float,                  # 0.0 if no lesion detected (Stage 2 didn't run)
        "p_ak": float,                   # 0.0 if no lesion detected
        "safety_escalated": bool,
        "safety_explanation": Optional[str],
        "lesion_detected": bool,
        "base_risk": str,                # risk level BEFORE the safety net, for transparency
        "top1_class": Optional[str],     # None if no lesion detected
        "p_lesion": float,                # Stage 1 sigmoid probability of "lesion_present"
        "probabilities": Optional[dict],  # full Stage 2 {class_name: prob}, None if no lesion
        "gradcam_overlay": Optional[bytes],   # currently always None, see gradcam_stub.py
        "model_versions": {
            "stage1_epoch": int,
            "stage2_epoch": Optional[int],   # None if Stage 2 didn't run
        },
    }

If a later session's actual backend/app/fusion.py or main.py wants
different field names, adjust the dict literals in `run_inference` below --
the pipeline logic itself (Stage 1 -> Stage 2 -> safety net) is the stable
part.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms as T

from model.gradcam_stub import generate_gradcam_overlay
from model.safety_net import apply_safety_net

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required for model/inference.py. "
        "pip install -r requirements-model.txt"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [inference] %(levelname)s %(message)s",
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Base-risk mapping from Stage 2's top-1 predicted class.
#
# This is the risk level BEFORE the safety net looks at p_scc / p_ak
# specifically -- the safety net (model/safety_net.py) may escalate this
# regardless of what top-1 says. Kept here, not in safety_net.py, because
# it's specific to how this pipeline derives a base risk, not part of the
# generic escalation rule itself.
# ---------------------------------------------------------------------------
BASE_RISK_BY_TOP1_CLASS = {
    "squamous_cell_carcinoma": "high",
    "actinic_keratosis": "moderate",
    "nevus": "low",
    "seborrheic_keratosis": "low",
}
NO_LESION_RISK = "none"


def _build_eval_transform(img_size: int, mean: list[float], std: list[float]) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    for required_key in ("model_state_dict", "metadata"):
        if required_key not in ckpt:
            raise ValueError(
                f"Checkpoint at '{checkpoint_path}' is missing required key "
                f"'{required_key}'. Refusing to load -- see PROJECT_CONTEXT.md §8."
            )
    if "class_names" not in ckpt["metadata"]:
        raise ValueError(
            f"Checkpoint at '{checkpoint_path}' metadata has no 'class_names'. "
            f"Never hardcode label order downstream -- see PROJECT_CONTEXT.md §8."
        )
    return ckpt


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    metadata = ckpt["metadata"]
    backbone = metadata.get("backbone", "efficientnet_b0")
    num_classes = len(metadata["class_names"])
    # Stage 1 is trained as a 1-logit BCE binary head; Stage 2 as an
    # N-logit softmax head. Both were built with timm.create_model(...,
    # num_classes=<1 or 4>), so re-derive num_classes the same way here.
    stage = metadata.get("stage", "")
    model_num_classes = 1 if stage == "stage1_lesion_presence" else num_classes
    model = timm.create_model(backbone, pretrained=False, num_classes=model_num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class NirikshanImagePipeline:
    """
    Loads both stage checkpoints once; call `run_inference(image_path)` per
    image. Reuse one instance across many predictions instead of reloading
    checkpoints per call.
    """

    def __init__(
        self,
        stage1_checkpoint_path: str,
        stage2_checkpoint_path: str,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stage1_ckpt = load_checkpoint(stage1_checkpoint_path, self.device)
        self.stage2_ckpt = load_checkpoint(stage2_checkpoint_path, self.device)

        self.stage1_class_names: list[str] = self.stage1_ckpt["metadata"]["class_names"]
        self.stage2_class_names: list[str] = self.stage2_ckpt["metadata"]["class_names"]

        if "squamous_cell_carcinoma" not in self.stage2_class_names or (
            "actinic_keratosis" not in self.stage2_class_names
        ):
            raise ValueError(
                "Stage 2 checkpoint's class_names is missing "
                "'squamous_cell_carcinoma' and/or 'actinic_keratosis', both "
                "required by the safety net. class_names found: "
                f"{self.stage2_class_names}"
            )

        self.stage1_model = build_model_from_checkpoint(self.stage1_ckpt, self.device)
        self.stage2_model = build_model_from_checkpoint(self.stage2_ckpt, self.device)

        s1_meta = self.stage1_ckpt["metadata"]
        s2_meta = self.stage2_ckpt["metadata"]
        self.stage1_transform = _build_eval_transform(
            s1_meta["img_size"], s1_meta["normalize_mean"], s1_meta["normalize_std"]
        )
        self.stage2_transform = _build_eval_transform(
            s2_meta["img_size"], s2_meta["normalize_mean"], s2_meta["normalize_std"]
        )

        self.scc_idx = self.stage2_class_names.index("squamous_cell_carcinoma")
        self.ak_idx = self.stage2_class_names.index("actinic_keratosis")

        logger.info(
            "Loaded Stage 1 (epoch %s, classes=%s) and Stage 2 (epoch %s, classes=%s)",
            s1_meta.get("epoch"),
            self.stage1_class_names,
            s2_meta.get("epoch"),
            self.stage2_class_names,
        )

    def run_inference(
        self,
        image_path: str,
        lesion_present_threshold: float = 0.5,
        with_gradcam: bool = False,
    ) -> dict:
        """
        Runs Stage 1 -> (conditionally) Stage 2 -> safety net on one image.
        Returns the structured result dict described in the module
        docstring.
        """
        image = Image.open(image_path).convert("RGB")

        # ---- Stage 1: lesion presence -------------------------------------
        stage1_input = self.stage1_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            stage1_logit = self.stage1_model(stage1_input).squeeze(1)
            p_lesion = torch.sigmoid(stage1_logit).item()
        lesion_detected = p_lesion >= lesion_present_threshold

        result: dict = {
            "image_risk_level": NO_LESION_RISK,
            "p_scc": 0.0,
            "p_ak": 0.0,
            "safety_escalated": False,
            "safety_explanation": None,
            "lesion_detected": lesion_detected,
            "base_risk": NO_LESION_RISK,
            "top1_class": None,
            "p_lesion": p_lesion,
            "probabilities": None,
            "gradcam_overlay": None,
            "model_versions": {
                "stage1_epoch": self.stage1_ckpt["metadata"].get("epoch"),
                "stage2_epoch": None,
            },
        }

        if not lesion_detected:
            logger.info(
                "Stage 1: no lesion detected (p_lesion=%.4f < threshold %.2f). "
                "Stage 2 skipped.",
                p_lesion,
                lesion_present_threshold,
            )
            return result

        # ---- Stage 2: disease classification (only runs here) -------------
        stage2_input = self.stage2_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            stage2_logits = self.stage2_model(stage2_input)
            stage2_probs = torch.softmax(stage2_logits, dim=1).squeeze(0)

        probabilities = {
            name: stage2_probs[idx].item() for idx, name in enumerate(self.stage2_class_names)
        }
        top1_idx = int(torch.argmax(stage2_probs).item())
        top1_class = self.stage2_class_names[top1_idx]
        p_scc = stage2_probs[self.scc_idx].item()
        p_ak = stage2_probs[self.ak_idx].item()

        base_risk = BASE_RISK_BY_TOP1_CLASS.get(top1_class, "low")

        # ---- Safety net (§6, VERBATIM rule) --------------------------------
        safety_result = apply_safety_net(p_scc=p_scc, p_ak=p_ak, base_risk=base_risk)

        gradcam_overlay = None
        if with_gradcam:
            gradcam_overlay = generate_gradcam_overlay(
                model=self.stage2_model,
                input_tensor=stage2_input,
                target_class_idx=top1_idx,
                original_image_path=image_path,
            )

        result.update(
            {
                "image_risk_level": safety_result.risk_level,
                "p_scc": p_scc,
                "p_ak": p_ak,
                "safety_escalated": safety_result.safety_escalated,
                "safety_explanation": safety_result.explanation,
                "base_risk": safety_result.base_risk,
                "top1_class": top1_class,
                "probabilities": probabilities,
                "gradcam_overlay": gradcam_overlay,
            }
        )
        result["model_versions"]["stage2_epoch"] = self.stage2_ckpt["metadata"].get("epoch")

        logger.info(
            "Stage 2: top1=%s p_scc=%.4f p_ak=%.4f base_risk=%s -> final image_risk_level=%s "
            "(safety_escalated=%s)",
            top1_class,
            p_scc,
            p_ak,
            base_risk,
            safety_result.risk_level,
            safety_result.safety_escalated,
        )
        return result


# ---------------------------------------------------------------------------
# CLI entry point (mainly for smoke-testing a single image from the shell)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image Nirikshan inference")
    parser.add_argument("--image", type=str, required=True)
    # NOTE: the Build Execution Guide's Session 5 (Colab) releases checkpoints
    # named stage1_final.pt / stage2_final.pt via `gh release create`. This
    # script's own training loop instead writes a running "best-by-val-F1"
    # copy named stage1_best.pt / stage2_best.pt. These are two different
    # training entry points that may end up naming their output differently
    # -- when wiring Session 6's models_io.py (or this CLI) against whatever
    # Session 5 actually publishes, just point --stage1-checkpoint /
    # --stage2-checkpoint at the real downloaded filename; nothing else here
    # depends on the literal name.
    parser.add_argument("--stage1-checkpoint", type=str, default="model/checkpoints/stage1_best.pt")
    parser.add_argument("--stage2-checkpoint", type=str, default="model/checkpoints/stage2_best.pt")
    parser.add_argument("--with-gradcam", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = NirikshanImagePipeline(args.stage1_checkpoint, args.stage2_checkpoint)
    result = pipeline.run_inference(args.image, with_gradcam=args.with_gradcam)
    # gradcam_overlay may be raw bytes; keep JSON output printable.
    printable = {**result, "gradcam_overlay": bool(result["gradcam_overlay"])}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
