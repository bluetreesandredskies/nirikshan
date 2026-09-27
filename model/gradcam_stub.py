"""
model/gradcam_stub.py

Placeholder for the Grad-CAM overlay feature described in
PROJECT_CONTEXT.md §3 (frontend: "Grad-CAM overlay on the uploaded image").

Full implementation lands in Session 8 (hooking into Stage 2's last conv
block, computing class-activation weighted feature maps, and rendering a
heatmap overlay image). This stub exists purely so model/inference.py has a
stable function signature to call today without erroring, and so the
frontend/backend can build the "gradcam_overlay" field into their response
shape now instead of retrofitting it later.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: "torch.Tensor",  # noqa: F821 - documented, torch imported by caller
    target_class_idx: int,
    original_image_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    Placeholder Grad-CAM overlay generator. NOT YET IMPLEMENTED.

    Intended eventual behavior (Session 8):
        Registers forward/backward hooks on Stage 2's final convolutional
        block, runs a forward + backward pass targeting `target_class_idx`,
        computes the Grad-CAM weighted activation map, upsamples it to the
        original image resolution, and composites it as a semi-transparent
        heatmap over `original_image_path` (or the input tensor if no
        original path is available), returning PNG-encoded bytes.

    Args:
        model: the trained Stage 2 disease-classifier model (already loaded,
            in eval mode).
        input_tensor: the same normalized (1, 3, H, W) tensor that was fed
            to the model's forward pass for this prediction.
        target_class_idx: the class index to compute the CAM for -- almost
            always the model's top-1 prediction index, but callable for any
            class (e.g. to visualize "why did it think this might be SCC"
            even when SCC wasn't top-1, which is relevant given the safety
            net in model/safety_net.py can escalate risk based on p_scc /
            p_ak even when neither is the top-1 class).
        original_image_path: optional path to the original, pre-resize
            image, so the overlay can be composited at native resolution
            rather than the 224x224 model input size.

    Returns:
        Currently always None. Once implemented: PNG-encoded image bytes of
        the heatmap-overlaid image, suitable for the backend to return as a
        base64 string or serve directly.
    """
    # Intentionally unimplemented -- see Session 8.
    return None
