"""
fusion.py -- combines the image-model risk with the exposure-history risk for Nirikshan.

Rules enforced here (PROJECT_CONTEXT sections 6, 7, 8):
  * The two components are ALWAYS returned separately, never pre-merged into one score.
  * The safety-net escalation rule (p_scc / p_ak) is applied to the IMAGE side only.
    Exposure risk never changes the image risk level, and vice versa.
  * The displayed recommendation matches the post-escalation risk, never the raw top-1 guess.

Expected image_risk input (stub of what /predict will produce from Stage 2 softmax):
    {"image_risk_level": "none" | "low" | "moderate" | "high", "p_scc": 0.0-1.0, "p_ak": 0.0-1.0}
Expected exposure_risk input: the dict returned by risk_fusion.assess_exposure_risk().
"""

from __future__ import annotations

from typing import Any, Mapping

# --- Safety-net thresholds (tunable constants) -------------------------------------------
SCC_ESCALATION_THRESHOLD = 0.15  # spec: p_scc >= 15% escalates to "high"
AK_ESCALATION_THRESHOLD = 0.10   # ESTIMATE, validate with clinical partner: spec says "slightly lower" than the SCC threshold

# "none" and "low" are the same rank; both escalate to "moderate" on the AK rule.
_LEVEL_RANK = {"none": 0, "low": 0, "moderate": 1, "high": 2}
_RANK_TO_LEVEL = {0: "low", 1: "moderate", 2: "high"}

RECOMMENDATIONS = {
    "low": "No urgent concern from this screening. Keep checking your skin regularly and see a doctor if anything changes, itches, bleeds or does not heal.",
    "moderate": "Please see a dermatologist or a primary health centre doctor within the next few weeks for an in-person skin examination.",
    "high": "Please see a dermatologist or doctor promptly, ideally within the next week, for an in-person examination and possible biopsy.",
}


def _validate_image_risk(image_risk: Mapping[str, Any]) -> None:
    missing = {"image_risk_level", "p_scc", "p_ak"} - set(image_risk)
    if missing:
        raise ValueError(f"image_risk is missing keys: {sorted(missing)}")
    if image_risk["image_risk_level"] not in _LEVEL_RANK:
        raise ValueError(f"image_risk_level must be one of {sorted(_LEVEL_RANK)}, got {image_risk['image_risk_level']!r}")
    for key in ("p_scc", "p_ak"):
        value = image_risk[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be a probability between 0 and 1, got {value!r}")


def apply_safety_net(image_risk: Mapping[str, Any]) -> dict:
    """Apply the B.4 escalation rule to the image-side risk only."""
    _validate_image_risk(image_risk)
    raw_level = image_risk["image_risk_level"]
    p_scc, p_ak = float(image_risk["p_scc"]), float(image_risk["p_ak"])

    level_rank = _LEVEL_RANK[raw_level]
    escalated = False
    explanation = None

    if p_scc >= SCC_ESCALATION_THRESHOLD and level_rank < _LEVEL_RANK["high"]:
        level_rank, escalated = _LEVEL_RANK["high"], True
        explanation = (
            f"Safety net: the image model gave a {p_scc:.0%} chance of squamous cell carcinoma, at or above the "
            f"{SCC_ESCALATION_THRESHOLD:.0%} caution threshold, so the risk was raised from '{raw_level}' to 'high' "
            "even though another label scored higher."
        )
    elif p_ak >= AK_ESCALATION_THRESHOLD and level_rank < _LEVEL_RANK["moderate"]:
        level_rank, escalated = _LEVEL_RANK["moderate"], True
        explanation = (
            f"Safety net: the image model gave a {p_ak:.0%} chance of actinic keratosis (a precancerous lesion), at or "
            f"above the {AK_ESCALATION_THRESHOLD:.0%} caution threshold, so the risk was raised from '{raw_level}' to 'moderate'."
        )

    return {
        "image_risk_level": _RANK_TO_LEVEL[level_rank],
        "raw_image_risk_level": raw_level,
        "p_scc": p_scc,
        "p_ak": p_ak,
        "safety_escalated": escalated,
        "safety_explanation": explanation,
    }


def fuse_risk(image_risk: Mapping[str, Any], exposure_risk: Mapping[str, Any]) -> dict:
    """Build the final response. Image and exposure components stay separate; the recommendation
    is driven by whichever component is more urgent, and says which one that was."""
    image = apply_safety_net(image_risk)
    exposure = {
        "exposure_risk_score": exposure_risk["exposure_risk_score"],
        "exposure_risk_level": exposure_risk["exposure_risk_level"],
        "explanation": list(exposure_risk["explanation"]),
    }
    for optional in ("routes_triggered", "breakdown"):
        if optional in exposure_risk:
            exposure[optional] = exposure_risk[optional]

    image_rank = _LEVEL_RANK[image["image_risk_level"]]
    exposure_rank = _LEVEL_RANK[exposure["exposure_risk_level"]]
    overall_rank = max(image_rank, exposure_rank)
    if image_rank == exposure_rank:
        basis = "both"
    else:
        basis = "image" if image_rank > exposure_rank else "exposure"

    overall_level = _RANK_TO_LEVEL[overall_rank]
    return {
        "image_risk": image,
        "exposure_risk": exposure,
        "recommendation": {
            "level": overall_level,
            "basis": basis,  # which component set the urgency; the two components are never averaged or merged
            "text": RECOMMENDATIONS[overall_level],
        },
        # Frontend shows a distinct amber notice when this is true.
        "safety_escalated": image["safety_escalated"],
    }
