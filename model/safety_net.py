"""
model/safety_net.py

Nirikshan — the safety-net escalation rule (PROJECT_CONTEXT.md §6, VERBATIM,
load-bearing logic). Carried over unchanged from the proven prototype.

Rule, restated exactly:
  - After Stage 2 produces softmax probabilities, check p_scc and p_ak
    specifically -- not just the top-1 label.
  - If p_scc >= 15% (tunable constant) and the base risk isn't already
    "high", escalate to "high", set safety_escalated: true, return a
    human-readable explanation.
  - A parallel, slightly lower threshold applies for p_ak, escalating
    "low"/"none" to "moderate".
  - The displayed recommendation always matches the *post-escalation* risk
    level, never the raw top-1 guess.

This module is deliberately pure logic with no model / tensor / I/O
dependencies, so it can be exhaustively unit tested (see
tests/test_safety_net.py) independent of the model forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable constants (named, per the spec -- do not inline magic numbers)
# ---------------------------------------------------------------------------

# If p_scc is at or above this fraction, risk is escalated to "high"
# regardless of the top-1 predicted class, because false negatives on SCC
# (an already-malignant, potentially metastatic lesion) are the single
# costliest failure mode this system can have.
P_SCC_ESCALATION_THRESHOLD: float = 0.15

# Deliberately a slightly LOWER bar than the SCC threshold: actinic
# keratosis is the precancerous precursor, not the malignancy itself, so
# it's appropriate to flag it (escalate to "moderate") at a somewhat lower
# predicted probability than we require to force a "high" SCC escalation --
# erring toward catching it early rather than missing it.
P_AK_ESCALATION_THRESHOLD: float = 0.12

# Canonical risk levels, in increasing severity order. The safety net only
# ever moves risk *up* this list, never down.
RISK_LEVELS: tuple[str, ...] = ("none", "low", "moderate", "high")


def _risk_rank(risk_level: str) -> int:
    try:
        return RISK_LEVELS.index(risk_level)
    except ValueError as exc:
        raise ValueError(
            f"Unknown risk level '{risk_level}'. Must be one of {RISK_LEVELS}."
        ) from exc


@dataclass
class SafetyNetResult:
    """Structured return value of apply_safety_net()."""

    base_risk: str
    risk_level: str
    safety_escalated: bool
    p_scc: float
    p_ak: float
    explanation: Optional[str]
    escalation_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "base_risk": self.base_risk,
            "risk_level": self.risk_level,
            "safety_escalated": self.safety_escalated,
            "p_scc": self.p_scc,
            "p_ak": self.p_ak,
            "explanation": self.explanation,
            "escalation_reasons": self.escalation_reasons,
        }


def apply_safety_net(
    p_scc: float,
    p_ak: float,
    base_risk: str,
    p_scc_threshold: float = P_SCC_ESCALATION_THRESHOLD,
    p_ak_threshold: float = P_AK_ESCALATION_THRESHOLD,
) -> SafetyNetResult:
    """
    Applies the §6 safety-net escalation rule.

    Args:
        p_scc: Stage 2 softmax probability for squamous_cell_carcinoma, in [0, 1].
        p_ak: Stage 2 softmax probability for actinic_keratosis, in [0, 1].
        base_risk: the risk level computed from the model's top-1 prediction
            (or upstream fusion logic) BEFORE this safety-net check is
            applied. Must be one of RISK_LEVELS.
        p_scc_threshold: tunable constant, see P_SCC_ESCALATION_THRESHOLD.
        p_ak_threshold: tunable constant, see P_AK_ESCALATION_THRESHOLD.

    Returns:
        SafetyNetResult with the POST-escalation risk_level (the only risk
        level that should ever be shown to the user or passed downstream),
        a safety_escalated flag, and a human-readable explanation of *why*
        if escalation fired.

    Both conditions are evaluated independently against the ORIGINAL
    base_risk (not chained sequentially), so that "both fired" is reported
    truthfully even when one of them wouldn't have changed the final
    outcome. The final risk_level is then the most severe of: base_risk,
    "high" (if the SCC condition fired), "moderate" (if the AK condition
    fired). This means:
        - If only SCC fires: risk_level -> "high".
        - If only AK fires: risk_level -> "moderate" (never higher).
        - If both fire: risk_level -> "high" (SCC's outcome dominates,
          since "high" outranks "moderate"), but BOTH explanations are
          still returned so the person sees the full clinical picture,
          not just whichever one happened to "win".
        - If base_risk is already "high": the SCC condition cannot fire
          (there's nothing higher to escalate to), and is correctly a
          no-op.
        - If base_risk is already "moderate" or higher: the AK condition
          (which only ever escalates "none"/"low") cannot fire, and is
          correctly a no-op.
    """
    for value, name in ((p_scc, "p_scc"), (p_ak, "p_ak")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if base_risk not in RISK_LEVELS:
        raise ValueError(f"base_risk must be one of {RISK_LEVELS}, got '{base_risk}'")

    reasons: list[str] = []
    escalation_targets: list[str] = [base_risk]

    # --- SCC check: independently evaluated against base_risk -------------
    scc_fires = p_scc >= p_scc_threshold and _risk_rank(base_risk) < _risk_rank("high")
    if scc_fires:
        escalation_targets.append("high")
        reasons.append(
            f"Squamous cell carcinoma probability ({p_scc:.1%}) meets or exceeds "
            f"the {p_scc_threshold:.0%} safety threshold. Risk escalated to "
            f"'high' regardless of the model's top-1 prediction, to minimize "
            f"the chance of missing a malignant lesion."
        )

    # --- AK check: independently evaluated against base_risk --------------
    ak_fires = p_ak >= p_ak_threshold and base_risk in ("none", "low")
    if ak_fires:
        escalation_targets.append("moderate")
        reasons.append(
            f"Actinic keratosis probability ({p_ak:.1%}) meets or exceeds the "
            f"{p_ak_threshold:.0%} safety threshold. Risk escalated to "
            f"'moderate', since AK is a precancerous precursor that warrants "
            f"a closer look even when it isn't the top prediction."
        )

    risk_level = max(escalation_targets, key=_risk_rank)
    safety_escalated = scc_fires or ak_fires
    explanation = " ".join(reasons) if reasons else None

    return SafetyNetResult(
        base_risk=base_risk,
        risk_level=risk_level,
        safety_escalated=safety_escalated,
        p_scc=p_scc,
        p_ak=p_ak,
        explanation=explanation,
        escalation_reasons=reasons,
    )
