"""
tests/test_safety_net.py

Unit tests for model/safety_net.py -- the §6 safety-net escalation rule.
Pure logic, no model/tensor dependencies, so these run fast and don't need
a GPU, checkpoints, or images.
"""

import pytest

from model.safety_net import (
    P_AK_ESCALATION_THRESHOLD,
    P_SCC_ESCALATION_THRESHOLD,
    apply_safety_net,
)


def test_no_escalation_when_both_probabilities_are_low():
    """
    Neither p_scc nor p_ak meets its threshold: risk_level should pass
    through unchanged and safety_escalated should be False.
    """
    result = apply_safety_net(p_scc=0.05, p_ak=0.03, base_risk="low")

    assert result.safety_escalated is False
    assert result.risk_level == "low"
    assert result.base_risk == "low"
    assert result.explanation is None
    assert result.escalation_reasons == []


def test_p_scc_escalation_to_high():
    """
    p_scc at/above threshold with a non-"high" base risk must escalate to
    "high", set safety_escalated True, and produce a non-empty explanation
    that mentions the SCC probability.
    """
    base_risk = "moderate"
    p_scc = P_SCC_ESCALATION_THRESHOLD + 0.03  # comfortably over the line
    result = apply_safety_net(p_scc=p_scc, p_ak=0.01, base_risk=base_risk)

    assert result.safety_escalated is True
    assert result.risk_level == "high"
    assert result.base_risk == base_risk
    assert result.explanation is not None
    assert "squamous cell carcinoma" in result.explanation.lower()
    assert len(result.escalation_reasons) == 1


def test_p_ak_escalation_to_moderate():
    """
    p_ak at/above threshold with base risk "none" or "low" must escalate to
    "moderate" (not higher), set safety_escalated True, and explain why.
    """
    base_risk = "low"
    p_ak = P_AK_ESCALATION_THRESHOLD + 0.02
    result = apply_safety_net(p_scc=0.01, p_ak=p_ak, base_risk=base_risk)

    assert result.safety_escalated is True
    assert result.risk_level == "moderate"
    assert result.base_risk == base_risk
    assert result.explanation is not None
    assert "actinic keratosis" in result.explanation.lower()
    assert len(result.escalation_reasons) == 1


def test_both_thresholds_fire_simultaneously():
    """
    When both p_scc and p_ak clear their thresholds, SCC's escalation to
    "high" must win (it's checked first and is the more severe outcome),
    the AK check must be a documented no-op once risk is already "high",
    and BOTH explanations should be recorded (not silently dropped), since
    the person should see the full clinical picture even though only the
    SCC one determined the final risk level.
    """
    base_risk = "none"
    p_scc = P_SCC_ESCALATION_THRESHOLD + 0.10
    p_ak = P_AK_ESCALATION_THRESHOLD + 0.10
    result = apply_safety_net(p_scc=p_scc, p_ak=p_ak, base_risk=base_risk)

    assert result.safety_escalated is True
    assert result.risk_level == "high"
    assert result.base_risk == base_risk
    assert len(result.escalation_reasons) == 2
    assert any("squamous cell carcinoma" in r.lower() for r in result.escalation_reasons)
    assert any("actinic keratosis" in r.lower() for r in result.escalation_reasons)
    assert result.explanation is not None


def test_no_escalation_when_base_risk_already_high():
    """
    If base_risk is already "high" (e.g. top-1 was already SCC), a further
    p_scc-threshold-clearing probability should not be reported as a new
    escalation -- risk was already at the ceiling.
    """
    p_scc = P_SCC_ESCALATION_THRESHOLD + 0.20
    result = apply_safety_net(p_scc=p_scc, p_ak=0.0, base_risk="high")

    assert result.risk_level == "high"
    assert result.safety_escalated is False
    assert result.explanation is None


def test_ak_check_is_a_no_op_once_risk_is_moderate_or_higher():
    """
    AK's escalation only fires from "none"/"low" -> "moderate". If base
    risk is already "moderate" (or higher) going in, and SCC doesn't fire,
    the AK check must not report an escalation.
    """
    p_ak = P_AK_ESCALATION_THRESHOLD + 0.10
    result = apply_safety_net(p_scc=0.0, p_ak=p_ak, base_risk="moderate")

    assert result.risk_level == "moderate"
    assert result.safety_escalated is False
    assert result.explanation is None


@pytest.mark.parametrize("bad_prob", [-0.01, 1.01])
def test_invalid_probability_raises(bad_prob):
    with pytest.raises(ValueError):
        apply_safety_net(p_scc=bad_prob, p_ak=0.0, base_risk="low")


def test_invalid_base_risk_raises():
    with pytest.raises(ValueError):
        apply_safety_net(p_scc=0.0, p_ak=0.0, base_risk="extreme")
