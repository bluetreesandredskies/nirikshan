"""Tests for the exposure-history risk fusion engine. Run from the repo root: pytest tests/"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.exposure_schema import ExposureHistory, FirePotUse, WaterSource  # noqa: E402
from backend.app.fusion import fuse_risk  # noqa: E402
from backend.app.risk_fusion import assess_exposure_risk, find_arsenic_district, load_arsenic_districts  # noqa: E402


@pytest.fixture(scope="module")
def districts():
    return load_arsenic_districts()


def make_payload(**overrides):
    """A baseline low-risk person; tests override only what they care about."""
    base = {
        "age_years": 30,
        "fire_pot_use": "never",
        "fire_pot_years": 0,
        "has_burn_scar": False,
        "water_source": "piped_treated",
        "district": "New Delhi",
        "sun_protection": "regular",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------------------
# 1. Low-risk case
# --------------------------------------------------------------------------------------
def test_low_risk_case(districts):
    result = assess_exposure_risk(make_payload(), districts)
    assert result["exposure_risk_score"] == 0
    assert result["exposure_risk_level"] == "low"
    assert result["routes_triggered"] == []
    assert result["explanation"] == ["No documented exposure risk factors were reported."]


# --------------------------------------------------------------------------------------
# 2. High-arsenic district case
# --------------------------------------------------------------------------------------
def test_severe_arsenic_district_scores_30_and_explains_it(districts):
    payload = make_payload(district="Murshidabad", state="West Bengal", water_source="handpump_tubewell", years_at_water_source=25)
    result = assess_exposure_risk(payload, districts)
    assert result["exposure_risk_score"] == 30
    assert result["exposure_risk_level"] == "moderate"
    assert result["routes_triggered"] == ["arsenic"]
    assert any("Murshidabad" in line and "(+30)" in line for line in result["explanation"])


def test_arsenic_weight_is_reduced_for_deep_borewell_and_short_residence(districts):
    deep = assess_exposure_risk(make_payload(district="Murshidabad", water_source="deep_borewell", years_at_water_source=25), districts)
    assert deep["exposure_risk_score"] == 15
    short = assess_exposure_risk(make_payload(district="Murshidabad", water_source="handpump_tubewell", years_at_water_source=3), districts)
    assert short["exposure_risk_score"] == 15
    both = assess_exposure_risk(make_payload(district="Murshidabad", water_source="deep_borewell", years_at_water_source=3), districts)
    assert both["exposure_risk_score"] == 8  # 30 * 0.25 = 7.5 rounds half up


def test_district_matching_handles_aliases_state_and_unknowns(districts):
    assert find_arsenic_district("Maldah", None, districts)["district"] == "Malda"
    assert find_arsenic_district("  hugli ", "west bengal", districts)["district"] == "Hooghly"
    assert find_arsenic_district("Malda", "Bihar", districts) is None  # wrong state
    assert find_arsenic_district("Jaipur", None, districts) is None
    unknown = assess_exposure_risk(make_payload(district="Jaipur", water_source="handpump_tubewell"), districts)
    assert unknown["exposure_risk_score"] == 0


# --------------------------------------------------------------------------------------
# 3. Safety-net escalated case (image side only)
# --------------------------------------------------------------------------------------
def test_safety_net_escalates_scc_and_keeps_components_separate(districts):
    exposure = assess_exposure_risk(make_payload(), districts)  # low exposure
    response = fuse_risk({"image_risk_level": "low", "p_scc": 0.20, "p_ak": 0.05}, exposure)

    assert response["image_risk"]["image_risk_level"] == "high"
    assert response["image_risk"]["raw_image_risk_level"] == "low"
    assert response["image_risk"]["safety_escalated"] is True
    assert "20%" in response["image_risk"]["safety_explanation"]
    assert response["safety_escalated"] is True

    # Exposure side untouched and still separate.
    assert response["exposure_risk"]["exposure_risk_level"] == "low"
    assert response["exposure_risk"]["exposure_risk_score"] == 0
    assert "final_score" not in response

    # Recommendation matches the POST-escalation level.
    assert response["recommendation"]["level"] == "high"
    assert response["recommendation"]["basis"] == "image"


def test_safety_net_thresholds_and_non_escalation(districts):
    exposure = assess_exposure_risk(make_payload(), districts)
    just_below = fuse_risk({"image_risk_level": "low", "p_scc": 0.14, "p_ak": 0.05}, exposure)
    assert just_below["image_risk"]["safety_escalated"] is False
    assert just_below["image_risk"]["image_risk_level"] == "low"

    ak = fuse_risk({"image_risk_level": "none", "p_scc": 0.02, "p_ak": 0.12}, exposure)
    assert ak["image_risk"]["image_risk_level"] == "moderate"
    assert ak["image_risk"]["safety_escalated"] is True

    already_high = fuse_risk({"image_risk_level": "high", "p_scc": 0.90, "p_ak": 0.05}, exposure)
    assert already_high["image_risk"]["safety_escalated"] is False


def test_high_exposure_does_not_escalate_the_image_side(districts):
    exposure = assess_exposure_risk(
        make_payload(fire_pot_use="current", fire_pot_years=25, district="Ballia", water_source="handpump_tubewell"), districts
    )
    assert exposure["exposure_risk_level"] == "high"
    response = fuse_risk({"image_risk_level": "low", "p_scc": 0.01, "p_ak": 0.01}, exposure)
    assert response["image_risk"]["image_risk_level"] == "low"
    assert response["image_risk"]["safety_escalated"] is False
    assert response["recommendation"]["level"] == "high"
    assert response["recommendation"]["basis"] == "exposure"


def test_invalid_image_risk_is_rejected(districts):
    exposure = assess_exposure_risk(make_payload(), districts)
    with pytest.raises(ValueError):
        fuse_risk({"image_risk_level": "low", "p_scc": 1.5, "p_ak": 0.0}, exposure)
    with pytest.raises(ValueError):
        fuse_risk({"image_risk_level": "low", "p_scc": 0.1}, exposure)


# --------------------------------------------------------------------------------------
# 4. Multiple exposure routes stacking
# --------------------------------------------------------------------------------------
def test_two_routes_stack_to_high(districts):
    payload = make_payload(
        age_years=50, fire_pot_use="current", fire_pot_years=25,
        district="Nadia", water_source="handpump_tubewell", years_at_water_source=40,
    )
    result = assess_exposure_risk(payload, districts)
    # 25 (fire-pot 20+ yrs) + 30 (severe arsenic) + 4 (age 45-59) = 59
    assert result["exposure_risk_score"] == 59
    assert result["exposure_risk_level"] == "high"
    assert result["routes_triggered"] == ["fire_pot", "arsenic"]
    assert any("stack" in line for line in result["explanation"])


def test_all_four_routes_are_capped_at_100(districts):
    payload = make_payload(
        age_years=65, fire_pot_use="past", fire_pot_years=30,
        has_burn_scar=True, burn_scar_age_years=25, burn_scar_nonhealing_ulcer=True,
        district="Ballia", water_source="handpump_tubewell", years_at_water_source=50,
        outdoor_hours_per_day=8, outdoor_work_years=30, sun_protection="never",
        family_history_skin_cancer=True, tobacco_use="both",
    )
    result = assess_exposure_risk(payload, districts)
    assert result["exposure_risk_score"] == 100
    assert result["exposure_risk_level"] == "high"
    assert set(result["routes_triggered"]) == {"fire_pot", "burn_scar", "arsenic", "occupational_uv"}
    assert result["breakdown"]["modifiers"] == 15  # 8 + 5 + 5 = 18, capped at 15


def test_nonhealing_ulcer_in_burn_scar_forces_high(districts):
    payload = make_payload(has_burn_scar=True, burn_scar_age_years=2, burn_scar_nonhealing_ulcer=True)
    result = assess_exposure_risk(payload, districts)
    assert result["exposure_risk_level"] == "high"
    assert result["exposure_risk_score"] == 50
    assert any("high' floor" in line for line in result["explanation"])


def test_old_burn_scar_alone_is_moderate(districts):
    result = assess_exposure_risk(make_payload(age_years=45, has_burn_scar=True, burn_scar_age_years=22), districts)
    assert result["exposure_risk_score"] == 34  # 30 scar + 4 age
    assert result["exposure_risk_level"] == "moderate"


# --------------------------------------------------------------------------------------
# Schema and data-file integrity
# --------------------------------------------------------------------------------------
def test_schema_rejects_inconsistent_answers():
    with pytest.raises(ValidationError):
        ExposureHistory(**make_payload(fire_pot_use="never", fire_pot_years=5))
    with pytest.raises(ValidationError):
        ExposureHistory(**make_payload(fire_pot_use="current", fire_pot_years=0))
    with pytest.raises(ValidationError):
        ExposureHistory(**make_payload(has_burn_scar=True))  # scar age missing
    with pytest.raises(ValidationError):
        ExposureHistory(**make_payload(burn_scar_nonhealing_ulcer=True))  # ulcer without scar
    with pytest.raises(ValidationError):
        ExposureHistory(**make_payload(age_years=20, outdoor_work_years=30))  # longer than age
    with pytest.raises(ValidationError):
        ExposureHistory(**make_payload(unexpected_field=1))
    assert ExposureHistory(**make_payload()).water_source == WaterSource.PIPED_TREATED


def test_assess_accepts_model_instance_and_dict(districts):
    payload = make_payload(district="Kolkata", water_source="handpump_tubewell", years_at_water_source=20)
    from_dict = assess_exposure_risk(payload, districts)
    from_model = assess_exposure_risk(ExposureHistory(**payload), districts)
    assert from_dict == from_model
    assert from_dict["exposure_risk_score"] == 20


def test_arsenic_district_file_is_well_formed(districts):
    assert districts["_sources"]
    seen = set()
    for entry in districts["districts"]:
        assert entry["tier"] in districts["_tiers"]
        assert entry["state"] and entry["district"]
        assert entry["source_ids"] and all(s in districts["_sources"] for s in entry["source_ids"])
        key = (entry["state"], entry["district"])
        assert key not in seen
        seen.add(key)
    assert {"West Bengal", "Bihar", "Uttar Pradesh", "Assam"} <= {e["state"] for e in districts["districts"]}


def test_i18n_tables_have_identical_keys_and_cover_the_schema():
    i18n_dir = REPO_ROOT / "backend" / "i18n"
    en = json.loads((i18n_dir / "exposure_form_en.json").read_text(encoding="utf-8"))
    hi = json.loads((i18n_dir / "exposure_form_hi.json").read_text(encoding="utf-8"))
    assert set(en) == set(hi)
    assert all(isinstance(v, str) and v.strip() for v in list(en.values()) + list(hi.values()))
    for field in ExposureHistory.model_fields:
        assert f"{field}.label" in en, f"missing label for {field}"
    for option in FirePotUse:
        assert f"fire_pot_use.option.{option.value}" in en
    for option in WaterSource:
        assert f"water_source.option.{option.value}" in en
