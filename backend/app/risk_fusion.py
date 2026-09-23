"""
risk_fusion.py -- transparent, rules-based exposure-risk scoring for Nirikshan.

This is a readable point-scoring rubric, NOT a model. Every weight is a named constant with an
inline comment giving the clinical reasoning. Where no published number exists, the weight is
tagged "# ESTIMATE, validate with clinical partner" -- no fake citations.

Public API
----------
load_arsenic_districts(path=None) -> dict
assess_exposure_risk(payload, arsenic_districts) -> dict
    {
      "exposure_risk_score": int 0-100,
      "exposure_risk_level": "low" | "moderate" | "high",
      "explanation": [str, ...],          # human-readable reasons, each with its (+points)
      "routes_triggered": [str, ...],     # extra: which of the 4 exposure routes fired
      "breakdown": {route: points, ...},  # extra: per-route points for the UI / audits
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from .exposure_schema import ExposureHistory, FirePotUse, SunProtection, TobaccoUse, WaterSource

DEFAULT_DISTRICTS_PATH = Path(__file__).with_name("arsenic_districts.json")

# =========================================================================================
# LEVEL THRESHOLDS (score is 0-100, capped)
# =========================================================================================
# ESTIMATE, validate with clinical partner.
# Chosen so that ONE strongly documented route at full strength lands in "moderate"
# (e.g. a severe arsenic district = 30, a burn scar >= 20 yrs old = 30) and TWO stacked
# routes, or a scar with a non-healing ulcer, land in "high".
MODERATE_MIN = 25
HIGH_MIN = 50
SCORE_CAP = 100

# =========================================================================================
# ROUTE 1 -- Chronic contact-thermal injury (kangri / sigri / angithi / bukhari)
# =========================================================================================
# Kangri cancer is a decades-long process: repeated heat -> erythema ab igne -> SCC. The clinical
# pathway has been documented since 1879, but I have no published dose-response table, so the
# duration bands are estimates. Longer cumulative use = more chronic thermal damage.
FIRE_POT_POINTS_UNDER_5_YEARS = 5    # ESTIMATE, validate with clinical partner (early exposure; EAI possible, SCC unlikely yet)
FIRE_POT_POINTS_5_TO_19_YEARS = 15   # ESTIMATE, validate with clinical partner (established EAI territory)
FIRE_POT_POINTS_20_PLUS_YEARS = 25   # ESTIMATE, validate with clinical partner (decades of use = classic kangri-cancer history)

# =========================================================================================
# ROUTE 2 -- Burn-scar malignant transformation (Marjolin's ulcer)
# =========================================================================================
# Context file: SCC is 70-96% of Marjolin cases; average latency 11-35 years; ~1-2% of burn scars
# transform; higher recurrence/metastasis than ordinary SCC. Bands follow that latency window.
SCAR_POINTS_UNDER_10_YEARS = 5    # ESTIMATE, validate with clinical partner (below the reported 11-year lower latency bound; rare acute forms exist)
SCAR_POINTS_10_TO_19_YEARS = 20   # ESTIMATE, validate with clinical partner (inside the reported 11-35 yr latency window)
SCAR_POINTS_20_PLUS_YEARS = 30    # ESTIMATE, validate with clinical partner (mid-to-late latency window; long-standing scars carry most transformation)
# A non-healing ulcer/sore/growth INSIDE a burn scar is the classic presentation of Marjolin's ulcer,
# and the transformed lesion is more aggressive than ordinary SCC, so it is weighted very heavily.
SCAR_NONHEALING_ULCER_POINTS = 25  # ESTIMATE, validate with clinical partner
# Safety floor: a non-healing lesion in a scar is escalated straight to "high" regardless of other answers.
SCAR_NONHEALING_ULCER_FLOOR_SCORE = HIGH_MIN

# =========================================================================================
# ROUTE 3 -- Arsenic-contaminated groundwater
# =========================================================================================
# Base points by district tier (see arsenic_districts.json "_tiers"). Tier reflects how widespread
# contamination above the 0.05 mg/L limit is. 30 for "severe" matches the example weight in the spec.
ARSENIC_POINTS_BY_TIER = {
    "severe": 30,      # ESTIMATE, validate with clinical partner (whole-district contamination; 70M+ people in the Bengal delta drink above-limit water)
    "documented": 20,  # ESTIMATE, validate with clinical partner (contamination above 0.05 mg/L in parts of the district -> individual exposure is patchy)
    "elevated": 10,    # ESTIMATE, validate with clinical partner (above WHO 0.01 mg/L but not widely above 0.05 mg/L)
}
ARSENIC_TIER_TEXT = {
    "severe": "a severely arsenic-affected groundwater zone",
    "documented": "a documented arsenic-endemic groundwater zone",
    "elevated": "a documented zone with groundwater arsenic above the WHO guideline",
}
# CGWB: arsenic sits mainly in shallow aquifers (<~100 m); deep aquifers are largely free.
# Treated/filtered water and surface water also bypass the shallow-tubewell pathway. Halve the district weight.
ARSENIC_SAFER_SOURCES = {
    WaterSource.DEEP_BOREWELL,
    WaterSource.PIPED_TREATED,
    WaterSource.BOTTLED_OR_FILTERED,
    WaterSource.SURFACE_WATER,
}
ARSENIC_SAFER_SOURCE_MULTIPLIER = 0.5  # ESTIMATE, validate with clinical partner (well/treatment quality is unverified, so reduced not removed)
# Arsenical skin lesions and cancers follow years-to-decades of cumulative exposure, so a short
# residence history in the district discounts the weight.
ARSENIC_SHORT_EXPOSURE_YEARS = 10
ARSENIC_SHORT_EXPOSURE_MULTIPLIER = 0.5  # ESTIMATE, validate with clinical partner

# =========================================================================================
# ROUTE 4 -- Chronic occupational / lifestyle UV (actinic keratosis -> SCC)
# =========================================================================================
# Cumulative UV dose is the established driver of AK and SCC; hours/day x years is the practical proxy.
UV_POINTS_HEAVY = 15     # ESTIMATE, validate with clinical partner (>= 6 h/day for >= 10 years: full-time outdoor labor, cumulative dose is high)
UV_HEAVY_HOURS, UV_HEAVY_YEARS = 6, 10
UV_POINTS_MODERATE = 10  # ESTIMATE, validate with clinical partner (>= 4 h/day for >= 5 years)
UV_MODERATE_HOURS, UV_MODERATE_YEARS = 4, 5
UV_POINTS_LIGHT = 5      # ESTIMATE, validate with clinical partner (>= 2 h/day for >= 1 year)
UV_LIGHT_HOURS, UV_LIGHT_YEARS = 2, 1
# Never using cover/sunscreen removes the main modifiable protection for the same outdoor hours.
UV_NO_PROTECTION_POINTS = 5  # ESTIMATE, validate with clinical partner
UV_ROUTE_CAP = 20  # keeps UV (the most common but lowest-individual-risk route) from dominating

# =========================================================================================
# GENERAL MODIFIERS (cap keeps them from swamping the four exposure routes)
# =========================================================================================
AGE_60_PLUS_POINTS = 8   # ESTIMATE, validate with clinical partner (SCC/AK incidence rises steeply with age; reflects cumulative lifetime exposure)
AGE_45_TO_59_POINTS = 4  # ESTIMATE, validate with clinical partner
FAMILY_HISTORY_POINTS = 5  # ESTIMATE, validate with clinical partner (first-degree relative with skin cancer: modestly raised risk)
TOBACCO_SMOKING_POINTS = 5   # ESTIMATE, validate with clinical partner (smoking is an established SCC risk factor)
TOBACCO_SMOKELESS_POINTS = 3  # ESTIMATE, validate with clinical partner (smokeless tobacco: weaker skin-SCC evidence)
MODIFIERS_CAP = 15

ROUTE_LABELS = {
    "fire_pot": "chronic contact-thermal injury (fire-pot/heater)",
    "burn_scar": "burn-scar malignant transformation (Marjolin's ulcer)",
    "arsenic": "arsenic-contaminated groundwater",
    "occupational_uv": "chronic occupational UV exposure",
}


# =========================================================================================
# District lookup
# =========================================================================================
def load_arsenic_districts(path: Optional[Union[str, Path]] = None) -> dict:
    """Load arsenic_districts.json (defaults to the file next to this module)."""
    with open(path or DEFAULT_DISTRICTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _norm(text: Optional[str]) -> str:
    """Lowercase and strip everything except letters/digits: 'Cooch-Behar ' -> 'coochbehar'."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


_TIER_RANK = {"elevated": 0, "documented": 1, "severe": 2}


def _entries(arsenic_districts: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]) -> list:
    if isinstance(arsenic_districts, Mapping):
        return list(arsenic_districts.get("districts", []))
    return list(arsenic_districts)


def find_arsenic_district(
    district: str,
    state: Optional[str],
    arsenic_districts: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]],
) -> Optional[Mapping[str, Any]]:
    """Return the matching district entry (highest tier if several match), or None.
    Matches on official name or any alias; if a state is given it must also match."""
    wanted = _norm(district)
    wanted_state = _norm(state)
    matches = []
    for entry in _entries(arsenic_districts):
        names = {_norm(entry["district"])} | {_norm(a) for a in entry.get("aliases", [])}
        if wanted not in names:
            continue
        if wanted_state and _norm(entry.get("state")) != wanted_state:
            continue
        matches.append(entry)
    return max(matches, key=lambda e: _TIER_RANK[e["tier"]]) if matches else None


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


# =========================================================================================
# Scoring
# =========================================================================================
def assess_exposure_risk(
    payload: Union[ExposureHistory, Mapping[str, Any]],
    arsenic_districts: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]],
) -> dict:
    """Score an exposure-history payload against the rubric above. Deterministic and explainable."""
    history = payload if isinstance(payload, ExposureHistory) else ExposureHistory(**payload)

    points = {"fire_pot": 0, "burn_scar": 0, "arsenic": 0, "occupational_uv": 0, "modifiers": 0}
    reasons: dict = {key: [] for key in points}

    # ---- Route 1: fire-pot / heater ------------------------------------------------------
    if history.fire_pot_use != FirePotUse.NEVER:
        years = history.fire_pot_years
        when = "current" if history.fire_pot_use == FirePotUse.CURRENT else "past"
        if years >= 20:
            pts = FIRE_POT_POINTS_20_PLUS_YEARS
        elif years >= 5:
            pts = FIRE_POT_POINTS_5_TO_19_YEARS
        else:
            pts = FIRE_POT_POINTS_UNDER_5_YEARS
        points["fire_pot"] = pts
        reasons["fire_pot"].append(
            f"{when.capitalize()} regular fire-pot/heater use for {years} year(s) "
            f"(chronic contact-thermal injury -> erythema ab igne -> SCC) (+{pts})"
        )

    # ---- Route 2: burn scar / Marjolin's ulcer -------------------------------------------
    scar_ulcer = False
    if history.has_burn_scar:
        scar_age = history.burn_scar_age_years or 0
        if scar_age >= 20:
            pts = SCAR_POINTS_20_PLUS_YEARS
        elif scar_age >= 10:
            pts = SCAR_POINTS_10_TO_19_YEARS
        else:
            pts = SCAR_POINTS_UNDER_10_YEARS
        points["burn_scar"] += pts
        reasons["burn_scar"].append(
            f"Burn scar/chronic wound present for {scar_age} year(s); malignant transformation "
            f"(Marjolin's ulcer) typically appears after 11-35 years (+{pts})"
        )
        if history.burn_scar_nonhealing_ulcer:
            scar_ulcer = True
            points["burn_scar"] += SCAR_NONHEALING_ULCER_POINTS
            reasons["burn_scar"].append(
                "Non-healing sore, ulcer or new growth inside a burn scar is the classic sign of "
                f"Marjolin's ulcer and needs prompt clinical review (+{SCAR_NONHEALING_ULCER_POINTS})"
            )

    # ---- Route 3: arsenic groundwater ----------------------------------------------------
    entry = find_arsenic_district(history.district, history.state, arsenic_districts)
    if entry is not None:
        base = ARSENIC_POINTS_BY_TIER[entry["tier"]]
        multiplier = 1.0
        adjustments = []
        if history.water_source in ARSENIC_SAFER_SOURCES:
            multiplier *= ARSENIC_SAFER_SOURCE_MULTIPLIER
            adjustments.append(
                f"water source '{history.water_source.value}' bypasses shallow arsenic-bearing aquifers"
            )
        if history.years_at_water_source is not None and history.years_at_water_source < ARSENIC_SHORT_EXPOSURE_YEARS:
            multiplier *= ARSENIC_SHORT_EXPOSURE_MULTIPLIER
            adjustments.append(
                f"only {history.years_at_water_source} year(s) at this water source (arsenical lesions need long exposure)"
            )
        pts = _round_half_up(base * multiplier)
        points["arsenic"] = pts
        reasons["arsenic"].append(
            f"{entry['district']} ({entry['state']}) is {ARSENIC_TIER_TEXT[entry['tier']]} (+{base})"
        )
        if adjustments:
            reasons["arsenic"].append(
                f"Adjusted down because {' and '.join(adjustments)} (-{base - pts}, net +{pts})"
            )

    # ---- Route 4: occupational UV --------------------------------------------------------
    hours, yrs = history.outdoor_hours_per_day, history.outdoor_work_years
    uv_pts = 0
    if hours >= UV_HEAVY_HOURS and yrs >= UV_HEAVY_YEARS:
        uv_pts = UV_POINTS_HEAVY
    elif hours >= UV_MODERATE_HOURS and yrs >= UV_MODERATE_YEARS:
        uv_pts = UV_POINTS_MODERATE
    elif hours >= UV_LIGHT_HOURS and yrs >= UV_LIGHT_YEARS:
        uv_pts = UV_POINTS_LIGHT
    if uv_pts:
        reasons["occupational_uv"].append(
            f"Outdoor work about {hours:g} h/day for {yrs} year(s): cumulative UV exposure -> actinic keratosis -> SCC (+{uv_pts})"
        )
        if history.sun_protection == SunProtection.NEVER:
            uv_pts += UV_NO_PROTECTION_POINTS
            reasons["occupational_uv"].append(
                f"Never uses hat/cloth cover/sunscreen while outdoors (+{UV_NO_PROTECTION_POINTS})"
            )
        uv_pts = min(uv_pts, UV_ROUTE_CAP)
    points["occupational_uv"] = uv_pts

    # ---- General modifiers ---------------------------------------------------------------
    mod = 0
    if history.age_years >= 60:
        mod += AGE_60_PLUS_POINTS
        reasons["modifiers"].append(f"Age {history.age_years}: skin-cancer incidence rises steeply after 60 (+{AGE_60_PLUS_POINTS})")
    elif history.age_years >= 45:
        mod += AGE_45_TO_59_POINTS
        reasons["modifiers"].append(f"Age {history.age_years}: cumulative lifetime exposure is rising (+{AGE_45_TO_59_POINTS})")
    if history.family_history_skin_cancer:
        mod += FAMILY_HISTORY_POINTS
        reasons["modifiers"].append(f"Family history of skin cancer (+{FAMILY_HISTORY_POINTS})")
    if history.tobacco_use in (TobaccoUse.SMOKING, TobaccoUse.BOTH):
        mod += TOBACCO_SMOKING_POINTS
        reasons["modifiers"].append(f"Tobacco smoking is an established squamous-cell cancer risk factor (+{TOBACCO_SMOKING_POINTS})")
    elif history.tobacco_use == TobaccoUse.SMOKELESS:
        mod += TOBACCO_SMOKELESS_POINTS
        reasons["modifiers"].append(f"Smokeless tobacco use (+{TOBACCO_SMOKELESS_POINTS})")
    if mod > MODIFIERS_CAP:
        reasons["modifiers"].append(f"Modifier points capped at {MODIFIERS_CAP} (was {mod})")
        mod = MODIFIERS_CAP
    points["modifiers"] = mod

    # ---- Total, floors, level ------------------------------------------------------------
    total = min(sum(points.values()), SCORE_CAP)
    explanation = [line for key in ("fire_pot", "burn_scar", "arsenic", "occupational_uv", "modifiers") for line in reasons[key]]

    if scar_ulcer and total < SCAR_NONHEALING_ULCER_FLOOR_SCORE:
        explanation.append(
            f"Score raised to the 'high' floor ({SCAR_NONHEALING_ULCER_FLOOR_SCORE}) because of a non-healing lesion inside a burn scar"
        )
        total = SCAR_NONHEALING_ULCER_FLOOR_SCORE

    if total >= HIGH_MIN:
        level = "high"
    elif total >= MODERATE_MIN:
        level = "moderate"
    else:
        level = "low"

    routes = [key for key in ROUTE_LABELS if points[key] > 0]
    if len(routes) > 1:
        explanation.append(
            f"{len(routes)} independent exposure routes apply, and their risks stack: "
            + "; ".join(ROUTE_LABELS[r] for r in routes)
        )
    if not explanation:
        explanation.append("No documented exposure risk factors were reported.")

    return {
        "exposure_risk_score": total,
        "exposure_risk_level": level,
        "explanation": explanation,
        "routes_triggered": routes,
        "breakdown": dict(points),
    }
