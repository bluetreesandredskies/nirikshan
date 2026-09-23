"""
exposure_schema.py -- Pydantic model for the Nirikshan exposure-history intake form.

15 fields, ~90 seconds to fill. Covers the four exposure routes in PROJECT_CONTEXT
section 2 (fire-pot / contact-thermal injury, burn scars / Marjolin's ulcer,
arsenic-contaminated groundwater, chronic occupational UV) plus a few
well-established general modifiers (age, family history, tobacco).

The enum *values* below are stable machine identifiers. The i18n string tables in
backend/i18n/exposure_form_*.json use them as keys (e.g. "water_source.option.deep_borewell"),
so do not rename them without updating both language files.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FirePotUse(str, Enum):
    """Kangri / sigri / angithi / bukhari (contact-thermal exposure -> erythema ab igne)."""

    NEVER = "never"
    PAST = "past"
    CURRENT = "current"


class WaterSource(str, Enum):
    """Main drinking-water source at home. Shallow sources are where arsenic concentrates
    (CGWB: contamination is mainly in aquifers up to ~100 m; deeper aquifers are largely free)."""

    HANDPUMP_TUBEWELL = "handpump_tubewell"
    DEEP_BOREWELL = "deep_borewell"
    DUG_WELL = "dug_well"
    PIPED_TREATED = "piped_treated"
    BOTTLED_OR_FILTERED = "bottled_or_filtered"
    SURFACE_WATER = "surface_water"
    OTHER_UNKNOWN = "other_unknown"


class SunProtection(str, Enum):
    """How often the person uses a hat/cloth cover, long sleeves or sunscreen outdoors."""

    REGULAR = "regular"
    SOMETIMES = "sometimes"
    NEVER = "never"


class TobaccoUse(str, Enum):
    NONE = "none"
    SMOKELESS = "smokeless"
    SMOKING = "smoking"
    BOTH = "both"


class ExposureHistory(BaseModel):
    """Validated intake-form payload. Extra/unknown keys are rejected so typos surface early."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- 1. Demographics ---------------------------------------------------------------
    age_years: int = Field(..., ge=0, le=120, description="Age in completed years.")

    # --- 2-3. Chronic contact-thermal injury (fire-pot / heater) -----------------------
    fire_pot_use: FirePotUse = Field(
        FirePotUse.NEVER, description="Kangri/sigri/angithi/bukhari use: never, in the past, or currently."
    )
    fire_pot_years: int = Field(
        0, ge=0, le=100, description="Total years of regular fire-pot/heater use (0 if never)."
    )

    # --- 4-6. Burn scar / chronic wound (Marjolin's ulcer route) -----------------------
    has_burn_scar: bool = Field(False, description="Any old burn scar or chronic wound.")
    burn_scar_age_years: Optional[int] = Field(
        None, ge=0, le=100, description="Years since the burn/wound occurred (required if has_burn_scar)."
    )
    burn_scar_nonhealing_ulcer: bool = Field(
        False,
        description="Is there a non-healing sore, ulcer, or new growth inside the scar? (Marjolin's warning sign)",
    )

    # --- 7-9. Arsenic-contaminated groundwater ------------------------------------------
    water_source: WaterSource = Field(..., description="Main drinking-water source at home.")
    district: str = Field(..., min_length=1, max_length=100, description="Home district.")
    state: Optional[str] = Field(
        None, max_length=100, description="Home state (optional; disambiguates district names)."
    )
    years_at_water_source: Optional[int] = Field(
        None, ge=0, le=120, description="Years drinking from this source / living in this district."
    )

    # --- 10-12. Chronic occupational / lifestyle UV --------------------------------------
    outdoor_hours_per_day: float = Field(
        0.0, ge=0, le=24, description="Average hours per day spent outdoors working (farming, construction, fishing, vending...)."
    )
    outdoor_work_years: int = Field(0, ge=0, le=80, description="Years spent in outdoor work.")
    sun_protection: SunProtection = Field(..., description="Regular use of hat/cloth cover/sunscreen outdoors.")

    # --- 13-14. General modifiers ---------------------------------------------------------
    family_history_skin_cancer: bool = Field(False, description="Skin cancer in a parent, sibling or child.")
    tobacco_use: TobaccoUse = Field(TobaccoUse.NONE, description="Tobacco use (smoking and/or smokeless).")

    # -------------------------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_consistency(self) -> "ExposureHistory":
        # Fire-pot answers must agree with each other.
        if self.fire_pot_use == FirePotUse.NEVER and self.fire_pot_years > 0:
            raise ValueError("fire_pot_years must be 0 when fire_pot_use is 'never'.")
        if self.fire_pot_use != FirePotUse.NEVER and self.fire_pot_years < 1:
            raise ValueError("fire_pot_years must be at least 1 when fire_pot_use is 'past' or 'current'.")

        # Burn-scar answers must agree with each other.
        if self.has_burn_scar and self.burn_scar_age_years is None:
            raise ValueError("burn_scar_age_years is required when has_burn_scar is true.")
        if not self.has_burn_scar and (self.burn_scar_age_years is not None or self.burn_scar_nonhealing_ulcer):
            raise ValueError(
                "burn_scar_age_years / burn_scar_nonhealing_ulcer must be empty when has_burn_scar is false."
            )

        # No duration can exceed the person's age.
        durations = {
            "fire_pot_years": self.fire_pot_years,
            "burn_scar_age_years": self.burn_scar_age_years,
            "years_at_water_source": self.years_at_water_source,
            "outdoor_work_years": self.outdoor_work_years,
        }
        for name, value in durations.items():
            if value is not None and value > self.age_years:
                raise ValueError(f"{name} ({value}) cannot exceed age_years ({self.age_years}).")
        return self
