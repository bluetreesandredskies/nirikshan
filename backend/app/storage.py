"""
backend/app/storage.py

Minimal persistence for ANONYMIZED exposure-history submissions ONLY (PROJECT_CONTEXT D.1).
Images are NEVER persisted -- they live in memory for the duration of one request.

CONSENT LANGUAGE (DPDP Act 2023 framing, D.1 / Part E.4)
--------------------------------------------------------
Nothing is stored unless the user explicitly opts in (`consent_store_anonymized=true` on
/risk-fusion). The frontend must show wording equivalent to:

    "With your permission, we will save a few non-identifying details from this screening
     (your district and state, the risk level we calculated, and which exposure types
     applied) so that health planners can see where risk is concentrated. We do not save your
     photo, name, phone number, age, or exact answers. You can still use Nirikshan if you say
     no, and nothing will be saved."

What is stored (data minimisation): submission DATE (no time), state, district, exposure risk
level/score, which exposure routes fired, and the image-risk level. NOT stored: image, name,
contact details, IP address, device ID, age, free-text answers, or the raw questionnaire.

Aggregation (/risk-map) applies small-cell suppression: a district with fewer than
MIN_CELL_SIZE submissions is not listed individually, so a district with 1-2 users cannot
single anyone out.

NOTE: on Render's free tier the filesystem is ephemeral, so this SQLite file resets on every
redeploy/restart. Fine for the demo; move to a managed DB before real use.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

MIN_CELL_SIZE = 5  # small-cell suppression threshold for /risk-map

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    submitted_on         TEXT    NOT NULL,   -- date only (YYYY-MM-DD), never a timestamp
    state                TEXT    NOT NULL,
    state_key            TEXT    NOT NULL,
    district             TEXT    NOT NULL,
    district_key         TEXT    NOT NULL,
    exposure_risk_level  TEXT    NOT NULL,
    exposure_risk_score  INTEGER NOT NULL,
    routes_triggered     TEXT    NOT NULL,   -- comma-separated route keys
    image_risk_level     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_district ON submissions (state_key, district_key);
"""


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


class SubmissionStore:
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per operation: safe with FastAPI's threadpool.
        return sqlite3.connect(self.db_path, timeout=10)

    def save_submission(
        self,
        *,
        state: Optional[str],
        district: Optional[str],
        exposure_risk_level: str,
        exposure_risk_score: int,
        routes_triggered: Iterable[str],
        image_risk_level: str,
    ) -> None:
        """Store one CONSENTED, anonymized submission. Callers must check consent first."""
        state_clean = (state or "").strip() or "unknown"
        district_clean = (district or "").strip() or "unknown"
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO submissions (submitted_on, state, state_key, district, district_key, "
                "exposure_risk_level, exposure_risk_score, routes_triggered, image_risk_level) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).date().isoformat(),
                    state_clean,
                    _key(state_clean),
                    district_clean,
                    _key(district_clean),
                    exposure_risk_level,
                    int(exposure_risk_score),
                    ",".join(routes_triggered),
                    image_risk_level,
                ),
            )

    def district_stats(self, min_cell_size: int = MIN_CELL_SIZE) -> dict:
        """Aggregated, anonymized district-level stats with small-cell suppression."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT MIN(state), MIN(district), COUNT(*), AVG(exposure_risk_score), "
                "SUM(exposure_risk_level = 'low'), SUM(exposure_risk_level = 'moderate'), "
                "SUM(exposure_risk_level = 'high') "
                "FROM submissions GROUP BY state_key, district_key ORDER BY COUNT(*) DESC, MIN(district)"
            ).fetchall()

        districts, suppressed_districts, suppressed_submissions, total = [], 0, 0, 0
        for state, district, n, mean_score, n_low, n_mod, n_high in rows:
            total += n
            if n < min_cell_size:
                suppressed_districts += 1
                suppressed_submissions += n
                continue
            districts.append(
                {
                    "state": state,
                    "district": district,
                    "n": n,
                    "mean_exposure_risk_score": round(float(mean_score), 1),
                    "exposure_risk_counts": {"low": n_low, "moderate": n_mod, "high": n_high},
                }
            )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "min_cell_size": min_cell_size,
            "total_submissions": total,
            "districts": districts,
            "suppressed": {"districts": suppressed_districts, "submissions": suppressed_submissions},
            "notes": (
                "Anonymized, consent-based submissions only. Districts with fewer than "
                f"{min_cell_size} submissions are not listed individually."
            ),
        }
