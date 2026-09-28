"""
backend/app/main.py -- Nirikshan API.

Run from the repo root:  uvicorn backend.app.main:app --reload --port 8000

Endpoints
  GET  /health           liveness (also what the keep-alive pinger calls)
  POST /predict          multipart: image                       -> image-based risk
  POST /risk-fusion      multipart: image + exposure (JSON str) -> image_risk AND exposure_risk, SEPARATE
  GET  /fairness-report  accuracy by skin-tone bin (stub until Session 8 generates it)
  GET  /risk-map         anonymized, small-cell-suppressed district stats

Config (env vars, all optional):
  FRONTEND_URL          allowed CORS origin(s), comma-separated   [http://localhost:5173]
  MODEL_CHECKPOINT_DIR  dir with stage1_final.pt / stage2_final.pt [model/checkpoints]
  SQLITE_PATH           anonymized-submissions DB                  [backend/nirikshan_submissions.sqlite3]

Rules enforced here (PROJECT_CONTEXT s6-s8): image and exposure risk are never merged; displayed
recommendation follows the POST-escalation risk; images are processed in memory only.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .exposure_schema import ExposureHistory
from .fusion import RECOMMENDATIONS, apply_safety_net as fusion_apply_safety_net, fuse_risk
from .models_io import MODEL_STATUS, load_model_backend
from .risk_fusion import assess_exposure_risk, load_arsenic_districts
from .storage import SubmissionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [api] %(levelname)s %(message)s")
logger = logging.getLogger("api")

REPO_ROOT = Path(__file__).resolve().parents[2]
FAIRNESS_REPORT_PATH = Path(__file__).with_name("fairness_report.json")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 256 * 1024  # file + multipart/form overhead
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "MPO", "BMP"}  # MPO = some phone JPEGs
Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard

# Starlette spools multipart files larger than 1 MB to a TEMP FILE ON DISK by default. Keep
# everything up to our upload cap in memory so images never touch disk (D.1).
try:  # pragma: no cover - attribute exists on current Starlette; harmless otherwise
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + 1024 * 1024
except Exception:  # noqa: BLE001
    logger.warning("Could not raise Starlette's multipart spool size; large uploads may spool to disk.")

ITA_BINS = ("very_light", "light", "intermediate", "tan", "dark", "very_dark", "unknown")

CLASS_LABELS = {
    "squamous_cell_carcinoma": "squamous cell carcinoma",
    "actinic_keratosis": "actinic keratosis (a precancerous lesion)",
    "nevus": "a benign mole (nevus)",
    "seborrheic_keratosis": "seborrheic keratosis (a benign growth)",
}
NO_LESION_EXPLANATION = (
    "No lesion was detected in this photo. This is not a clearance: the photo check can miss "
    "lesions. If you have a spot that worries you, please have it examined in person."
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    frontend_urls: tuple
    checkpoint_dir: Optional[str]
    sqlite_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        raw = os.environ.get("FRONTEND_URL", "http://localhost:5173")
        urls = tuple(u.strip().rstrip("/") for u in raw.split(",") if u.strip())
        sqlite_path = os.environ.get("SQLITE_PATH", "backend/nirikshan_submissions.sqlite3")
        if sqlite_path != ":memory:" and not Path(sqlite_path).is_absolute():
            sqlite_path = str(REPO_ROOT / sqlite_path)
        return cls(
            frontend_urls=urls,
            checkpoint_dir=os.environ.get("MODEL_CHECKPOINT_DIR") or None,
            sqlite_path=sqlite_path,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _read_image(upload: UploadFile) -> Image.Image:
    """Decode an upload into an RGB PIL image, entirely in memory. 400/413/415 on bad input."""
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 10 MB limit.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_IMAGE_FORMATS:
            raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a JPEG, PNG or WebP photo.")
        img.load()  # decode fully so truncated files fail here, not mid-inference
        # Phones store rotation in EXIF; apply it so the lesion is upright (does not resample).
        return ImageOps.exif_transpose(img).convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(status_code=415, detail="The uploaded file is not a valid image.")


def _parse_exposure(raw: str) -> ExposureHistory:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="'exposure' must be a JSON object string.")
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="'exposure' must be a JSON object.")
    try:
        return ExposureHistory(**data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json()))


def _recommendation_text(level: str, lesion_detected: bool) -> str:
    text = RECOMMENDATIONS[level]  # always keyed by the POST-escalation level
    if level == "low" and not lesion_detected:
        text += (
            " Note: no lesion was detected in this photo, which is not a clearance. If a spot "
            "worries you, please have it examined in person."
        )
    return text


def _build_image_risk(raw: dict) -> dict:
    """
    Turn the backend result into the final image-side risk payload.

    model/safety_net.py (run inside the backend) is authoritative, and fusion.apply_safety_net
    is applied on top as an idempotent second pass. The two flags/explanations are merged,
    because fusion.py alone would report safety_escalated=False for an already-escalated input.
    """
    second = fusion_apply_safety_net(
        {"image_risk_level": raw["image_risk_level"], "p_scc": raw["p_scc"], "p_ak": raw["p_ak"]}
    )
    escalated = bool(raw["safety_escalated"] or second["safety_escalated"])
    safety_explanation = raw["safety_explanation"] or second["safety_explanation"]
    level = second["image_risk_level"]  # >= backend level; never "none"

    if escalated:
        explanation = safety_explanation
    elif not raw["lesion_detected"]:
        explanation = NO_LESION_EXPLANATION
    else:
        top1 = raw["top1_class"]
        explanation = (
            f"The image model's most likely match is {CLASS_LABELS.get(top1, top1)} "
            f"({raw['probabilities'][top1]:.0%}). This is a screening result, not a diagnosis."
        )
    return {
        "image_risk_level": level,
        "raw_image_risk_level": raw["base_risk"],  # before the safety net
        "p_scc": raw["p_scc"],
        "p_ak": raw["p_ak"],
        "safety_escalated": escalated,
        "safety_explanation": safety_explanation,
        "explanation": explanation,
        "top1_class": raw["top1_class"],
        "probabilities": raw["probabilities"],
    }


def _shared_fields(raw: dict, image_risk: dict) -> dict:
    return {
        "safety_escalated": image_risk["safety_escalated"],
        "lesion_detected": raw["lesion_detected"],  # Stage 1 verdict, separate from risk
        "stage1_lesion_prob": raw["stage1_lesion_prob"],
        "model_status": raw["model_status"],
        "gradcam_png_base64": raw["gradcam_png_base64"],  # null until Session 8
        "model_versions": raw["model_versions"],
    }


def _stub_fairness_report() -> dict:
    return {
        "generated_at": None,
        "bins": [
            {"ita_bin": b, "n": 0, "accuracy": None, "macro_f1": None, "low_confidence": True}
            for b in ITA_BINS
        ],
        "notes": "Placeholder: no fairness evaluation has been generated yet.",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "backend", None) is not None,
        "model_status": MODEL_STATUS,
    }


@router.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)) -> dict:
    pil_image = await _read_image(image)
    raw = await run_in_threadpool(request.app.state.backend.predict, pil_image)
    image_risk = _build_image_risk(raw)
    return {
        "image_risk": image_risk,
        "recommendation": {
            "level": image_risk["image_risk_level"],
            "text": _recommendation_text(image_risk["image_risk_level"], raw["lesion_detected"]),
        },
        **_shared_fields(raw, image_risk),
    }


@router.post("/risk-fusion")
async def risk_fusion(
    request: Request,
    image: UploadFile = File(...),
    exposure: str = Form(..., description="ExposureHistory as a JSON object string"),
    consent_store_anonymized: bool = Form(False),
) -> dict:
    state = request.app.state
    history = _parse_exposure(exposure)  # cheap; validate before running the models
    pil_image = await _read_image(image)

    raw = await run_in_threadpool(state.backend.predict, pil_image)
    image_risk = _build_image_risk(raw)
    exposure_risk = assess_exposure_risk(history, state.arsenic_districts)

    fused = fuse_risk(
        {"image_risk_level": image_risk["image_risk_level"], "p_scc": image_risk["p_scc"], "p_ak": image_risk["p_ak"]},
        exposure_risk,
    )
    fused["image_risk"] = image_risk  # ours carries the merged safety flags + explanation
    fused["safety_escalated"] = image_risk["safety_escalated"]
    fused["recommendation"]["text"] = _recommendation_text(
        fused["recommendation"]["level"], raw["lesion_detected"]
    )

    stored = False
    if consent_store_anonymized:
        try:
            state.store.save_submission(
                state=history.state,
                district=history.district,
                exposure_risk_level=exposure_risk["exposure_risk_level"],
                exposure_risk_score=exposure_risk["exposure_risk_score"],
                routes_triggered=exposure_risk.get("routes_triggered", []),
                image_risk_level=image_risk["image_risk_level"],
            )
            stored = True
        except Exception:  # noqa: BLE001 - never fail a screening because analytics storage failed
            logger.exception("Could not store anonymized submission")

    return {**fused, **_shared_fields(raw, image_risk), "submission_stored": stored}


@router.get("/fairness-report")
def fairness_report() -> Any:
    try:
        return json.loads(FAIRNESS_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("fairness_report.json missing or invalid; serving empty stub")
        return _stub_fairness_report()


@router.get("/risk-map")
def risk_map(request: Request) -> dict:
    return request.app.state.store.district_stats()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    t0 = time.perf_counter()
    app.state.backend = load_model_backend(settings.checkpoint_dir)  # ONCE, not per request
    app.state.store = SubmissionStore(settings.sqlite_path)
    app.state.arsenic_districts = load_arsenic_districts()
    logger.info("Startup complete in %.2fs: %s", time.perf_counter() - t0, app.state.backend.info())
    yield


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Nirikshan API", version="0.1.0", lifespan=_lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if request.method == "POST" and length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        return await call_next(request)

    # Added last so it is outermost: even 413/4xx responses carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_urls),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
