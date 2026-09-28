"""
tests/test_api.py -- API tests that do NOT need the real (gitignored) checkpoints.

A session-scoped fixture builds tiny randomly-initialised checkpoints in the same on-disk
format as the real ones ({"model_state_dict", "metadata": {class_names, img_size, ...}}) and
points the app at them through MODEL_CHECKPOINT_DIR. One extra test runs against the real
checkpoints and is skipped when they are absent.

Run from the repo root:  python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from backend.app.exposure_schema import ExposureHistory, FirePotUse, SunProtection, TobaccoUse, WaterSource  # noqa: E402
from backend.app.main import MAX_UPLOAD_BYTES, create_app  # noqa: E402
from backend.app.models_io import MODEL_STATUS, ModelBackend, RawInference  # noqa: E402

STAGE1_CLASSES = ["no_lesion", "lesion_present"]
STAGE2_CLASSES = ["squamous_cell_carcinoma", "actinic_keratosis", "nevus", "seborrheic_keratosis"]
FRONTEND_ORIGIN = "https://nirikshan-test.vercel.app"
REAL_CKPT_DIR = REPO_ROOT / "model" / "checkpoints"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _metadata(class_names: list, stage: str) -> dict:
    return {
        "class_names": class_names,
        "img_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "backbone": "efficientnet_b0",
        "stage": stage,
        "epoch": 1,
    }


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ckpts")
    torch.manual_seed(0)
    stage1 = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)  # 1-logit BCE head
    stage2 = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(STAGE2_CLASSES))
    torch.save({"model_state_dict": stage1.state_dict(), "metadata": _metadata(STAGE1_CLASSES, "stage1_lesion_presence")}, out / "stage1_final.pt")
    torch.save({"model_state_dict": stage2.state_dict(), "metadata": _metadata(STAGE2_CLASSES, "stage2_disease_classifier")}, out / "stage2_final.pt")
    return out


@pytest.fixture(scope="module")
def client(checkpoint_dir, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_CHECKPOINT_DIR", str(checkpoint_dir))
    mp.setenv("SQLITE_PATH", str(tmp_path_factory.mktemp("db") / "test.sqlite3"))
    mp.setenv("FRONTEND_URL", FRONTEND_ORIGIN)
    with TestClient(create_app()) as tc:  # context manager => lifespan (model load) runs
        yield tc
    mp.undo()


def image_bytes(fmt: str = "JPEG", size=(256, 256)) -> bytes:
    import random

    rng = random.Random(1)
    img = Image.new("RGB", size)
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def exposure_payload(_validate: bool = True, **overrides) -> dict:
    no_tobacco = next(m for m in TobaccoUse if m not in {TobaccoUse.SMOKING, TobaccoUse.SMOKELESS, TobaccoUse.BOTH})
    payload = {
        "fire_pot_use": FirePotUse.NEVER.value,
        "fire_pot_years": 0,
        "has_burn_scar": False,
        "burn_scar_age_years": None,
        "burn_scar_nonhealing_ulcer": False,
        "district": "TestDistrict",
        "state": "Testland",
        "water_source": WaterSource.DEEP_BOREWELL.value,
        "years_at_water_source": 20,
        "outdoor_hours_per_day": 0,
        "outdoor_work_years": 0,
        "sun_protection": SunProtection.NEVER.value,
        "age_years": 35,
        "family_history_skin_cancer": False,
        "tobacco_use": no_tobacco.value,
    }
    payload.update(overrides)
    try:
        if _validate:
            ExposureHistory(**payload)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"exposure_payload() no longer matches backend/app/exposure_schema.py: {exc}")
    return payload


def post_fusion(client, payload=None, consent=False, img=None):
    return client.post(
        "/risk-fusion",
        files={"image": ("lesion.jpg", img or image_bytes(), "image/jpeg")},
        data={"exposure": json.dumps(payload or exposure_payload()), "consent_store_anonymized": str(consent).lower()},
    )


class FakeBackend(ModelBackend):
    """Deterministic backend: proves the abstraction works and lets us pin Stage-1/Stage-2 combos."""

    def __init__(self, p_lesion: float, stage2_probs: list):
        super().__init__(STAGE2_CLASSES, {"stage1_epoch": 0, "stage2_epoch": 0})
        self._raw = RawInference(p_lesion=p_lesion, stage2_probs=stage2_probs)

    def infer_raw(self, image):
        return self._raw


@pytest.fixture()
def use_backend(client):
    original = client.app.state.backend

    def _set(p_lesion, probs):
        client.app.state.backend = FakeBackend(p_lesion, probs)

    yield _set
    client.app.state.backend = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_predict_shape(client):
    r = client.post("/predict", files={"image": ("a.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    risk = body["image_risk"]
    assert risk["image_risk_level"] in {"low", "moderate", "high"}  # never "none"
    assert list(risk["probabilities"]) == STAGE2_CLASSES  # order comes from checkpoint metadata
    assert sum(risk["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert isinstance(body["lesion_detected"], bool)
    assert 0.0 <= body["stage1_lesion_prob"] <= 1.0
    assert body["model_status"] == MODEL_STATUS == "stage1_placeholder"
    assert body["gradcam_png_base64"] is None
    assert isinstance(body["safety_escalated"], bool)
    assert "exposure_risk" not in body  # /predict is image-side only


def test_predict_rejects_bad_uploads(client):
    assert client.post("/predict", files={"image": ("a.txt", b"not an image", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"image": ("a.jpg", b"", "image/jpeg")}).status_code == 400
    too_big = b"\xff" * (MAX_UPLOAD_BYTES + 1)
    assert client.post("/predict", files={"image": ("big.jpg", too_big, "image/jpeg")}).status_code == 413
    assert client.post("/predict").status_code == 422  # missing file


def test_risk_fusion_keeps_components_separate(client):
    r = post_fusion(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"image_risk", "exposure_risk", "recommendation"} <= set(body)
    assert "image_risk_level" in body["image_risk"] and "exposure_risk_level" not in body["image_risk"]
    assert {"exposure_risk_score", "exposure_risk_level", "explanation"} <= set(body["exposure_risk"])
    assert "image_risk_level" not in body["exposure_risk"]
    assert not {"risk_score", "combined_score", "final_score"} & set(body)  # never pre-merged
    assert body["recommendation"]["basis"] in {"image", "exposure", "both"}
    assert body["gradcam_png_base64"] is None
    assert body["submission_stored"] is False  # no consent given


def test_risk_fusion_validation_errors(client):
    files = {"image": ("a.jpg", image_bytes(), "image/jpeg")}
    assert client.post("/risk-fusion", files=files, data={"exposure": "{not json"}).status_code == 422
    assert client.post("/risk-fusion", files=files, data={"exposure": "[]"}).status_code == 422
    assert client.post("/risk-fusion", files=files, data={"exposure": json.dumps(exposure_payload(_validate=False, age_years="not-a-number"))}).status_code == 422
    assert client.post("/risk-fusion", data={"exposure": json.dumps(exposure_payload())}).status_code == 422


def test_safety_net_fires_even_when_stage1_says_no_lesion(client, use_backend):
    # Stage 1: 1% lesion. Stage 2: nevus is top-1, but p_scc = 0.30 (>= 15%).
    use_backend(0.01, [0.30, 0.02, 0.50, 0.18])
    body = post_fusion(client).json()
    assert body["lesion_detected"] is False
    assert body["image_risk"]["image_risk_level"] == "high"
    assert body["image_risk"]["raw_image_risk_level"] == "low"
    assert body["image_risk"]["safety_escalated"] is True and body["safety_escalated"] is True
    assert body["image_risk"]["safety_explanation"]
    assert body["recommendation"]["level"] == "high"  # follows post-escalation, not top-1 (nevus)


def test_no_lesion_no_escalation_is_low_and_not_a_clearance(client, use_backend):
    use_backend(0.01, [0.02, 0.03, 0.60, 0.35])
    body = post_fusion(client).json()
    assert body["lesion_detected"] is False
    assert body["image_risk"]["image_risk_level"] == "low"
    assert body["image_risk"]["safety_escalated"] is False
    assert "not a clearance" in body["image_risk"]["explanation"].lower()
    if body["recommendation"]["level"] == "low":
        assert "not a clearance" in body["recommendation"]["text"].lower()


def test_stage2_probabilities_looked_up_by_name(client, use_backend):
    use_backend(0.99, [0.70, 0.10, 0.10, 0.10])
    risk = client.post("/predict", files={"image": ("a.jpg", image_bytes(), "image/jpeg")}).json()["image_risk"]
    assert risk["p_scc"] == pytest.approx(0.70) and risk["p_ak"] == pytest.approx(0.10)
    assert risk["image_risk_level"] == "high"


def test_consented_submissions_feed_risk_map_with_small_cell_suppression(client):
    payload = exposure_payload(district="RiskMapTown", state="Testland")
    for i in range(4):
        assert post_fusion(client, payload, consent=True).json()["submission_stored"] is True
    early = client.get("/risk-map").json()
    assert all(d["district"] != "RiskMapTown" for d in early["districts"])  # n=4 < 5: suppressed
    assert early["suppressed"]["districts"] >= 1

    post_fusion(client, payload, consent=True)
    later = client.get("/risk-map").json()
    town = next(d for d in later["districts"] if d["district"] == "RiskMapTown")
    assert town["n"] == 5
    assert set(town) == {"state", "district", "n", "mean_exposure_risk_score", "exposure_risk_counts"}


def test_fairness_report_stub_shape(client):
    body = client.get("/fairness-report").json()
    assert set(body) >= {"generated_at", "bins", "notes"}
    assert {b["ita_bin"] for b in body["bins"]} >= {"very_light", "light", "intermediate", "tan", "dark", "very_dark", "unknown"}
    for b in body["bins"]:
        assert set(b) == {"ita_bin", "n", "accuracy", "macro_f1", "low_confidence"}
        assert b["low_confidence"] is (b["n"] < 30)
        if b["n"] == 0:
            assert b["accuracy"] is None and b["macro_f1"] is None


def test_cors_allows_frontend_origin(client):
    r = client.options(
        "/predict",
        headers={"Origin": FRONTEND_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == FRONTEND_ORIGIN
    blocked = client.options("/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in blocked.headers


@pytest.mark.skipif(
    not ((REAL_CKPT_DIR / "stage1_final.pt").exists() and (REAL_CKPT_DIR / "stage2_final.pt").exists()),
    reason="real checkpoints (gitignored) not present",
)
def test_real_checkpoints_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT_DIR", str(REAL_CKPT_DIR))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "real.sqlite3"))
    with TestClient(create_app()) as tc:
        info = tc.app.state.backend.info()
        assert info["stage2_class_names"] == STAGE2_CLASSES
        r = tc.post("/predict", files={"image": ("a.jpg", image_bytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_status"] == "stage1_placeholder"
        assert list(body["image_risk"]["probabilities"]) == STAGE2_CLASSES
