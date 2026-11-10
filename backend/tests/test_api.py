"""Tests des endpoints FastAPI avec un faux detecteur (pas besoin du vrai modele)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class FakeDetector:
    def predict(self, image):
        height, width = image.shape[:2]
        det = {
            "class_id": 8,
            "label": "palm",
            "confidence": 0.93,
            "box": [10.0, 20.0, width / 2, height / 2],
        }
        return [det], 12.5


def png_bytes(width=64, height=48):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(create_app(detector=FakeDetector())) as c:
        yield c


@pytest.fixture
def client_sans_modele(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.onnx"))
    with TestClient(create_app()) as c:
        yield c


def test_health_indique_que_le_modele_est_charge(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_metadata_liste_les_19_classes(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    assert len(response.json()["names"]) == 19


def test_predict_renvoie_les_detections_et_la_latence(client):
    files = {"file": ("main.png", png_bytes(), "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["detections"][0]["label"] == "palm"
    assert body["inference_ms"] == 12.5
    assert (body["image_width"], body["image_height"]) == (64, 48)


def test_predict_refuse_un_fichier_qui_n_est_pas_une_image(client):
    files = {"file": ("notes.txt", b"bonjour", "text/plain")}
    assert client.post("/predict", files=files).status_code == 400


def test_predict_refuse_une_image_corrompue(client):
    files = {"file": ("casse.png", b"pas une image", "image/png")}
    assert client.post("/predict", files=files).status_code == 400


def test_sans_modele_l_api_repond_503(client_sans_modele):
    assert client_sans_modele.get("/health").json()["model_loaded"] is False
    files = {"file": ("main.png", png_bytes(), "image/png")}
    assert client_sans_modele.post("/predict", files=files).status_code == 503


def test_metrics_expose_les_compteurs_prometheus(client):
    client.post("/predict", files={"file": ("m.png", png_bytes(), "image/png")})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "touchfree_predict_requests_total" in response.text
