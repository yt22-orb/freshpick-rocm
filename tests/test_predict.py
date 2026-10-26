import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from freshpick import api
from freshpick.predict import _pad, response


def test_response_uses_joint_winner_and_sums_all_produce():
    names = [f"{stage}_{produce}" for stage in ("fresh", "semi_fresh", "rotten")
             for produce in ("banana", "bitter melon", "cucumber", "eggplant", "orange", "papaya", "pineapple", "tomato")]
    probabilities = [0.01] * 24
    probabilities[8] = 0.5
    probabilities[0] = 0.28
    result = response(names, probabilities, "test")
    assert result["produce"] == "banana"
    assert result["freshness"] == "semi_fresh"
    assert result["class_confidence"] == 0.5
    assert result["freshness_confidence"] == pytest.approx(0.57)
    assert sum(result["freshness_probabilities"].values()) == pytest.approx(1)
    assert len(result["class_probabilities"]) == 24


def test_padding_preserves_aspect_and_exif_orientation():
    image = Image.new("RGB", (40, 20), "red")
    image.getexif()[274] = 6
    padded = _pad(image)
    assert padded.size == (224, 224)
    assert padded.getpixel((0, 112)) == (0, 0, 0)
    assert padded.getpixel((112, 0)) == (255, 0, 0)


def test_api_errors_and_real_adapter(monkeypatch):
    client = TestClient(api.app)
    assert client.post("/predict", files={"file": ("bad.jpg", b"invalid")}).status_code == 400
    assert client.post("/predict").status_code == 422
    data = io.BytesIO()
    Image.new("RGB", (10, 10)).save(data, format="PNG")
    monkeypatch.setattr(api, "_predictor", None)
    monkeypatch.setenv("FRESHPICK_CHECKPOINT", "/nonexistent/checkpoint.pt")
    assert client.post("/predict", files={"file": ("image.png", data.getvalue())}).status_code == 503


def test_api_unsupported_oversized_and_inference_failure(monkeypatch):
    client = TestClient(api.app)
    data = io.BytesIO()
    Image.new("RGB", (10, 10)).save(data, format="GIF")
    assert client.post("/predict", files={"file": ("image.gif", data.getvalue())}).status_code == 415
    monkeypatch.setattr(api, "MAX_BYTES", 2)
    assert client.post("/predict", files={"file": ("large.png", b"123")}).status_code == 413
    monkeypatch.setattr(api, "MAX_BYTES", 1024)
    class BrokenPredictor:
        def predict(self, image):
            raise RuntimeError("internal failure")
    monkeypatch.setattr(api, "_predictor", BrokenPredictor())
    data = io.BytesIO()
    Image.new("RGB", (10, 10)).save(data, format="PNG")
    result = client.post("/predict", files={"file": ("image.png", data.getvalue())})
    assert result.status_code == 503
    assert "internal failure" not in result.text
