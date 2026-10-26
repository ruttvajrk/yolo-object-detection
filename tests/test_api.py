"""API tests. Need models/best.onnx (made by scripts/export_benchmark.py)."""
from pathlib import Path
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytestmark = pytest.mark.skipif(not Path("models/best.onnx").exists(), reason="export the ONNX model first")
SAMPLE = Path("tests/sample_zebra.jpg")


@pytest.fixture(scope="module")
def client():
    import api
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["classes"] == ["buffalo", "elephant", "rhino", "zebra"]


def test_detect_counts_animals(client):
    r = client.post("/detect", files={"file": ("z.jpg", SAMPLE.read_bytes(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200
    assert body["counts"]["zebra"] >= 1
    assert body["total"] == sum(body["counts"].values()) == len(body["detections"])


def test_rejects_non_image(client):
    r = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_blank_image_has_no_animals(client):
    buf = io.BytesIO(); Image.new("RGB", (640, 480), "white").save(buf, "JPEG")
    r = client.post("/detect", files={"file": ("w.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["total"] == 0
