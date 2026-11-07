import io
import os
from unittest import mock

os.environ["INFERENCE_MODE"] = "remote"
os.environ["API_URL"] = "http://fake-api"
os.environ["DATABASE_URL"] = "sqlite://"

from PIL import Image  # noqa: E402
from app import create_app  # noqa: E402


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(buf, "PNG")
    return buf.getvalue()


def client():
    return create_app().test_client()


def test_health():
    assert client().get("/health").status_code == 200


def test_rejects_wrong_type():
    r = client().post("/predict", data={"image": (io.BytesIO(b"x"), "a.txt")})
    assert r.status_code == 415


def test_rejects_fake_image():
    r = client().post("/predict", data={"image": (io.BytesIO(b"not an image"), "a.jpg")})
    assert r.status_code == 422


def test_predict_and_history():
    c = client()
    fake = mock.MagicMock()
    fake.json.return_value = {"label": "Real", "confidence": 0.9, "model": "m"}
    with mock.patch("app.routes.requests.post", return_value=fake):
        r = c.post("/predict", data={"image": (io.BytesIO(png_bytes()), "a.png")})
    assert r.status_code == 200
    assert b"a.png" in c.get("/history").data