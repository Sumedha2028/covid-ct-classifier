
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


def create_test_image():
    image = Image.new("RGB", (512, 512), color="gray")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_home_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "CT Image Classifier API is running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_valid_image_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    create_test_image(),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["predicted_class"] in {"NiCT", "nCT", "pCT"}
    assert set(data["probabilities"].keys()) == {"NiCT", "nCT", "pCT"}
    assert "notice" in data


def test_unsupported_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"test", "text/plain")},
        )

    assert response.status_code == 415


def test_malformed_image():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
        )

    assert response.status_code == 400
