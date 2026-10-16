import pytest
from fastapi.testclient import TestClient

# Import the FastAPI application. Adjust the import path if the app is located elsewhere.
from app.main import app  # noqa: F401

# Import the River drift detector used in the application.
# The detector is expected to be a module‑level instance (e.g., ADWIN) that can be reset for testing.
from app.drift import drift_detector  # noqa: F401

client = TestClient(app)


def reset_drift_detector():
    """
    Reset the global drift detector to a clean state.
    This function assumes the detector follows the River API and provides a `reset` method.
    If the detector does not have a `reset` method, we re‑instantiate it.
    """
    if hasattr(drift_detector, "reset"):
        drift_detector.reset()
    else:
        # Re‑create a fresh detector instance; this requires the original class to be importable.
        from river.drift import ADWIN

        # Replace the global instance with a new one.
        globals()["drift_detector"] = ADWIN()


@pytest.fixture(autouse=True)
def clean_environment():
    """
    Ensure each test runs with a fresh drift detector.
    """
    reset_drift_detector()
    yield
    reset_drift_detector()


def test_status_endpoint():
    """
    Verify that the health‑check endpoint returns a 200 status and the expected payload.
    """
    response = client.get("/status")
    assert response.status_code == 200, "Status endpoint should return HTTP 200"
    json_body = response.json()
    assert isinstance(json_body, dict), "Response should be a JSON object"
    assert json_body.get("status") == "ok", "Status payload should contain {'status': 'ok'}"


def test_predict_success():
    """
    Send a well‑formed request to the /predict endpoint and assert that the response
    contains the expected keys and types.
    """
    payload = {"text": "I am feeling great today!", "lang": "en"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, "Valid request should return HTTP 200"
    data = response.json()
    # Expected response schema
    assert isinstance(data, dict), "Response must be a JSON object"
    assert "emotion" in data, "Response must contain an 'emotion' field"
    assert "confidence" in data, "Response must contain a 'confidence' field"
    assert isinstance(data["emotion"], str), "'emotion' should be a string"
    assert isinstance(data["confidence"], float), "'confidence' should be a float"
    # Confidence should be in the [0, 1] range
    assert 0.0 <= data["confidence"] <= 1.0, "'confidence' must be between 0 and 1"


def test_predict_missing_text():
    """
    Ensure the API validates the request payload and returns a 422 error when required
    fields are missing.
    """
    payload = {"lang": "en"}  # 'text' is omitted
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Missing required fields should trigger HTTP 422"


def test_drift_detection_no_change():
    """
    Feed a stable data stream to the drift detector and verify that no drift is reported.
    """
    # Simulate 200 observations from a stable distribution (e.g., constant value 0.5)
    for _ in range(200):
        changed = drift_detector.update(0.5)
        # River's ADWIN returns True when a change is detected; we expect False throughout.
        assert not changed, "Stable data should not trigger drift"

    # Explicitly check the detector's internal flag if available
    if hasattr(drift_detector, "detect_change"):
        assert not drift_detector.detect_change(), "Detector should report no change after stable data"


def test_drift_detection_with_change():
    """
    Feed a data stream that changes its underlying distribution and assert that drift is detected.
    """
    # First, feed a stable segment.
    for _ in range(100):
        drift_detector.update(0.2)

    # No drift should be reported yet.
    if hasattr(drift_detector, "detect_change"):
        assert not drift_detector.detect_change()

    # Now introduce a shift in the data distribution.
    drift_occurred = False
    for i in range(50):
        changed = drift_detector.update(0.9)
        if changed:
            drift_occurred = True
            break

    # At least one update after the shift should signal drift.
    assert drift_occurred, "Drift should be detected after a distribution shift"

    # Verify the detector's explicit change flag if present.
    if hasattr(drift_detector, "detect_change"):
        assert drift_detector.detect_change(), "Detector's detect_change() should return True after drift"


def test_predict_and_drift_integration():
    """
    End‑to‑end test that sends multiple predictions and ensures the drift detector
    processes the underlying confidence scores.
    """
    # Send a batch of predictions with similar confidence values.
    for _ in range(30):
        payload = {"text": "I love this!", "lang": "en"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        confidence = response.json()["confidence"]
        drift_detector.update(confidence)

    # No drift expected yet.
    if hasattr(drift_detector, "detect_change"):
        assert not drift_detector.detect_change()

    # Send predictions with markedly different confidence scores to provoke drift.
    drift_triggered = False
    for _ in range(10):
        payload = {"text": "I hate this!", "lang": "en"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        confidence = response.json()["confidence"]
        if drift_detector.update(confidence):
            drift_triggered = True
            break

    assert drift_triggered, "Drift should be triggered by a sudden change in confidence scores"
    if hasattr(drift_detector, "detect_change"):
        assert drift_detector.detect_change(), "Detector should report drift after confidence shift"