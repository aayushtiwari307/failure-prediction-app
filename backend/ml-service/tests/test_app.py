import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app as app_v2

@pytest.fixture
def client():
    app_v2.load_artifact()
    app_v2.app.config["TESTING"] = True
    with app_v2.app.test_client() as c:
        yield c


VALID_PAYLOAD = {
    "air_temperature_k": 300.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 10.0,
    "type": "M",
}


def test_health_live(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.get_json()["status"] == "alive"


def test_health_ready_when_model_loaded(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ready"
    assert "modelVersion" in body


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.get_json()
    assert body["calibrated"] is False
    assert "riskScore" not in body  # sanity: this isn't a prediction endpoint
    assert len(body["featureSchema"]) == 6
    assert "MODEL_PATH" not in str(body)  # no internal paths leaked


def test_valid_prediction(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.get_json()
    assert 0.0 <= body["riskScore"] <= 1.0
    assert body["riskLevel"] in ("NORMAL", "ACTION_REQUIRED")
    assert "modelVersion" in body
    assert body["thresholdPolicy"]["threshold"] == 0.03
    assert body["thresholdPolicy"]["costRatioAssumed"] == 5


def test_missing_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["torque_nm"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert "torque_nm" in body["error"]["message"]


def test_unexpected_field_rejected(client):
    payload = dict(VALID_PAYLOAD)
    payload["extra_field"] = 123
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "extra_field" in r.get_json()["error"]["message"]


def test_invalid_type_value(client):
    payload = dict(VALID_PAYLOAD)
    payload["type"] = "X"  # not in L/M/H
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_non_numeric_field(client):
    payload = dict(VALID_PAYLOAD)
    payload["torque_nm"] = "not a number"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_out_of_range_value_rejected(client):
    payload = dict(VALID_PAYLOAD)
    payload["rotational_speed_rpm"] = -500.0
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_infinite_value_rejected(client):
    payload = dict(VALID_PAYLOAD)
    payload["torque_nm"] = float("inf")
    r = client.post("/predict", json=payload)
    assert r.status_code in (400, 422)  # JSON can't natively encode inf but guard anyway


def test_malformed_json(client):
    r = client.post("/predict", data="not json", content_type="application/json")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "INVALID_JSON"


def test_model_not_ready_returns_503(client, monkeypatch):
    monkeypatch.setitem(app_v2.STATE, "ready", False)
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "MODEL_NOT_READY"


def test_no_stack_trace_leaked_on_inference_error(client, monkeypatch):
    def broken_predict(model, row):
        raise RuntimeError("some internal sklearn error with a file path /home/x/y.pkl")
    monkeypatch.setattr(app_v2, "predict_one", broken_predict)
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 500
    body = r.get_json()
    assert "file path" not in body["error"]["message"]
    assert body["error"]["code"] == "INFERENCE_FAILED"


def test_risk_level_boundaries_match_locked_metadata(client):
    md = app_v2.STATE["metadata"]
    op = md["thresholdPolicy"]["selectedOperatingPoint"]
    assert op["threshold"] == 0.03
    assert op["costRatioAssumed"] == 5
    # the 0.57 value must exist ONLY in the sensitivity table, never as an
    # active classification boundary
    assert "riskLevelBoundaries" not in md["thresholdPolicy"]


def test_classify_risk_is_binary_not_three_tier():
    md = app_v2.STATE["metadata"]
    assert app_v2.classify_risk(0.02, md) == "NORMAL"
    assert app_v2.classify_risk(0.029999, md) == "NORMAL"
    assert app_v2.classify_risk(0.03, md) == "ACTION_REQUIRED"
    assert app_v2.classify_risk(0.5, md) == "ACTION_REQUIRED"  # between the two sensitivity thresholds - still ACTION_REQUIRED, not a fabricated MEDIUM
    assert app_v2.classify_risk(0.99, md) == "ACTION_REQUIRED"


def test_0_57_threshold_never_used_for_classification():
    """The 1:1/2:1 threshold (0.57) must never appear as a classification
    boundary - only as reference data in the sensitivity table."""
    md = app_v2.STATE["metadata"]
    # a score just below 0.57 must still classify the same as a score just
    # above 0.03 - proving 0.57 plays no role in classify_risk
    assert app_v2.classify_risk(0.04, md) == app_v2.classify_risk(0.56, md) == "ACTION_REQUIRED"
    sensitivity = md["thresholdPolicy"]["sensitivityTable"]
    assert any(row["threshold"] == 0.57 for row in sensitivity)  # still documented
    assert md["thresholdPolicy"]["selectedOperatingPoint"]["threshold"] != 0.57  # but not active


def test_range_bounds_derived_from_training_data_not_guessed():
    # air temperature observed range in training data is ~295-305K;
    # a guessed "250-350K" bound (the old, wrong bound) would wrongly
    # accept values the model has no evidence about.
    lo, hi = app_v2.RANGE_BOUNDS["air_temperature_k"]
    assert 290 < lo < 300
    assert 300 < hi < 310
