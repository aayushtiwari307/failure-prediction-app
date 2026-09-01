"""
Failure Prediction System V2 - ML Service (Milestone 2)

Single source of truth for: model inference AND risk-level classification.
Node/Express and React must never recompute risk thresholds themselves -
they only forward/render what this service returns.
"""
import logging
import math
from datetime import datetime, timezone

import joblib
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

app = Flask(__name__)

MODEL_PATH = "artifact/model.pkl"
METADATA_PATH = "artifact/metadata.json"

# Populated at startup by load_artifact(). READY stays False until every
# check passes, so /health/ready fails fast instead of the first request
# discovering a broken model.
STATE = {"model": None, "metadata": None, "ready": False, "error": None}

EXPECTED_FIELDS = {
    "air_temperature_k": float,
    "process_temperature_k": float,
    "rotational_speed_rpm": float,
    "torque_nm": float,
    "tool_wear_min": float,
    "type": str,
}
ALLOWED_TYPES = {"L", "M", "H"}
# Bounds derived directly from the AI4I 2020 training data's observed
# min/max, padded 5% and physically clipped at 0 where negative values are
# meaningless (torque, tool wear). These represent the range the model was
# actually trained on - NOT physical/industry engineering limits. Values
# outside this range are rejected because the model has no evidence about
# how it behaves there, not because they are physically impossible.
RANGE_BOUNDS = {
    "air_temperature_k": (294.8, 305.0),
    "process_temperature_k": (305.3, 314.2),
    "rotational_speed_rpm": (1082.0, 2972.0),
    "torque_nm": (0.0, 80.2),
    "tool_wear_min": (0.0, 265.7),
}


def load_artifact():
    """Load model + metadata once at startup. Sets STATE['ready'] only if
    every check passes: files exist, model loads, metadata matches the
    model's expected feature count, and a real inference call succeeds."""
    import json
    import os

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

        with open(METADATA_PATH) as f:
            metadata = json.load(f)

        model = joblib.load(MODEL_PATH)

        expected_feature_count = len(metadata["featureSchema"])
        if expected_feature_count != len(EXPECTED_FIELDS):
            raise ValueError(
                f"Metadata declares {expected_feature_count} features but "
                f"service contract expects {len(EXPECTED_FIELDS)}"
            )

        # Real test inference - not a shape check, an actual predict_proba call.
        test_row = {
            "air_temperature_k": 300.0, "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500.0, "torque_nm": 40.0,
            "tool_wear_min": 10.0, "type": "M",
        }
        _ = predict_one(model, test_row)

        STATE.update(model=model, metadata=metadata, ready=True, error=None)
        logger.info("Model loaded and ready. version=%s", metadata.get("modelVersion"))
    except Exception as e:  # noqa: BLE001 - startup validation, log real cause
        STATE.update(model=None, metadata=None, ready=False, error=str(e))
        logger.error("Model failed to load: %s", e)


def predict_one(model, row: dict) -> float:
    """row uses EXPECTED_FIELDS names; converts to the column names/order
    the sklearn Pipeline was fit on."""
    import pandas as pd
    df_row = pd.DataFrame([{
        "Air temperature [K]": row["air_temperature_k"],
        "Process temperature [K]": row["process_temperature_k"],
        "Rotational speed [rpm]": row["rotational_speed_rpm"],
        "Torque [Nm]": row["torque_nm"],
        "Tool wear [min]": row["tool_wear_min"],
        "Type": row["type"],
    }])
    return float(model.predict_proba(df_row)[0][1])


def classify_risk(score: float, metadata: dict) -> str:
    """SINGLE SOURCE OF TRUTH for risk level.

    Milestone 1's threshold-sensitivity experiment established ONE selected
    operating point (threshold=0.03, under the project's stated 5:1 FN:FP
    cost assumption) - not three risk bands. The 0.57 value in the
    sensitivity table came from a DIFFERENT, mutually exclusive cost
    assumption (1:1/2:1); it is a different operating point under a
    different assumption, not automatically the MEDIUM/HIGH boundary of a
    single combined policy. No experiment has validated that pairing, so
    it is not used here. If a three-level policy is wanted later, it needs
    its own validation evidence, not two unrelated numbers stitched together.
    """
    policy = metadata["thresholdPolicy"]["selectedOperatingPoint"]
    if score >= policy["threshold"]:
        return "ACTION_REQUIRED"
    return "NORMAL"


def error_response(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def validate_input(payload: dict):
    """Returns (cleaned_dict, None) on success or (None, error_message)."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object"

    missing = [f for f in EXPECTED_FIELDS if f not in payload]
    if missing:
        return None, f"Missing required field(s): {', '.join(missing)}"

    unexpected = [f for f in payload if f not in EXPECTED_FIELDS]
    if unexpected:
        return None, f"Unexpected field(s): {', '.join(unexpected)}"

    cleaned = {}
    for field, expected_type in EXPECTED_FIELDS.items():
        value = payload[field]
        if field == "type":
            if not isinstance(value, str) or value not in ALLOWED_TYPES:
                return None, f"'type' must be one of {sorted(ALLOWED_TYPES)}"
            cleaned[field] = value
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"'{field}' must be numeric"
        value = float(value)
        if not math.isfinite(value):
            return None, f"'{field}' must be a finite number"
        lo, hi = RANGE_BOUNDS[field]
        if not (lo <= value <= hi):
            return None, f"'{field}' must be between {lo} and {hi}, got {value}"
        cleaned[field] = value

    return cleaned, None


@app.route("/health/live", methods=["GET"])
def health_live():
    return jsonify({"status": "alive"}), 200


@app.route("/health/ready", methods=["GET"])
def health_ready():
    if STATE["ready"]:
        return jsonify({
            "status": "ready",
            "modelVersion": STATE["metadata"]["modelVersion"],
        }), 200
    return jsonify({
        "status": "not_ready",
        "error": STATE["error"] or "model not loaded",
    }), 503


@app.route("/model-info", methods=["GET"])
def model_info():
    if not STATE["ready"]:
        return error_response("MODEL_NOT_READY", "Model is not currently loaded", 503)
    md = STATE["metadata"]
    # Deliberately expose only safe, useful fields - no file paths, no env details.
    return jsonify({
        "modelVersion": md["modelVersion"],
        "modelType": md["modelType"],
        "featureSchema": md["featureSchema"],
        "predictionSemantics": md["predictionSemantics"],
        "calibrated": False,
        "outputMeaning": "risk score (uncalibrated) - not a guaranteed real-world probability",
        "thresholdPolicy": md["thresholdPolicy"]["selectedOperatingPoint"],
        "alternativeOperatingPoints": md["thresholdPolicy"]["sensitivityTable"],
        "finalTestMetrics": {
            k: v for k, v in md["finalTestMetrics"].items()
            if k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "recall95CI", "precision95CI")
        },
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not STATE["ready"]:
        return error_response("MODEL_NOT_READY", "Prediction service temporarily unavailable", 503)

    payload = request.get_json(silent=True)
    if payload is None:
        return error_response("INVALID_JSON", "Request body must be valid JSON", 400)

    cleaned, err = validate_input(payload)
    if err:
        return error_response("VALIDATION_FAILED", err, 422)

    try:
        score = predict_one(STATE["model"], cleaned)
    except Exception as e:  # noqa: BLE001
        logger.error("Inference failed: %s", e)
        return error_response("INFERENCE_FAILED", "Prediction service temporarily unavailable", 500)

    risk_level = classify_risk(score, STATE["metadata"])

    return jsonify({
        "modelVersion": STATE["metadata"]["modelVersion"],
        "riskScore": round(score, 4),
        "riskLevel": risk_level,
        "thresholdPolicy": STATE["metadata"]["thresholdPolicy"]["selectedOperatingPoint"],
        "predictedAt": datetime.now(timezone.utc).isoformat(),
    }), 200


load_artifact()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
