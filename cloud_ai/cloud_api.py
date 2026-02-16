"""FastAPI advisory service for cloud AI maintenance analytics."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from cloud_ai.explanation import explain_fault
from cloud_ai.failure_model import FAILURE_FEATURES
from cloud_ai.recommendation import recommend_action
from cloud_ai.rul_model import RUL_FEATURES
from cloud_ai.schemas import CloudInput, CloudOutput

app = FastAPI(title="Cloud AI Advisory API", version="2.0.0")

RUL_MODEL_PATH = Path("rul_model.pkl")
FAILURE_MODEL_PATH = Path("failure_model.pkl")


class ModelRegistry:
    rul_model = None
    failure_model = None


@app.on_event("startup")
def load_models() -> None:
    if not RUL_MODEL_PATH.exists() or not FAILURE_MODEL_PATH.exists():
        raise RuntimeError(
            "Model files not found. Train models first: "
            "python -m cloud_ai.rul_model && python -m cloud_ai.failure_model"
        )

    ModelRegistry.rul_model = joblib.load(RUL_MODEL_PATH)
    ModelRegistry.failure_model = joblib.load(FAILURE_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "advisory-only",
        "authority": "cloud_has_no_actuation_control",
    }


@app.post("/analyze", response_model=CloudOutput)
def analyze(data: CloudInput) -> CloudOutput:
    if ModelRegistry.rul_model is None or ModelRegistry.failure_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    rul_feature_map = {
        "thermal_stress_index": data.thermal_stress_index,
        "brake_health_index": data.brake_rul_pct / 100.0,
        "mechanical_vibration_anomaly_score": data.mechanical_vibration_anomaly_score,
        "electrical_charging_efficiency_score": data.electrical_charging_efficiency_score,
        "vehicle_health_score": data.vehicle_health_score,
    }
    rul_vector = pd.DataFrame([rul_feature_map])[RUL_FEATURES]
    predicted_engine_rul_pct = float(ModelRegistry.rul_model.predict(rul_vector)[0])
    predicted_engine_rul_pct = max(0.0, min(100.0, predicted_engine_rul_pct))

    failure_vector = pd.DataFrame(
        [
            {
                "engine_rul_pct": predicted_engine_rul_pct,
                "brake_rul_pct": data.brake_rul_pct,
                "battery_rul_pct": data.battery_rul_pct,
                "thermal_stress_index": data.thermal_stress_index,
                "mechanical_vibration_anomaly_score": data.mechanical_vibration_anomaly_score,
            }
        ]
    )[FAILURE_FEATURES]
    failure_prob = float(ModelRegistry.failure_model.predict_proba(failure_vector)[0][1])

    fault_primary, contributors = explain_fault(data, failure_prob)
    recommendation = recommend_action(
        failure_prob=failure_prob,
        engine_rul_pct=predicted_engine_rul_pct,
        brake_rul_pct=data.brake_rul_pct,
        fault_primary=fault_primary,
    )

    return CloudOutput(
        vehicle_id=data.vehicle_id,
        timestamp_ms=data.timestamp_ms,
        engine_rul_pct=round(predicted_engine_rul_pct, 2),
        brake_rul_pct=round(data.brake_rul_pct, 2),
        battery_rul_pct=round(data.battery_rul_pct, 2),
        fault_failure_probability_7d=round(failure_prob, 2),
        fault_primary=fault_primary,
        fault_contributing_factors=contributors,
        recommendation=recommendation,
    )
