"""Full cloud AI validation script: data generation, training, and local advisory inference."""

import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_ai.data_generation import generate_synthetic_cloud_history
from cloud_ai.explanation import explain_fault
from cloud_ai.failure_model import FAILURE_FEATURES, train_failure_model
from cloud_ai.recommendation import recommend_action
from cloud_ai.rul_model import predict_future_rul, train_rul_model
from cloud_ai.schemas import CloudInput


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "cloud_health_history.csv")
        rul_model_path = os.path.join(tmpdir, "rul_model.pkl")
        failure_model_path = os.path.join(tmpdir, "failure_model.pkl")

        df = generate_synthetic_cloud_history(output_path=data_path, rows=600, seed=7)
        print(f"[1/4] Generated synthetic training data rows={len(df)} at {data_path}")

        train_rul_model(data_path=data_path, model_path=rul_model_path)
        train_failure_model(data_path=data_path, model_path=failure_model_path)
        print("[2/4] Trained both cloud models")

        rul_model = joblib.load(rul_model_path)
        failure_model = joblib.load(failure_model_path)

        payload = CloudInput(
            vehicle_id="VIT_CAR_001",
            timestamp_ms=1707051123456,
            thermal_brake_margin=-0.21,
            thermal_engine_margin=0.34,
            thermal_stress_index=0.82,
            mechanical_vibration_anomaly_score=0.77,
            mechanical_dominant_fault_band_hz=142,
            mechanical_vibration_rms=0.84,
            electrical_charging_efficiency_score=0.81,
            electrical_battery_health_pct=87,
            engine_rul_pct=62,
            brake_rul_pct=28,
            battery_rul_pct=74,
            vehicle_health_score=0.64,
        )

        future_rul = predict_future_rul(
            rul_model,
            engine_rul_pct=payload.engine_rul_pct,
            brake_rul_pct=payload.brake_rul_pct,
            battery_rul_pct=payload.battery_rul_pct,
        )
        predicted_engine_rul_pct = future_rul["engine"]
        predicted_brake_rul_pct = future_rul["brake"]
        predicted_battery_rul_pct = future_rul["battery"]

        failure_features = pd.DataFrame(
            [
                {
                    "engine_rul_pct": predicted_engine_rul_pct,
                    "brake_rul_pct": predicted_brake_rul_pct,
                    "battery_rul_pct": predicted_battery_rul_pct,
                    "thermal_stress_index": payload.thermal_stress_index,
                    "mechanical_vibration_anomaly_score": payload.mechanical_vibration_anomaly_score,
                }
            ]
        )[FAILURE_FEATURES]
        failure_prob = float(failure_model.predict_proba(failure_features)[0][1])

        fault_primary, contributors = explain_fault(payload, failure_prob)
        recommendation = recommend_action(
            failure_prob=failure_prob,
            engine_rul_pct=predicted_engine_rul_pct,
            brake_rul_pct=predicted_brake_rul_pct,
            fault_primary=fault_primary,
        )
        print("[3/4] Local inference pipeline completed")

        if not 0.0 <= predicted_engine_rul_pct <= 100.0:
            raise RuntimeError("Predicted engine RUL out of range")
        if not 0.0 <= predicted_brake_rul_pct <= 100.0:
            raise RuntimeError("Predicted brake RUL out of range")
        if not 0.0 <= predicted_battery_rul_pct <= 100.0:
            raise RuntimeError("Predicted battery RUL out of range")
        if not 0.0 <= failure_prob <= 1.0:
            raise RuntimeError("Failure probability out of range")
        if not contributors:
            raise RuntimeError("Contributors list must not be empty")
        if "recommendation_service_priority" not in recommendation:
            raise RuntimeError("Recommendation structure invalid")

        print(
            "[4/4] Advisory output validated:",
            {
                "engine_rul_pct_future": round(predicted_engine_rul_pct, 2),
                "brake_rul_pct_future": round(predicted_brake_rul_pct, 2),
                "battery_rul_pct_future": round(predicted_battery_rul_pct, 2),
                "fault_failure_probability_7d": round(failure_prob, 2),
                "fault_primary": fault_primary,
                "fault_contributing_factors": contributors,
                "recommendation": recommendation,
            },
        )


if __name__ == "__main__":
    main()
