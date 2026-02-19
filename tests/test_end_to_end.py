import os
import tempfile
import unittest

import joblib
import pandas as pd

from cloud_ai.data_generation import generate_synthetic_cloud_history
from cloud_ai.explanation import explain_fault
from cloud_ai.failure_model import FAILURE_FEATURES, train_failure_model
from cloud_ai.recommendation import recommend_action
from cloud_ai.rul_model import RUL_FEATURES, train_rul_model
from cloud_ai.schemas import CloudInput


class EndToEndCloudAITests(unittest.TestCase):
    def test_training_and_local_advisory_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "cloud_health_history.csv")
            rul_path = os.path.join(tmpdir, "rul_model.pkl")
            failure_path = os.path.join(tmpdir, "failure_model.pkl")

            generate_synthetic_cloud_history(output_path=csv_path, rows=300, seed=7)
            train_rul_model(data_path=csv_path, model_path=rul_path)
            train_failure_model(data_path=csv_path, model_path=failure_path)

            rul_model = joblib.load(rul_path)
            failure_model = joblib.load(failure_path)

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

            rul_vector = pd.DataFrame(
                [
                    {
                        "thermal_stress_index": payload.thermal_stress_index,
                        "brake_health_index": payload.brake_rul_pct / 100.0,
                        "mechanical_vibration_anomaly_score": payload.mechanical_vibration_anomaly_score,
                        "electrical_charging_efficiency_score": payload.electrical_charging_efficiency_score,
                        "vehicle_health_score": payload.vehicle_health_score,
                    }
                ]
            )[RUL_FEATURES]
            predicted_engine_rul_pct = max(0.0, min(100.0, float(rul_model.predict(rul_vector)[0])))

            failure_vector = pd.DataFrame(
                [
                    {
                        "engine_rul_pct": predicted_engine_rul_pct,
                        "brake_rul_pct": payload.brake_rul_pct,
                        "battery_rul_pct": payload.battery_rul_pct,
                        "thermal_stress_index": payload.thermal_stress_index,
                        "mechanical_vibration_anomaly_score": payload.mechanical_vibration_anomaly_score,
                    }
                ]
            )[FAILURE_FEATURES]
            failure_prob = float(failure_model.predict_proba(failure_vector)[0][1])

            fault_primary, contributors = explain_fault(payload, failure_prob)
            recommendation = recommend_action(
                failure_prob=failure_prob,
                engine_rul_pct=predicted_engine_rul_pct,
                brake_rul_pct=payload.brake_rul_pct,
                fault_primary=fault_primary,
            )

            self.assertGreaterEqual(predicted_engine_rul_pct, 0.0)
            self.assertLessEqual(predicted_engine_rul_pct, 100.0)
            self.assertGreaterEqual(failure_prob, 0.0)
            self.assertLessEqual(failure_prob, 1.0)
            self.assertIsInstance(fault_primary, str)
            self.assertGreaterEqual(len(contributors), 1)
            self.assertIn("recommendation_service_priority", recommendation)
            self.assertIn("recommendation_suggested_action", recommendation)
            self.assertIn("recommendation_safe_operating_limit_km", recommendation)


if __name__ == "__main__":
    unittest.main()
