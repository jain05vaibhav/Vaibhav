import unittest

from cloud_ai.explanation import explain_fault
from cloud_ai.recommendation import recommend_action
from cloud_ai.schemas import CloudInput


class AdvisoryLogicTests(unittest.TestCase):
    def _payload(self) -> CloudInput:
        return CloudInput(
            vehicle_id="VIT_CAR_001",
            timestamp_ms=1707051123456,
            thermal_brake_margin=-0.21,
            thermal_engine_margin=0.34,
            thermal_stress_index=0.82,
            mechanical_vibration_anomaly_score=0.77,
            mechanical_dominant_fault_band_hz=142.0,
            mechanical_vibration_rms=0.84,
            electrical_charging_efficiency_score=0.81,
            electrical_battery_health_pct=87,
            engine_rul_pct=62,
            brake_rul_pct=28,
            battery_rul_pct=74,
            vehicle_health_score=0.64,
        )

    def test_fault_explanation_returns_primary_and_contributors(self):
        primary, contributors = explain_fault(self._payload(), 0.61)
        self.assertEqual(primary, "BRAKE_THERMAL_SATURATION")
        self.assertIn("high_thermal_stress_index", contributors)
        self.assertIn("low_brake_rul_pct", contributors)

    def test_recommendation_policy(self):
        rec = recommend_action(0.8, 55, 35, "MULTI_FACTOR_FAILURE_RISK")
        self.assertEqual(rec["recommendation_service_priority"], "high")

        rec = recommend_action(0.5, 55, 35, "NO_DOMINANT_FAULT")
        self.assertEqual(rec["recommendation_service_priority"], "medium")

        rec = recommend_action(0.5, 45, 55, "NO_DOMINANT_FAULT")
        self.assertEqual(rec["recommendation_service_priority"], "low")

        rec = recommend_action(0.2, 70, 80, "NO_DOMINANT_FAULT")
        self.assertEqual(rec["recommendation_service_priority"], "normal")


if __name__ == "__main__":
    unittest.main()
