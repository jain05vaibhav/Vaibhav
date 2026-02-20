from fastapi import FastAPI, Request
from cloud_ai.schemas import CloudInput

app = FastAPI()

MOCK_DATA = {
  "vehicle_id": "V-123",
  "timestamp_ms": 1700000000,
  "fog_decision_critical_class": 1,
  "fog_decision_actuation_triggered": 0,
  "fog_decision_confidence": 0.95,
  "thermal_brake_margin": 15.0,
  "thermal_engine_margin": 20.0,
  "thermal_stress_index": 0.8,
  "mechanical_vibration_anomaly_score": 0.75,
  "mechanical_dominant_fault_band_hz": 40.0,
  "mechanical_vibration_rms": 0.5,
  "electrical_charging_efficiency_score": 0.9,
  "electrical_battery_health_pct": 98.0,
  "engine_rul_pct": 60.0,
  "brake_rul_pct": 25.0,
  "battery_rul_pct": 90.0,
  "vehicle_health_score": 0.6,
  "trigger_measured_brake_temp_c": 300,
  "trigger_brake_temp_rise_rate": 5,
  "trigger_brake_health_index": 0.3,
  "fog_thermal_protection_active": True,
  "fog_brake_stress_mitigation_active": True,
  "fog_vibration_damping_mode_active": False,
  "fog_predictive_service_required": True,
  "fog_emergency_safeguard_active": False
}


@app.get("/api/mock/source")
def get_mock_source(limit: int = 50):
    # Return a list of records
    return [MOCK_DATA] * limit

@app.post("/api/mock/destination")
async def post_mock_destination(request: Request):
    data = await request.json()
    print("\n--- RECEIVED AT DESTINATION ---")
    import json
    print(json.dumps(data, indent=2))
    print("-------------------------------\n")
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
