import os
import time
import logging
import requests
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from cloud_ai.explanation import explain_fault
from cloud_ai.failure_model import FAILURE_FEATURES
from cloud_ai.history import blend, summarize_history
from cloud_ai.recommendation import recommend_action
from cloud_ai.rul_model import RUL_FEATURES
from cloud_ai.schemas import CloudInput

# Force override env vars from .env file
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("main_poller")

# Environment vars mapped to variables
SOURCE_GET_API_URL = os.getenv("SOURCE_GET_API_URL", "http://localhost:8000/api/intelligence/data/unprocessed")
DESTINATION_POST_API_URL = os.getenv("DESTINATION_POST_API_URL", "http://localhost:8000/api/mock/destination")
POINTS_TO_FETCH = int(os.getenv("POINTS_TO_FETCH", "10"))
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "10"))
VEHICLES_TO_MONITOR = [v.strip() for v in os.getenv("VEHICLES_TO_MONITOR", "V-123").split(",") if v.strip()]

# Paths for models
RUL_MODEL_PATH = Path("cloud_ai") / "rul_model.pkl" if (Path("cloud_ai") / "rul_model.pkl").exists() else Path("rul_model.pkl")
FAILURE_MODEL_PATH = Path("cloud_ai") / "failure_model.pkl" if (Path("cloud_ai") / "failure_model.pkl").exists() else Path("failure_model.pkl")


class CloudAIPipeline:
    """Encapsulates the ML models and prediction logic."""

    def __init__(self):
        if not RUL_MODEL_PATH.exists() or not FAILURE_MODEL_PATH.exists():
            raise RuntimeError(
                f"Model files not found. Needed: {RUL_MODEL_PATH} and {FAILURE_MODEL_PATH}"
            )
        logger.info("Loading RUL and Failure models...")
        self.rul_model = joblib.load(RUL_MODEL_PATH)
        self.failure_model = joblib.load(FAILURE_MODEL_PATH)
        logger.info("Models loaded successfully.")

    def process_vehicle_data(self, history_records: list[dict]) -> dict:
        """
        Process a list of historical records (newest to oldest).
        Returns a dict of AI-augmented fields to append to the payload.
        """
        if not history_records:
            logger.warning("No history records provided to AI pipeline.")
            return {}

        # 1. The newest record is the "current" state
        # We assume history_records is sorted newest first or we just take the first one
        current_data_dict = history_records[0]
        
        # We validate the current data using the CloudInput schema to ensure types and required fields
        # If the input misses some fields that CloudInput strictly requires, this might raise ValidationError.
        # We catch it in the caller.
        try:
            data = CloudInput(**current_data_dict)
        except Exception as e:
            logger.error(f"Validation error on incoming data for vehicle {current_data_dict.get('vehicle_id')}: {e}")
            raise

        # 2. Summarize historical data
        history_snapshot = summarize_history(history_records)

        # 3. Blend and predict Engine RUL
        rul_feature_map = {
            "thermal_stress_index": blend(data.thermal_stress_index, history_snapshot.avg_thermal_stress_index),
            "brake_health_index": blend(data.brake_rul_pct / 100.0, None if history_snapshot.avg_brake_rul_pct is None else history_snapshot.avg_brake_rul_pct / 100.0),
            "mechanical_vibration_anomaly_score": blend(data.mechanical_vibration_anomaly_score, history_snapshot.avg_vibration_anomaly_score),
            "electrical_charging_efficiency_score": blend(data.electrical_charging_efficiency_score, history_snapshot.avg_charging_efficiency),
            "vehicle_health_score": blend(data.vehicle_health_score, history_snapshot.avg_vehicle_health_score),
        }
        rul_vector = pd.DataFrame([rul_feature_map])[RUL_FEATURES]
        predicted_engine_rul_pct = float(self.rul_model.predict(rul_vector)[0])
        predicted_engine_rul_pct = max(0.0, min(100.0, predicted_engine_rul_pct))

        # 4. Blend and predict Failure Probability
        blended_engine_rul_for_failure = blend(predicted_engine_rul_pct, history_snapshot.avg_engine_rul_pct)
        blended_brake_rul_for_failure = blend(data.brake_rul_pct, history_snapshot.avg_brake_rul_pct)
        blended_battery_rul_for_failure = blend(data.battery_rul_pct, history_snapshot.avg_battery_rul_pct)
        blended_thermal_stress_for_failure = blend(data.thermal_stress_index, history_snapshot.avg_thermal_stress_index)
        blended_vibration_for_failure = blend(
            data.mechanical_vibration_anomaly_score,
            history_snapshot.avg_vibration_anomaly_score,
        )

        failure_vector = pd.DataFrame(
            [
                {
                    "engine_rul_pct": blended_engine_rul_for_failure,
                    "brake_rul_pct": blended_brake_rul_for_failure,
                    "battery_rul_pct": blended_battery_rul_for_failure,
                    "thermal_stress_index": blended_thermal_stress_for_failure,
                    "mechanical_vibration_anomaly_score": blended_vibration_for_failure,
                }
            ]
        )[FAILURE_FEATURES]
        failure_prob = float(self.failure_model.predict_proba(failure_vector)[0][1])

        # 5. Explain Fault & Recommend Action
        fault_primary, contributors = explain_fault(data, failure_prob)
        recommendation = recommend_action(
            failure_prob=failure_prob,
            engine_rul_pct=predicted_engine_rul_pct,
            brake_rul_pct=data.brake_rul_pct,
            battery_rul_pct=data.battery_rul_pct,
            fault_primary=fault_primary,
            contributing_factors=contributors,
        )

        # 6. Return Augmented Fields Dict
        return {
            "source_id": "cloud_ai_agent",
            "engine_rul_pct": round(predicted_engine_rul_pct, 2),
            "brake_rul_pct": round(data.brake_rul_pct, 2),
            "battery_rul_pct": round(data.battery_rul_pct, 2),
            "fault_primary": fault_primary,
            "fault_contributing_factor": contributors,
            "fault_failure_probability": round(failure_prob, 2),
            "recommendation_service_priority": recommendation.get("recommendation_service_priority", "normal"),
            "recommendation_suggested_action": recommendation.get("recommendation_suggested_action", ""),
            "recommendation_safe_operating_limit_km": recommendation.get("recommendation_safe_operating_limit_km", 0),
        }


def poll_and_forward(pipeline: CloudAIPipeline, vehicle_id: str):
    """Hits the external GET endpoint, runs AI processing, and POSTs the result."""
    try:
        # 1. Fetch data from source GET endpoint
        # The API requires vehicle_id and limit as query parameters
        
        url_with_params = f"{SOURCE_GET_API_URL}?vehicle_id={vehicle_id}&limit={POINTS_TO_FETCH}"
        logger.debug(f"Fetching from {url_with_params}")
        
        response = requests.get(url_with_params, timeout=10)
        response.raise_for_status()
        
        data_payload = response.json()
        
        # Handle cases where the endpoint returns a list of history records, or a dict.
        # We'll normalize it to a list of records representing the vehicle's history.
        if isinstance(data_payload, dict):
            # If it's a dict containing a 'data' list
            if "data" in data_payload and isinstance(data_payload["data"], list):
                history_records = data_payload["data"]
            else:
                # Treat as a single record
                history_records = [data_payload]
        elif isinstance(data_payload, list):
            history_records = data_payload
        else:
            logger.error("Unexpected payload format from source GET API.")
            return

        if not history_records:
            logger.info("No data received from source API.")
            return
            
        # Ensure it's sorted newest first based on timestamp_ms
        history_records.sort(key=lambda x: x.get("timestamp_ms", 0), reverse=True)
            
        # 2. Extract the newest record (the "current" state) to forward
        current_record = history_records[0]
        
        # 3. Process the history array through the AI pipeline
        ai_insights = pipeline.process_vehicle_data(history_records)
        
        # 4. Merge original record + AI insights
        forward_payload = {**current_record, **ai_insights}
        
        # 5. POST to destination API
        logger.debug(f"Forwarding augmented payload to {DESTINATION_POST_API_URL}")
        post_response = requests.post(DESTINATION_POST_API_URL, json=forward_payload, timeout=10)
        post_response.raise_for_status()
        
        logger.info(f"Successfully processed and forwarded data for vehicle {forward_payload.get('vehicle_id', "unknown")}.")

    except requests.RequestException as e:
        logger.error(f"HTTP Request failed: {e}")
    except Exception as e:
        logger.error(f"Error during polling/processing cycle: {e}", exc_info=True)


def main():
    logger.info("Starting Cloud AI Main Polling Service...")
    logger.info(f"Source API: {SOURCE_GET_API_URL}")
    logger.info(f"Destination API: {DESTINATION_POST_API_URL}")
    logger.info(f"Monitoring Vehicles: {VEHICLES_TO_MONITOR}")
    logger.info(f"Poll Interval: {POLL_INTERVAL_SEC} seconds")
    
    pipeline = CloudAIPipeline()
    
    while True:
        for vehicle_id in VEHICLES_TO_MONITOR:
            poll_and_forward(pipeline, vehicle_id)
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
