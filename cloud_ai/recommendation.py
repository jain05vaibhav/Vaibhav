"""Policy-based maintenance recommendation engine (advisory only)."""


def recommend_action(
    failure_prob: float,
    engine_rul_pct: float,
    brake_rul_pct: float,
    fault_primary: str,
) -> dict[str, str | int]:
    if failure_prob > 0.75 or fault_primary == "BRAKE_THERMAL_SATURATION":
        return {
            "recommendation_service_priority": "high",
            "recommendation_suggested_action": "Brake inspection and pad replacement",
            "recommendation_safe_operating_limit_km": 120,
        }

    if brake_rul_pct < 40:
        return {
            "recommendation_service_priority": "medium",
            "recommendation_suggested_action": "Schedule brake maintenance within 2 weeks",
            "recommendation_safe_operating_limit_km": 300,
        }

    if engine_rul_pct < 50:
        return {
            "recommendation_service_priority": "low",
            "recommendation_suggested_action": "Monitor engine health trend and service soon",
            "recommendation_safe_operating_limit_km": 600,
        }

    return {
        "recommendation_service_priority": "normal",
        "recommendation_suggested_action": "No immediate maintenance required",
        "recommendation_safe_operating_limit_km": 1000,
    }
