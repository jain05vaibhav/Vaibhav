"""Historical data access and feature aggregation for cloud advisory inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HistoryProvider(Protocol):
    """Abstract provider for fetching recent Section-6 cloud records."""

    def fetch_recent(self, vehicle_id: str, limit: int) -> list[dict]:
        """Return recent records for a vehicle sorted by recency (newest first)."""

    def save_record(self, record: dict) -> None:
        """Persist a Section-6 style record for future historical inference."""


class NoopHistoryProvider:
    """Default provider when no history backend is configured."""

    def fetch_recent(self, vehicle_id: str, limit: int) -> list[dict]:
        return []

    def save_record(self, record: dict) -> None:
        return None




class InMemoryHistoryProvider:
    """Simple in-process store, useful for local/testing history-aware inference."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def fetch_recent(self, vehicle_id: str, limit: int) -> list[dict]:
        filtered = [r for r in self._records if r.get("vehicle_id") == vehicle_id]
        filtered.sort(key=lambda row: row.get("timestamp_ms", 0), reverse=True)
        return filtered[:limit]

    def save_record(self, record: dict) -> None:
        self._records.append(dict(record))


@dataclass
class HistoricalSnapshot:
    points_used: int
    avg_thermal_stress_index: float | None = None
    avg_vibration_anomaly_score: float | None = None
    avg_charging_efficiency: float | None = None
    avg_vehicle_health_score: float | None = None
    avg_engine_rul_pct: float | None = None
    avg_brake_rul_pct: float | None = None
    avg_battery_rul_pct: float | None = None


def build_history_provider_from_env(backend: str | None = None) -> HistoryProvider:
    """Build history provider from a lightweight backend selector.

    Supported values:
    - ``memory``: keep recent Section-6 records in process memory
    - anything else: disable history persistence/fetching
    """

    selected = (backend or "").strip().lower()
    if selected == "memory":
        return InMemoryHistoryProvider()
    return NoopHistoryProvider()


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_history(records: list[dict]) -> HistoricalSnapshot:
    """Create aggregate snapshot from recent Section-6 historical records."""

    def collect(field: str) -> list[float]:
        result: list[float] = []
        for row in records:
            value = row.get(field)
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    return HistoricalSnapshot(
        points_used=len(records),
        avg_thermal_stress_index=_safe_mean(collect("thermal_stress_index")),
        avg_vibration_anomaly_score=_safe_mean(collect("mechanical_vibration_anomaly_score")),
        avg_charging_efficiency=_safe_mean(collect("electrical_charging_efficiency_score")),
        avg_vehicle_health_score=_safe_mean(collect("vehicle_health_score")),
        avg_engine_rul_pct=_safe_mean(collect("engine_rul_pct")),
        avg_brake_rul_pct=_safe_mean(collect("brake_rul_pct")),
        avg_battery_rul_pct=_safe_mean(collect("battery_rul_pct")),
    )


def blend(current: float, historical: float | None, current_weight: float = 0.7) -> float:
    """Blend current observation with historical aggregate when available."""

    if historical is None:
        return current
    historical_weight = 1.0 - current_weight
    return (current_weight * current) + (historical_weight * historical)
