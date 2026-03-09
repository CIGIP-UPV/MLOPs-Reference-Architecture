from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_recent_health(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "events": 0,
            "accuracy": 0.0,
            "false_alarm_rate": 0.0,
            "missed_alarm_rate": 0.0,
            "average_drift": 0.0,
            "average_latency_ms": 0.0,
            "status": "insufficient-data",
            "rollback_recommended": False,
        }
    accuracy = float((frame["prediction"] == frame["actual_breakage"]).mean())
    false_alarms = frame[(frame["prediction"] == 1) & (frame["actual_breakage"] == 0)]
    misses = frame[(frame["prediction"] == 0) & (frame["actual_breakage"] == 1)]
    safe_den = max(len(frame), 1)
    false_alarm_rate = len(false_alarms) / safe_den
    missed_alarm_rate = len(misses) / safe_den
    average_drift = float(frame["drift_score"].fillna(0.0).mean())
    average_latency_ms = float(frame["latency_ms"].fillna(0.0).mean())
    rollback_recommended = accuracy < 0.78 or missed_alarm_rate > 0.18 or average_drift > 0.72
    status = "healthy"
    if rollback_recommended:
        status = "rollback-recommended"
    elif accuracy < 0.84 or average_drift > 0.45:
        status = "watch"
    return {
        "events": int(len(frame)),
        "accuracy": round(accuracy, 6),
        "false_alarm_rate": round(false_alarm_rate, 6),
        "missed_alarm_rate": round(missed_alarm_rate, 6),
        "average_drift": round(average_drift, 6),
        "average_latency_ms": round(average_latency_ms, 6),
        "status": status,
        "rollback_recommended": rollback_recommended,
    }
