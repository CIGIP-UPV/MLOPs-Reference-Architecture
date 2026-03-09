from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


COMPANY_DATASET_ROOT = Path(__file__).resolve().parent / "data" / "company_cnc"
COMPANY_DATASET_PATH = COMPANY_DATASET_ROOT / "nakamura_reference_windows.csv"
COMPANY_METADATA_PATH = COMPANY_DATASET_ROOT / "nakamura_reference_windows.metadata.json"
COMPANY_SCHEMA_NAME = "company-nakamura-reference"
COMPANY_FEATURE_COLUMNS = [
    "spindle_speed_p1_rpm",
    "spindle_speed_p2_rpm",
    "feed_rate_p1_mm_min",
    "feed_rate_p2_mm_min",
    "spindle_motor_load_p1_pct",
    "spindle_motor_load_p2_pct",
    "servo_load_current_p1_pct",
    "servo_load_current_p2_pct",
    "cutting_time_p1_s",
    "cutting_time_p2_s",
    "temp_apc_p1_c",
    "temp_apc_p2_c",
]
COMPANY_TARGET_COLUMN = "actual_breakage"


def company_dataset_exists() -> bool:
    return COMPANY_DATASET_PATH.exists() and COMPANY_METADATA_PATH.exists()


def load_company_reference_dataset() -> pd.DataFrame:
    if not COMPANY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Company CNC dataset not found at {COMPANY_DATASET_PATH}")
    frame = pd.read_csv(COMPANY_DATASET_PATH)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, format="ISO8601")
    return frame


def load_company_reference_metadata() -> dict[str, object]:
    if not COMPANY_METADATA_PATH.exists():
        raise FileNotFoundError(f"Company CNC metadata not found at {COMPANY_METADATA_PATH}")
    return json.loads(COMPANY_METADATA_PATH.read_text(encoding="utf-8"))


def build_company_stream_payload(
    row: pd.Series,
    *,
    machine_id: str,
    cell_id: str,
    cycle_id: int,
    tool_id: int,
    event_time: str,
) -> dict[str, object]:
    seconds_to_target = row.get("seconds_to_target_alarm")
    if pd.isna(seconds_to_target) or seconds_to_target == "":
        seconds_to_target = None
    payload: dict[str, object] = {
        "schema_name": COMPANY_SCHEMA_NAME,
        "event_time": event_time,
        "machine_id": machine_id,
        "cell_id": cell_id,
        "cycle_id": cycle_id,
        "tool_id": tool_id,
        "actual_breakage": int(row[COMPANY_TARGET_COLUMN]),
        "current_alarm_code": str(row.get("current_alarm_code", "0.0")),
        "seconds_to_target_alarm": float(seconds_to_target) if seconds_to_target is not None else None,
        "selection_reason": str(row.get("selection_reason", "company-replay")),
        "source_protocol": "mqtt-company",
    }
    for feature in COMPANY_FEATURE_COLUMNS:
        payload[feature] = float(row[feature])
    return payload
