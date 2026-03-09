from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "spindle_speed",
    "feed_rate",
    "vibration_x",
    "vibration_y",
    "acoustic_emission",
    "spindle_temp",
    "motor_current",
    "tool_wear",
    "material_hardness",
]

TARGET_COLUMN = "actual_breakage"


@dataclass
class CNCMachineState:
    machine_id: str = "cnc-cell-01"
    cell_id: str = "cell-01"
    tool_id: int = 1
    cycle_id: int = 0
    tool_wear: float = 0.08
    spindle_speed_base: float = 8200.0
    feed_rate_base: float = 0.42


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def _sample_operating_point(rng: np.random.Generator, wear: float) -> dict[str, float]:
    spindle_speed = rng.normal(8200 - 350 * wear, 180)
    feed_rate = rng.normal(0.42 - 0.03 * wear, 0.02)
    vibration_x = rng.normal(0.35 + 0.55 * wear, 0.07)
    vibration_y = rng.normal(0.30 + 0.45 * wear, 0.06)
    acoustic_emission = rng.normal(57 + 26 * wear, 5.5)
    spindle_temp = rng.normal(50 + 28 * wear, 4.0)
    motor_current = rng.normal(18 + 7.5 * wear, 1.2)
    material_hardness = rng.normal(215 + 8 * math.sin(wear * math.pi), 5.0)
    return {
        "spindle_speed": round(float(spindle_speed), 3),
        "feed_rate": round(float(feed_rate), 4),
        "vibration_x": round(float(max(vibration_x, 0.05)), 4),
        "vibration_y": round(float(max(vibration_y, 0.05)), 4),
        "acoustic_emission": round(float(max(acoustic_emission, 10.0)), 3),
        "spindle_temp": round(float(spindle_temp), 3),
        "motor_current": round(float(max(motor_current, 1.0)), 3),
        "tool_wear": round(float(np.clip(wear, 0.0, 1.4)), 4),
        "material_hardness": round(float(material_hardness), 3),
    }


def _breakage_probability(row: pd.Series | dict[str, float]) -> float:
    tool_wear = float(row["tool_wear"])
    vibration = (float(row["vibration_x"]) + float(row["vibration_y"])) / 2.0
    acoustic = float(row["acoustic_emission"])
    spindle_temp = float(row["spindle_temp"])
    motor_current = float(row["motor_current"])
    operating_stress = 0.0015 * max(float(row["spindle_speed"]) - 7800.0, 0.0)
    probability = _sigmoid(
        -8.8
        + 8.4 * tool_wear
        + 2.2 * vibration
        + 0.045 * acoustic
        + 0.06 * spindle_temp
        + 0.11 * motor_current
        + operating_stress
    )
    return float(np.clip(probability, 0.001, 0.995))


def build_reference_profile(frame: pd.DataFrame) -> dict[str, dict[str, float | list[float]]]:
    profile: dict[str, dict[str, float | list[float]]] = {}
    for feature in FEATURE_COLUMNS:
        values = frame[feature].astype(float).to_numpy()
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, 11)).tolist()
        profile[feature] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values) + 1e-9), 6),
            "quantiles": [round(float(item), 6) for item in quantiles],
            "sample": [round(float(item), 6) for item in values[: min(len(values), 400)]],
        }
    return profile


def generate_historical_dataset(
    n_samples: int = 2500,
    seed: int = 42,
    machine_id: str = "cnc-cell-01",
    cell_id: str = "cell-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | datetime]] = []
    start_time = datetime.now(timezone.utc) - timedelta(minutes=n_samples)
    state = CNCMachineState(machine_id=machine_id, cell_id=cell_id)
    for offset in range(n_samples):
        state.cycle_id += 1
        state.tool_wear = float(np.clip(state.tool_wear + rng.uniform(0.0008, 0.0045), 0.02, 1.35))
        point = _sample_operating_point(rng, state.tool_wear)
        probability = _breakage_probability(point)
        label = int(rng.random() < probability)
        row = {
            "event_time": start_time + timedelta(minutes=offset),
            "machine_id": machine_id,
            "cell_id": cell_id,
            "cycle_id": state.cycle_id,
            "tool_id": state.tool_id,
            **point,
            "actual_breakage": label,
            "breakage_probability": round(probability, 6),
            "source_protocol": "seed",
        }
        rows.append(row)
        if label or state.tool_wear > 1.05:
            state.tool_id += 1
            state.tool_wear = float(rng.uniform(0.04, 0.12))
    return pd.DataFrame(rows)


def generate_stream_event(state: CNCMachineState, rng: np.random.Generator | None = None) -> dict[str, float | int | str]:
    rng = rng or np.random.default_rng()
    state.cycle_id += 1
    wear_increment = rng.uniform(0.0025, 0.013)
    state.tool_wear = float(np.clip(state.tool_wear + wear_increment, 0.02, 1.35))
    point = _sample_operating_point(rng, state.tool_wear)
    probability = _breakage_probability(point)
    label = int(rng.random() < probability)
    payload = {
        "event_time": datetime.now(timezone.utc).isoformat(),
        "machine_id": state.machine_id,
        "cell_id": state.cell_id,
        "cycle_id": state.cycle_id,
        "tool_id": state.tool_id,
        **point,
        "actual_breakage": label,
        "breakage_probability": round(probability, 6),
        "source_protocol": "mqtt",
    }
    if label or state.tool_wear > 1.08:
        state.tool_id += 1
        state.tool_wear = float(rng.uniform(0.04, 0.10))
    return payload


def frame_from_events(events: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(events))
    if frame.empty:
        return frame
    return frame.sort_values("event_time").reset_index(drop=True)
