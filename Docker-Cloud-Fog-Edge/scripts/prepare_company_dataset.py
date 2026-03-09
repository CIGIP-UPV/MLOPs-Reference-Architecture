#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable


TRAINING_DATA_MEMBER = "TRAINING DATA/data.csv"
TARGET_ALARMS = {"409.0", "410.0", "411.0"}
DEFAULT_HORIZON_SECONDS = 120

RAW_TO_CURATED_COLUMNS = {
    "average_01 Speed S P1": "spindle_speed_p1_rpm",
    "average_02 Speed S P2": "spindle_speed_p2_rpm",
    "average_03 Feed Rate P1": "feed_rate_p1_mm_min",
    "average_04 Feed Rate P2": "feed_rate_p2_mm_min",
    "average_05 Spindle Motor Load nakamura2 P1 S0": "spindle_motor_load_p1_pct",
    "average_06 Spindle Motor Load nakamura2 P2 S0": "spindle_motor_load_p2_pct",
    "average_24 Servo load current(a) Nakamura2 P1 A1": "servo_load_current_p1_pct",
    "average_28 Servo load current(a) Nakamura2 P2 A1": "servo_load_current_p2_pct",
    "average_31 Cutting time nakamura2 P1": "cutting_time_p1_s",
    "average_32 Cutting Time nakamura2 P2": "cutting_time_p2_s",
    "average_33 Temp #APC Nakamura2 P1 A0": "temp_apc_p1_c",
    "average_34 Temp #APC Nakamura2 P2 A0": "temp_apc_p2_c",
}


@dataclass(frozen=True)
class RowCounts:
    total_rows: int
    positive_rows: int
    healthy_rows: int
    other_alarm_rows: int
    time_min: str | None
    time_max: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact, reproducible Nakamura CNC reference dataset from the company archive.",
    )
    parser.add_argument(
        "--source-zip",
        required=True,
        help="Path to the company ZIP archive or repaired ZIP copy.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the curated CSV, metadata and review files will be written.",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=int,
        default=DEFAULT_HORIZON_SECONDS,
        help="Prediction horizon used to mark a row as positive.",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%d %H:%M:%S.%f+00:00"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value}")


def run(command: list[str], stdout=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, stdout=stdout, stderr=subprocess.PIPE)


def extract_training_csv(source_zip: Path, workdir: Path) -> Path:
    extracted_csv = workdir / "company_training_data.csv"
    with extracted_csv.open("w", encoding="utf-8") as handle:
        result = run(["unzip", "-p", str(source_zip), TRAINING_DATA_MEMBER], stdout=handle)
    if extracted_csv.exists() and extracted_csv.stat().st_size > 0:
        return extracted_csv

    local_zip_copy = workdir / source_zip.name
    shutil.copyfile(source_zip, local_zip_copy)
    repaired_zip = workdir / "company_training_data_repaired.zip"
    repair = run(["zip", "-FF", str(local_zip_copy), "--out", str(repaired_zip)])
    if repair.returncode != 0 or not repaired_zip.exists():
        raise RuntimeError(
            "The company ZIP archive could not be repaired automatically.\n"
            f"unzip stderr:\n{result.stderr}\nzip stderr:\n{repair.stderr}"
        )

    with extracted_csv.open("w", encoding="utf-8") as handle:
        repaired_result = run(["unzip", "-p", str(repaired_zip), TRAINING_DATA_MEMBER], stdout=handle)
    if not extracted_csv.exists() or extracted_csv.stat().st_size == 0:
        raise RuntimeError(
            "The repaired ZIP archive still could not provide TRAINING DATA/data.csv.\n"
            f"stderr:\n{repaired_result.stderr}"
        )
    return extracted_csv


def collect_target_timestamps(csv_path: Path) -> list[datetime]:
    timestamps: list[datetime] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Alarm"] in TARGET_ALARMS:
                timestamps.append(parse_timestamp(row["measurement_bucket"]))
    if not timestamps:
        raise RuntimeError("No target CNC alarms (409/410/411) were found in TRAINING DATA/data.csv.")
    return timestamps


def seconds_to_next_target(event_time: datetime, targets: list[datetime], target_index: int) -> tuple[float | None, int]:
    while target_index < len(targets) and targets[target_index] < event_time:
        target_index += 1
    if target_index >= len(targets):
        return None, target_index
    seconds = (targets[target_index] - event_time).total_seconds()
    return seconds, target_index


def count_rows(csv_path: Path, targets: list[datetime], horizon_seconds: int) -> RowCounts:
    total_rows = 0
    positive_rows = 0
    healthy_rows = 0
    other_alarm_rows = 0
    time_min: str | None = None
    time_max: str | None = None
    target_index = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            event_time = row["measurement_bucket"]
            time_min = event_time if time_min is None or event_time < time_min else time_min
            time_max = event_time if time_max is None or event_time > time_max else time_max
            seconds, target_index = seconds_to_next_target(parse_timestamp(event_time), targets, target_index)
            current_alarm = row["Alarm"]
            if seconds is not None and 0 <= seconds <= horizon_seconds:
                positive_rows += 1
            elif current_alarm == "0.0":
                healthy_rows += 1
            else:
                other_alarm_rows += 1

    return RowCounts(
        total_rows=total_rows,
        positive_rows=positive_rows,
        healthy_rows=healthy_rows,
        other_alarm_rows=other_alarm_rows,
        time_min=time_min,
        time_max=time_max,
    )


def should_select(index: int, total: int, budget: int) -> bool:
    if budget <= 0 or total <= 0:
        return False
    previous = (index * budget) // total
    current = ((index + 1) * budget) // total
    return current > previous


def build_curated_row(
    row: dict[str, str],
    actual_breakage: int,
    seconds_to_target: float | None,
    selection_reason: str,
) -> dict[str, str | int | float]:
    curated = {
        "event_time": row["measurement_bucket"],
        "selection_reason": selection_reason,
        "current_alarm_code": row["Alarm"],
        "seconds_to_target_alarm": round(seconds_to_target, 3) if seconds_to_target is not None else "",
        "actual_breakage": actual_breakage,
    }
    for raw_name, curated_name in RAW_TO_CURATED_COLUMNS.items():
        curated[curated_name] = float(row[raw_name])
    return curated


def select_rows(
    csv_path: Path,
    targets: list[datetime],
    counts: RowCounts,
    horizon_seconds: int,
) -> list[dict[str, str | int | float]]:
    healthy_budget = min(counts.healthy_rows, counts.positive_rows * 2)
    other_alarm_budget = min(counts.other_alarm_rows, counts.positive_rows)
    selected_rows: list[dict[str, str | int | float]] = []

    healthy_index = 0
    other_alarm_index = 0
    target_index = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_time = parse_timestamp(row["measurement_bucket"])
            seconds, target_index = seconds_to_next_target(event_time, targets, target_index)
            current_alarm = row["Alarm"]
            if seconds is not None and 0 <= seconds <= horizon_seconds:
                selected_rows.append(build_curated_row(row, actual_breakage=1, seconds_to_target=seconds, selection_reason="target_horizon"))
                continue

            if current_alarm == "0.0":
                if should_select(healthy_index, counts.healthy_rows, healthy_budget):
                    selected_rows.append(build_curated_row(row, actual_breakage=0, seconds_to_target=seconds, selection_reason="sampled_healthy"))
                healthy_index += 1
                continue

            if should_select(other_alarm_index, counts.other_alarm_rows, other_alarm_budget):
                selected_rows.append(build_curated_row(row, actual_breakage=0, seconds_to_target=seconds, selection_reason="sampled_other_alarm"))
            other_alarm_index += 1

    selected_rows.sort(key=lambda item: item["event_time"])
    return selected_rows


def assign_splits(rows: list[dict[str, str | int | float]]) -> None:
    total = len(rows)
    train_cutoff = int(total * 0.70)
    validation_cutoff = int(total * 0.85)
    for index, row in enumerate(rows):
        if index < train_cutoff:
            row["split"] = "train"
        elif index < validation_cutoff:
            row["split"] = "validation"
        else:
            row["split"] = "test"


def write_csv(path: Path, rows: Iterable[dict[str, str | int | float]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No curated rows were generated.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | int | float]], counts: RowCounts, source_zip: Path, horizon_seconds: int) -> dict[str, object]:
    selection_reasons: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    current_alarm_counts: dict[str, int] = {}
    positive_rows = 0
    for row in rows:
        selection_reasons[row["selection_reason"]] = selection_reasons.get(row["selection_reason"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        current_alarm = str(row["current_alarm_code"])
        current_alarm_counts[current_alarm] = current_alarm_counts.get(current_alarm, 0) + 1
        positive_rows += int(row["actual_breakage"])

    return {
        "dataset_name": "nakamura_reference_windows",
        "source_zip": str(source_zip),
        "source_member": TRAINING_DATA_MEMBER,
        "target_alarm_codes": sorted(TARGET_ALARMS),
        "label_definition": f"actual_breakage=1 if a target alarm occurs within the next {horizon_seconds} seconds.",
        "raw_dataset": {
            "rows": counts.total_rows,
            "positive_rows": counts.positive_rows,
            "healthy_rows": counts.healthy_rows,
            "other_alarm_rows": counts.other_alarm_rows,
            "time_min": counts.time_min,
            "time_max": counts.time_max,
        },
        "curated_dataset": {
            "rows": len(rows),
            "positive_rows": positive_rows,
            "negative_rows": len(rows) - positive_rows,
            "selection_reasons": selection_reasons,
            "split_counts": split_counts,
            "current_alarm_code_counts": current_alarm_counts,
        },
        "kept_columns": list(RAW_TO_CURATED_COLUMNS.values()),
        "dropped_raw_columns": "All max/min statistics and auxiliary files were excluded to keep the repository compact.",
    }


def main() -> None:
    args = parse_args()
    source_zip = Path(args.source_zip).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="company-cnc-") as temp_dir:
        temp_path = Path(temp_dir)
        extracted_csv = extract_training_csv(source_zip, temp_path)
        target_timestamps = collect_target_timestamps(extracted_csv)
        counts = count_rows(extracted_csv, target_timestamps, args.horizon_seconds)
        rows = select_rows(extracted_csv, target_timestamps, counts, args.horizon_seconds)
        assign_splits(rows)

        csv_path = output_dir / "nakamura_reference_windows.csv"
        metadata_path = output_dir / "nakamura_reference_windows.metadata.json"
        write_csv(csv_path, rows)
        metadata = summarize(rows, counts, source_zip, args.horizon_seconds)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
