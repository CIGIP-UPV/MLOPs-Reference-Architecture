from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import psycopg2
import requests

from industrial_mlops.cnc_data import CNCMachineState, generate_stream_event
from industrial_mlops.config import CONFIG
from industrial_mlops.db import load_training_dataset
from industrial_mlops.orchestration import manual_promote
from industrial_mlops.registry import get_production_version, train_and_register
from industrial_mlops.security import sign_payload


RESULT_DIR = Path(os.environ.get("OTA_RESULTS_DIR", "/results/ota_continuity"))
EVENT_COUNT = int(os.environ.get("OTA_PROFILE_EVENT_COUNT", "1200"))
EVENT_INTERVAL = float(os.environ.get("OTA_PROFILE_EVENT_INTERVAL", "0.05"))
UPDATE_TRIGGER_SECONDS = float(os.environ.get("OTA_UPDATE_TRIGGER_SECONDS", "8.0"))
EDGE_STATE_PATH = Path(os.environ.get("EDGE_STATE_PATH", "/edge-state/deployment_state.json"))


class PublisherThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result: dict[str, Any] = {}
        self._finished = threading.Event()

    def run(self) -> None:
        rng = np.random.default_rng(42)
        state = CNCMachineState(machine_id=CONFIG.edge_machine_id, cell_id=CONFIG.edge_cell_id)
        client = mqtt.Client()
        if CONFIG.mqtt_username:
            client.username_pw_set(CONFIG.mqtt_username, CONFIG.mqtt_password)
        client.connect(CONFIG.mqtt_host, CONFIG.mqtt_port, 60)
        client.loop_start()
        sent = 0
        started = time.time()
        try:
            for _ in range(EVENT_COUNT):
                payload = generate_stream_event(state, rng)
                payload["schema_name"] = "synthetic-cnc"
                payload["signature"] = sign_payload(payload, CONFIG.shared_secret)
                info = client.publish(CONFIG.mqtt_sensor_topic, json.dumps(payload), qos=1)
                info.wait_for_publish()
                sent += 1
                if EVENT_INTERVAL > 0:
                    time.sleep(EVENT_INTERVAL)
        finally:
            client.loop_stop()
            client.disconnect()
            self.result = {
                "event_count_requested": EVENT_COUNT,
                "event_count_sent": sent,
                "elapsed_seconds": round(time.time() - started, 6),
            }
            self._finished.set()

    def wait(self) -> dict[str, Any]:
        self._finished.wait(timeout=max(EVENT_COUNT * max(EVENT_INTERVAL, 0.0) + 120.0, 300.0))
        return self.result


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def db_conn():
    return psycopg2.connect(CONFIG.admin_dsn(CONFIG.factory_db_name))


def wait_for(name: str, fn, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            if fn():
                return
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {name}. Last error: {last_error}")


def wait_for_edge_ready() -> bool:
    return requests.get("http://edge-inference-svc.edge-tier.svc.cluster.local:8010/metrics", timeout=5).status_code == 200


def wait_for_sync_ready() -> bool:
    return requests.get("http://edge-sync-svc.edge-tier.svc.cluster.local:8012/metrics", timeout=5).status_code == 200


def wait_for_db() -> bool:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1


def metrics_snapshot() -> dict[str, float | None]:
    edge_metrics = requests.get("http://edge-inference-svc.edge-tier.svc.cluster.local:8010/metrics", timeout=5)
    sync_metrics = requests.get("http://edge-sync-svc.edge-tier.svc.cluster.local:8012/metrics", timeout=5)
    edge_metrics.raise_for_status()
    sync_metrics.raise_for_status()
    return {
        "edge_inference_predictions_total": extract_metric(edge_metrics.text, "edge_inference_predictions_total"),
        "edge_inference_schema_mismatch_total": extract_metric(edge_metrics.text, "edge_inference_schema_mismatch_total"),
        "edge_sync_generation": extract_metric(sync_metrics.text, "edge_sync_generation"),
        "edge_sync_commands_total": extract_metric(sync_metrics.text, "edge_sync_commands_total"),
        "edge_sync_failures_total": extract_metric(sync_metrics.text, "edge_sync_failures_total"),
    }


def extract_metric(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(name + " "):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


def read_deployment_state() -> dict[str, Any] | None:
    if not EDGE_STATE_PATH.exists():
        return None
    return json.loads(EDGE_STATE_PATH.read_text(encoding="utf-8"))


def query_frame(query: str, params: tuple[Any, ...]) -> pd.DataFrame:
    with db_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)


def query_inference_rows(started_at: datetime, ended_at: datetime) -> pd.DataFrame:
    start = iso_timestamp(started_at - timedelta(seconds=2))
    end = iso_timestamp(ended_at + timedelta(seconds=2))
    return query_frame(
        "SELECT event_time, cycle_id, deployment_generation, model_name, model_version, prediction, actual_breakage, drift_score, latency_ms "
        "FROM inference_events WHERE event_time >= %s AND event_time <= %s ORDER BY event_time",
        (start, end),
    )


def query_edge_sync_rows(started_at: datetime, ended_at: datetime) -> pd.DataFrame:
    start = iso_timestamp(started_at - timedelta(seconds=5))
    end = iso_timestamp(ended_at + timedelta(seconds=5))
    return query_frame(
        "SELECT observed_at, machine_id, deployment_generation, model_version, sync_state, ota_latency_ms, notes "
        "FROM edge_sync_status WHERE observed_at >= %s AND observed_at <= %s ORDER BY observed_at",
        (start, end),
    )


def generation_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame["deployment_generation"].astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def generation_switches(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    series = frame["deployment_generation"].astype(int).tolist()
    transitions = 0
    for previous, current in zip(series, series[1:]):
        if current != previous:
            transitions += 1
    return transitions


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    wait_for("edge metrics", wait_for_edge_ready)
    wait_for("edge sync metrics", wait_for_sync_ready)
    wait_for("timescale", wait_for_db)

    before_state = read_deployment_state()
    before_metrics = metrics_snapshot()
    started_at = now_utc()

    publisher = PublisherThread()
    publisher.start()
    time.sleep(UPDATE_TRIGGER_SECONDS)
    issue_started_at = now_utc()

    previous = get_production_version(CONFIG.edge_bootstrap_model_name)
    frame = load_training_dataset(limit=5000)
    candidate = train_and_register(
        frame,
        reason="k3s-ota-continuity-candidate",
        run_name="k3s-ota-continuity-candidate",
    )
    promotion = manual_promote(
        candidate["model_version"],
        reason="k3s-ota-continuity-experiment",
        issued_by="k3s-ota-continuity",
    )
    current = get_production_version(CONFIG.edge_bootstrap_model_name)
    promotion_finished_at = now_utc()

    publisher_result = publisher.wait()
    ended_at = now_utc()
    time.sleep(5)

    after_state = read_deployment_state()
    after_metrics = metrics_snapshot()
    inference = query_inference_rows(started_at, ended_at)
    sync_rows = query_edge_sync_rows(started_at, ended_at)

    inference_path = RESULT_DIR / "inference_timeline.csv"
    update_path = RESULT_DIR / "update_timeline.csv"
    inference.to_csv(inference_path, index=False)
    update_rows: list[dict[str, Any]] = []
    if before_state:
        update_rows.append({"phase": "before", **before_state})
    update_rows.append(
        {
            "phase": "promotion",
            "issued_at": iso_timestamp(issue_started_at),
            "finished_at": iso_timestamp(promotion_finished_at),
            "previous_version": None if not previous else previous.get("version"),
            "candidate_version": candidate.get("model_version"),
            "current_version": None if not current else current.get("version"),
            "promotion_status": promotion.get("status"),
        }
    )
    if after_state:
        update_rows.append({"phase": "after", **after_state})
    if not sync_rows.empty:
        for row in sync_rows.to_dict(orient="records"):
            row["phase"] = "edge_sync_status"
            update_rows.append(row)
    with update_path.open("w", encoding="utf-8", newline="") as handle:
        if update_rows:
            fieldnames = sorted({key for row in update_rows for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(update_rows)

    counts = generation_counts(inference)
    prev_generation = None if not before_state else int(before_state.get("deployment_generation", 0))
    next_generation = None if not after_state else int(after_state.get("deployment_generation", 0))
    first_new_cycle = None
    last_old_cycle = None
    if not inference.empty:
        if next_generation is not None:
            rows_new = inference[inference["deployment_generation"].astype(int) == next_generation]
            if not rows_new.empty:
                first_new_cycle = int(rows_new.iloc[0]["cycle_id"])
        if prev_generation is not None:
            rows_old = inference[inference["deployment_generation"].astype(int) == prev_generation]
            if not rows_old.empty:
                last_old_cycle = int(rows_old.iloc[-1]["cycle_id"])

    ota_latency_values = sync_rows["ota_latency_ms"].astype(float).tolist() if not sync_rows.empty and "ota_latency_ms" in sync_rows else []
    summary = {
        "observed_at": iso_timestamp(now_utc()),
        "publisher": publisher_result,
        "previous_generation": prev_generation,
        "accepted_generation_after": next_generation,
        "previous_version": None if not previous else previous.get("version"),
        "accepted_version_after": None if not current else current.get("version"),
        "persisted_inferences": int(len(inference)),
        "generation_counts": counts,
        "generation_switches": generation_switches(inference),
        "first_new_generation_cycle": first_new_cycle,
        "last_previous_generation_cycle": last_old_cycle,
        "sync_failures_delta": int(round((after_metrics["edge_sync_failures_total"] or 0.0) - (before_metrics["edge_sync_failures_total"] or 0.0))),
        "sync_commands_delta": int(round((after_metrics["edge_sync_commands_total"] or 0.0) - (before_metrics["edge_sync_commands_total"] or 0.0))),
        "schema_mismatch_delta": int(round((after_metrics["edge_inference_schema_mismatch_total"] or 0.0) - (before_metrics["edge_inference_schema_mismatch_total"] or 0.0))),
        "prediction_counter_delta": int(round((after_metrics["edge_inference_predictions_total"] or 0.0) - (before_metrics["edge_inference_predictions_total"] or 0.0))),
        "ota_latency_ms_mean": round(float(np.mean(ota_latency_values)), 6) if ota_latency_values else None,
        "ota_latency_ms_max": round(float(np.max(ota_latency_values)), 6) if ota_latency_values else None,
        "single_clean_switch_observed": generation_switches(inference) == 1,
        "issue_time": iso_timestamp(issue_started_at),
        "promotion_finish_time": iso_timestamp(promotion_finished_at),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# OTA Continuity Report",
        "",
        "## Scope",
        "",
        "This Job publishes a continuous synthetic stream and promotes a new validated model generation while the stream is still active.",
        "",
        "## Directly observed",
        "",
        f"- Requested events: `{publisher_result['event_count_requested']}`.",
        f"- Sent events: `{publisher_result['event_count_sent']}`.",
        f"- Persisted inference rows: `{summary['persisted_inferences']}`.",
        f"- Previous generation / accepted generation after: `{summary['previous_generation']}` -> `{summary['accepted_generation_after']}`.",
        f"- Previous version / accepted version after: `{summary['previous_version']}` -> `{summary['accepted_version_after']}`.",
        f"- Generation switch count in persisted sequence: `{summary['generation_switches']}`.",
        f"- First cycle with the new generation: `{summary['first_new_generation_cycle']}`.",
        f"- Last cycle with the previous generation: `{summary['last_previous_generation_cycle']}`.",
        f"- Sync commands delta: `{summary['sync_commands_delta']}`.",
        f"- Sync failures delta: `{summary['sync_failures_delta']}`.",
        f"- Schema mismatch delta: `{summary['schema_mismatch_delta']}`.",
        f"- Mean OTA latency from `edge_sync_status`: `{summary['ota_latency_ms_mean']}` ms.",
        f"- Max OTA latency from `edge_sync_status`: `{summary['ota_latency_ms_max']}` ms.",
        "",
        "## Caveats",
        "",
        "- Continuity is inferred from persisted inference rows, sync status rows, and metrics counters.",
        "- This Job does not prove zero-interruption switching at a hard real-time timescale.",
        "- The result is an operational observation for the local K3s deployment.",
    ]
    report = "\n".join(report_lines) + "\n"
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print("OTA_CONTINUITY_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
