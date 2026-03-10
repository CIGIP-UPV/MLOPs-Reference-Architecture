from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import psycopg2
import requests

from industrial_mlops.cnc_data import CNCMachineState, generate_stream_event
from industrial_mlops.config import CONFIG
from industrial_mlops.security import sign_payload


RESULT_DIR = Path(os.environ.get("EDGE_PROFILE_RESULTS_DIR", "/results/edge_profiling"))
EVENT_COUNT = int(os.environ.get("EDGE_PROFILE_EVENT_COUNT", "300"))
EVENT_INTERVAL = float(os.environ.get("EDGE_PROFILE_EVENT_INTERVAL", "0.05"))
STATS_INTERVAL = float(os.environ.get("EDGE_PROFILE_STATS_INTERVAL", "1.0"))
POD_NAMESPACE = os.environ.get("POD_NAMESPACE", "edge-tier")
TARGET_SELECTOR = os.environ.get("EDGE_PROFILE_SELECTOR", "app=edge-inference")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


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


def db_conn():
    return psycopg2.connect(CONFIG.admin_dsn(CONFIG.factory_db_name))


def metrics_text() -> str:
    response = requests.get("http://edge-inference-svc.edge-tier.svc.cluster.local:8010/metrics", timeout=5)
    response.raise_for_status()
    return response.text


def extract_metric(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(name + " "):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


def psql_frame(started_at: datetime, ended_at: datetime) -> pd.DataFrame:
    start = iso_timestamp(started_at - timedelta(seconds=2))
    end = iso_timestamp(ended_at + timedelta(seconds=2))
    query = (
        "SELECT event_time, cycle_id, latency_ms, prediction, actual_breakage, drift_score, model_name, model_version "
        "FROM inference_events "
        "WHERE event_time >= %s AND event_time <= %s "
        "ORDER BY event_time"
    )
    with db_conn() as conn:
        return pd.read_sql_query(query, conn, params=(start, end))


def parse_cpu_to_cores(raw: str) -> float:
    raw = raw.strip()
    if raw.endswith("n"):
        return float(raw[:-1]) / 1_000_000_000.0
    if raw.endswith("u"):
        return float(raw[:-1]) / 1_000_000.0
    if raw.endswith("m"):
        return float(raw[:-1]) / 1000.0
    return float(raw)


def parse_memory_to_mib(raw: str) -> float:
    raw = raw.strip()
    suffixes = {
        "Ki": 1.0 / 1024.0,
        "Mi": 1.0,
        "Gi": 1024.0,
        "Ti": 1024.0 * 1024.0,
        "K": 1000.0 / (1024.0 * 1024.0),
        "M": 1000.0 * 1000.0 / (1024.0 * 1024.0),
        "G": 1000.0 * 1000.0 * 1000.0 / (1024.0 * 1024.0),
    }
    for suffix, factor in suffixes.items():
        if raw.endswith(suffix):
            return float(raw[:-len(suffix)]) * factor
    return float(raw) / (1024.0 * 1024.0)


class KubernetesMetricsSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.supported = True
        self.error: str | None = None
        self.base_url = "https://kubernetes.default.svc"
        token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        self.token = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""
        self.verify = str(ca_path) if ca_path.exists() else True

    def _request(self, path: str, params: dict[str, str] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(self.base_url + path, headers=headers, params=params, timeout=5, verify=self.verify)
        response.raise_for_status()
        return response.json()

    def target_pod(self) -> str:
        payload = self._request(f"/api/v1/namespaces/{POD_NAMESPACE}/pods", params={"labelSelector": TARGET_SELECTOR})
        items = payload.get("items", [])
        if not items:
            raise RuntimeError("No edge-inference pod found for resource sampling.")
        return items[0]["metadata"]["name"]

    def sample(self) -> None:
        if not self.supported:
            return
        try:
            pod = self.target_pod()
            payload = self._request(f"/apis/metrics.k8s.io/v1beta1/namespaces/{POD_NAMESPACE}/pods/{pod}")
            container = next((item for item in payload.get("containers", []) if item.get("name") == "edge-inference"), None)
            if not container:
                raise RuntimeError("edge-inference container metrics not found.")
            self.samples.append(
                {
                    "observed_at": iso_timestamp(now_utc()),
                    "pod": pod,
                    "cpu_cores": parse_cpu_to_cores(container["usage"]["cpu"]),
                    "memory_mib": parse_memory_to_mib(container["usage"]["memory"]),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime fallback only
            self.supported = False
            self.error = str(exc)


def wait_for_edge_metrics() -> bool:
    return requests.get("http://edge-inference-svc.edge-tier.svc.cluster.local:8010/metrics", timeout=5).status_code == 200


def wait_for_db() -> bool:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1


def publish_events() -> dict[str, Any]:
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
    return {
        "event_count_requested": EVENT_COUNT,
        "event_count_sent": sent,
        "elapsed_seconds": round(time.time() - started, 6),
    }


def summarize_series(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "p99": round(float(np.percentile(arr, 99)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    wait_for("edge metrics", wait_for_edge_metrics)
    wait_for("timescale", wait_for_db)

    before_metrics = metrics_text()
    predictions_before = extract_metric(before_metrics, "edge_inference_predictions_total") or 0.0
    mismatches_before = extract_metric(before_metrics, "edge_inference_schema_mismatch_total") or 0.0

    sampler = KubernetesMetricsSampler()
    started_at = now_utc()
    publish_started = time.time()
    publisher_result = publish_events()
    publish_ended_at = now_utc()

    while time.time() - publish_started < max(EVENT_COUNT * max(EVENT_INTERVAL, 0.0) + 10.0, 15.0):
        sampler.sample()
        if time.time() - publish_started >= publisher_result["elapsed_seconds"] + 5.0:
            break
        time.sleep(STATS_INTERVAL)

    time.sleep(5)
    after_metrics = metrics_text()
    predictions_after = extract_metric(after_metrics, "edge_inference_predictions_total") or 0.0
    mismatches_after = extract_metric(after_metrics, "edge_inference_schema_mismatch_total") or 0.0

    frame = psql_frame(started_at, publish_ended_at)
    frame.to_csv(RESULT_DIR / "metrics.csv", index=False)

    latency_summary = summarize_series(frame["latency_ms"].astype(float).tolist() if not frame.empty else [])
    cpu_summary = summarize_series([sample["cpu_cores"] for sample in sampler.samples])
    memory_summary = summarize_series([sample["memory_mib"] for sample in sampler.samples])

    summary = {
        "observed_at": iso_timestamp(now_utc()),
        "publisher": publisher_result,
        "persisted_predictions": int(len(frame)),
        "prediction_counter_delta": int(round(predictions_after - predictions_before)),
        "schema_mismatch_delta": int(round(mismatches_after - mismatches_before)),
        "latency_ms": latency_summary,
        "resource_metrics_supported": sampler.supported and bool(sampler.samples),
        "resource_metrics_error": sampler.error,
        "cpu_cores": cpu_summary,
        "memory_mib": memory_summary,
        "resource_samples": len(sampler.samples),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Edge Profiling Report",
        "",
        "## Scope",
        "",
        "This Job publishes a controlled stream of valid synthetic CNC events to the in-cluster MQTT broker and then reads persisted inference rows plus edge metrics from the running K3s deployment.",
        "",
        "## Directly observed",
        "",
        f"- Requested events: `{publisher_result['event_count_requested']}`.",
        f"- Sent events: `{publisher_result['event_count_sent']}`.",
        f"- Persisted inference rows in the observed window: `{summary['persisted_predictions']}`.",
        f"- Prediction counter delta from `/metrics`: `{summary['prediction_counter_delta']}`.",
        f"- Schema mismatch delta from `/metrics`: `{summary['schema_mismatch_delta']}`.",
        f"- Mean latency: `{latency_summary['mean']}` ms.",
        f"- p50/p95/p99 latency: `{latency_summary['p50']}` / `{latency_summary['p95']}` / `{latency_summary['p99']}` ms.",
        f"- Min/max latency: `{latency_summary['min']}` / `{latency_summary['max']}` ms.",
        "",
        "## Resource sampling",
        "",
    ]
    if summary["resource_metrics_supported"]:
        report_lines.extend(
            [
                f"- Resource samples: `{summary['resource_samples']}`.",
                f"- CPU mean/p95/max: `{cpu_summary['mean']}` / `{cpu_summary['p95']}` / `{cpu_summary['max']}` cores.",
                f"- Memory mean/p95/max: `{memory_summary['mean']}` / `{memory_summary['p95']}` / `{memory_summary['max']}` MiB.",
            ]
        )
    else:
        report_lines.extend(
            [
                "- Kubernetes metrics API was not available to the Job.",
                f"- Runtime error: `{summary['resource_metrics_error']}`.",
                "- In this case CPU and memory must be read from Rancher workload graphs during the same execution window.",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- These values are operational observations from the local K3s validation environment, not cross-device performance benchmarks.",
            "- The Job reports persisted inference latency and in-cluster counters. It does not establish hard real-time guarantees.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print("EDGE_PROFILE_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
