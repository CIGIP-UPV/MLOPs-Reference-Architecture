from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "Docker-Cloud-Fog-Edge" / "docker-compose.yml"
RESULT_DIR = ROOT / "experiments" / "results" / "edge_profiling"
PUBLISHER_SCRIPT = ROOT / "experiments" / "publish_synthetic_edge_load.py"

PROJECT_ENV = {
    "SIMULATOR_PROFILE": "synthetic",
    "EDGE_BOOTSTRAP_MODEL_NAME": "cnc_tool_breakage_classifier",
    "MLFLOW_HOST_PORT": "5500",
}

EVENT_COUNT = int(os.environ.get("EDGE_PROFILE_EVENT_COUNT", "300"))
EVENT_INTERVAL = float(os.environ.get("EDGE_PROFILE_EVENT_INTERVAL", "0.05"))
STATS_SAMPLE_SECONDS = float(os.environ.get("EDGE_PROFILE_STATS_INTERVAL", "1.0"))
EDGE_CONTAINER = "ind-edge-inference"
SYNC_CONTAINER = "ind-edge-sync"
DB_CONTAINER = "ind-timescale"
DEPLOYMENT_STATE_PATH = "/var/lib/industrial-mlops/edge/deployment_state.json"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def run(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(ROOT),
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def docker_compose_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PROJECT_ENV)
    return env


def run_compose(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["docker", "compose", "-f", str(COMPOSE_FILE), *args], check=check, env=docker_compose_env())


def run_psql(query: str) -> str:
    cmd = [
        "docker",
        "exec",
        DB_CONTAINER,
        "psql",
        "-U",
        "admin",
        "-d",
        "factory_db",
        "-At",
        "-F",
        "\t",
        "-c",
        query,
    ]
    return run(cmd).stdout


def wait_for(condition, timeout_seconds: float, label: str) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            if condition():
                return
        except Exception as exc:  # pragma: no cover - defensive wrapper
            last_error = str(exc)
        time.sleep(1)
    detail = f" Last error: {last_error}" if last_error else ""
    raise TimeoutError(f"Timed out waiting for {label}.{detail}")


def deployment_state_ready() -> bool:
    result = run(
        ["docker", "exec", SYNC_CONTAINER, "sh", "-lc", f"test -f {DEPLOYMENT_STATE_PATH} && echo ready"],
        check=False,
    )
    return result.returncode == 0 and "ready" in result.stdout


def edge_metrics_ready() -> bool:
    probe = (
        "import urllib.request; "
        "print(urllib.request.urlopen('http://localhost:8010/metrics', timeout=2).status)"
    )
    result = run(["docker", "exec", EDGE_CONTAINER, "python", "-c", probe], check=False)
    return result.returncode == 0 and "200" in result.stdout


def query_recent_inference_count(started_at: datetime, ended_at: datetime) -> int:
    start = iso_timestamp(started_at - timedelta(seconds=2))
    end = iso_timestamp(ended_at + timedelta(seconds=2))
    output = run_psql(
        "SELECT COUNT(*) FROM inference_events "
        f"WHERE event_time >= TIMESTAMPTZ '{start}' AND event_time <= TIMESTAMPTZ '{end}';"
    ).strip()
    return int(output or "0")


def wait_for_inference_stability(started_at: datetime) -> int:
    last = None
    stable_reads = 0
    for _ in range(20):
        current = query_recent_inference_count(started_at, now_utc())
        if current == last:
            stable_reads += 1
            if stable_reads >= 3:
                return current
        else:
            stable_reads = 0
        last = current
        time.sleep(2)
    return int(last or 0)


def parse_percent(raw: str) -> float:
    return float(raw.strip().replace("%", ""))


def parse_mem_usage(raw: str) -> tuple[float, float]:
    usage, limit = [part.strip() for part in raw.split("/", 1)]
    return parse_size_to_mib(usage), parse_size_to_mib(limit)


def parse_size_to_mib(raw: str) -> float:
    match = re.match(r"([0-9.]+)([KMG]i?)?B?$", raw.strip())
    if not match:
        raise ValueError(f"Unsupported memory size format: {raw}")
    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    factors = {
        "": 1.0 / (1024.0 * 1024.0),
        "k": 1.0 / 1024.0,
        "ki": 1.0 / 1024.0,
        "m": 1.0,
        "mi": 1.0,
        "g": 1024.0,
        "gi": 1024.0,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported memory unit: {raw}")
    return value * factors[unit]


def sample_docker_stats() -> dict[str, Any]:
    fmt = "{{json .}}"
    result = run(["docker", "stats", "--no-stream", "--format", fmt, EDGE_CONTAINER])
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    mem_usage_mib, mem_limit_mib = parse_mem_usage(payload["MemUsage"])
    return {
        "record_type": "docker_stats",
        "observed_at": iso_timestamp(now_utc()),
        "event_time": "",
        "cycle_id": "",
        "latency_ms": "",
        "prediction": "",
        "actual_breakage": "",
        "drift_score": "",
        "model_name": "",
        "model_version": "",
        "cpu_pct": parse_percent(payload["CPUPerc"]),
        "mem_usage_mib": round(mem_usage_mib, 6),
        "mem_limit_mib": round(mem_limit_mib, 6),
        "mem_pct": parse_percent(payload["MemPerc"]),
    }


def run_publisher() -> subprocess.Popen[str]:
    env_flags = [
        "-e",
        f"EDGE_PROFILE_EVENT_COUNT={EVENT_COUNT}",
        "-e",
        f"EDGE_PROFILE_EVENT_INTERVAL={EVENT_INTERVAL}",
    ]
    command = ["docker", "exec", *env_flags, "-i", SYNC_CONTAINER, "python", "-"]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )


def fetch_edge_metrics_text() -> str:
    probe = (
        "import urllib.request; "
        "print(urllib.request.urlopen('http://localhost:8010/metrics', timeout=5).read().decode('utf-8'))"
    )
    return run(["docker", "exec", EDGE_CONTAINER, "python", "-c", probe]).stdout


def extract_prometheus_metric(metrics_text: str, name: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{.*\}})?\s+([0-9.eE+-]+)$", re.MULTILINE)
    matches = pattern.findall(metrics_text)
    if not matches:
        return None
    return float(matches[-1])


def copy_query_to_rows(started_at: datetime, ended_at: datetime) -> list[dict[str, Any]]:
    start = iso_timestamp(started_at - timedelta(seconds=2))
    end = iso_timestamp(ended_at + timedelta(seconds=2))
    sql = (
        "COPY ("
        "SELECT event_time, cycle_id, latency_ms, prediction, actual_breakage, drift_score, model_name, model_version "
        "FROM inference_events "
        f"WHERE event_time >= TIMESTAMPTZ '{start}' AND event_time <= TIMESTAMPTZ '{end}' "
        "ORDER BY event_time"
        ") TO STDOUT WITH CSV HEADER"
    )
    result = run(
        ["docker", "exec", DB_CONTAINER, "psql", "-U", "admin", "-d", "factory_db", "-c", sql]
    ).stdout
    reader = csv.DictReader(result.splitlines())
    rows: list[dict[str, Any]] = []
    for row in reader:
        rows.append(
            {
                "record_type": "inference_event",
                "observed_at": "",
                "event_time": row["event_time"],
                "cycle_id": int(row["cycle_id"]),
                "latency_ms": float(row["latency_ms"]),
                "prediction": int(row["prediction"]),
                "actual_breakage": int(row["actual_breakage"]),
                "drift_score": float(row["drift_score"]),
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "cpu_pct": "",
                "mem_usage_mib": "",
                "mem_limit_mib": "",
                "mem_pct": "",
            }
        )
    return rows


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("No values available for percentile calculation.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summary_stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": round(percentile(ordered, 50), 6),
        "p95": round(percentile(ordered, 95), 6),
        "p99": round(percentile(ordered, 99), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def load_deployment_state() -> dict[str, Any]:
    return json.loads(
        run(["docker", "exec", SYNC_CONTAINER, "cat", DEPLOYMENT_STATE_PATH]).stdout
    )


def write_metrics_csv(rows: list[dict[str, Any]]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULT_DIR / "metrics.csv"
    fieldnames = [
        "record_type",
        "observed_at",
        "event_time",
        "cycle_id",
        "latency_ms",
        "prediction",
        "actual_breakage",
        "drift_score",
        "model_name",
        "model_version",
        "cpu_pct",
        "mem_usage_mib",
        "mem_limit_mib",
        "mem_pct",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_summary_json(payload: dict[str, Any]) -> Path:
    destination = RESULT_DIR / "summary.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def write_report_md(payload: dict[str, Any], commands: list[str]) -> Path:
    destination = RESULT_DIR / "report.md"
    lines = [
        "# Edge Profiling Report",
        "",
        "## Scope",
        "",
        "This experiment measures repeated edge inference latency, container CPU usage, container memory usage, prediction count, and schema mismatch count in the local CNC validation stack.",
        "",
        "## Minimal execution path",
        "",
        "1. Start the minimal edge stack with the synthetic baseline model.",
        "2. Wait until `edge-sync` materializes the deployment state and `edge-inference` exposes its metrics endpoint.",
        "3. Publish a controlled batch of valid synthetic telemetry events over MQTT.",
        "4. Sample `docker stats` for `ind-edge-inference` while the batch is being processed.",
        "5. Query persisted `inference_events` from TimescaleDB and process-local Prometheus metrics from `edge-inference`.",
        "",
        "## Directly observed results",
        "",
        f"- Requested events: `{payload['experiment']['event_count_requested']}`",
        f"- Sent events: `{payload['experiment']['publisher']['event_count_sent']}`",
        f"- Persisted inference events in the experiment window: `{payload['direct_observations']['persisted_inference_events']}`",
        f"- Prediction counter observed at `edge-inference`: `{payload['direct_observations']['prediction_count_metric']}`",
        f"- Schema mismatch counter observed at `edge-inference`: `{payload['direct_observations']['schema_mismatch_count']}`",
        f"- Latency summary (ms): mean `{payload['latency_ms']['mean']}`, p50 `{payload['latency_ms']['p50']}`, p95 `{payload['latency_ms']['p95']}`, p99 `{payload['latency_ms']['p99']}`, min `{payload['latency_ms']['min']}`, max `{payload['latency_ms']['max']}`",
        f"- CPU summary (%): mean `{payload['cpu_pct']['mean']}`, p95 `{payload['cpu_pct']['p95']}`, max `{payload['cpu_pct']['max']}`",
        f"- Memory summary (MiB): mean `{payload['memory_usage_mib']['mean']}`, p95 `{payload['memory_usage_mib']['p95']}`, max `{payload['memory_usage_mib']['max']}`",
        "",
        "## Caveats",
        "",
        "- These observations come from the current local Docker environment and host hardware.",
        "- They are suitable as operational evidence for the instantiated architecture, not as cross-device benchmarks.",
        "- CPU and memory values are container-level observations from `docker stats`, not process-level profiler traces.",
        "- `docker stats` reports aggregate CPU usage across host cores, so container CPU percentages may exceed `100%` on multicore hosts.",
        "- Schema mismatches are counted from the process-local `edge-inference` metrics endpoint after a fresh service start.",
        "",
        "## Unsupported by this experiment",
        "",
        "- Cross-device transferability claims",
        "- Energy consumption claims",
        "- Hard real-time guarantees during OTA update",
        "- Negative-path security claims beyond the counters collected here",
        "",
        "## Commands executed",
        "",
    ]
    lines.extend(f"- `{command}`" for command in commands)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    commands: list[str] = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def log_command(cmd: list[str]) -> None:
        commands.append(" ".join(cmd))

    try:
        up_cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "edge-inference"]
        log_command(up_cmd)
        run_compose(["up", "-d", "edge-inference"])

        log_command(["docker", "exec", SYNC_CONTAINER, "sh", "-lc", f"test -f {DEPLOYMENT_STATE_PATH} && echo ready"])
        wait_for(deployment_state_ready, timeout_seconds=180, label="edge deployment state")

        log_command(["docker", "exec", EDGE_CONTAINER, "python", "-c", "urllib probe localhost:8010/metrics"])
        wait_for(edge_metrics_ready, timeout_seconds=120, label="edge metrics endpoint")

        started_at = now_utc()
        publisher = run_publisher()
        publisher_script = PUBLISHER_SCRIPT.read_text(encoding="utf-8")
        assert publisher.stdin is not None
        publisher.stdin.write(publisher_script)
        publisher.stdin.close()

        docker_samples: list[dict[str, Any]] = []
        stats_cmd = ["docker", "stats", "--no-stream", "--format", "{{json .}}", EDGE_CONTAINER]
        log_command(stats_cmd)
        while publisher.poll() is None:
            docker_samples.append(sample_docker_stats())
            time.sleep(STATS_SAMPLE_SECONDS)

        publisher.wait(timeout=60)
        stdout = publisher.stdout.read() if publisher.stdout is not None else ""
        stderr = publisher.stderr.read() if publisher.stderr is not None else ""
        if publisher.returncode != 0:
            raise RuntimeError(f"Publisher failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        publisher_summary = json.loads(stdout.strip().splitlines()[-1])

        time.sleep(3)
        persisted_count = wait_for_inference_stability(started_at)
        ended_at = now_utc()

        metrics_cmd = ["docker", "exec", EDGE_CONTAINER, "python", "-c", "urllib metrics fetch"]
        log_command(metrics_cmd)
        metrics_text = fetch_edge_metrics_text()

        db_cmd = [
            "docker",
            "exec",
            DB_CONTAINER,
            "psql",
            "-U",
            "admin",
            "-d",
            "factory_db",
            "-c",
            "COPY (...) TO STDOUT WITH CSV HEADER",
        ]
        log_command(db_cmd)
        inference_rows = copy_query_to_rows(started_at, ended_at)

        state_cmd = ["docker", "exec", SYNC_CONTAINER, "cat", DEPLOYMENT_STATE_PATH]
        log_command(state_cmd)
        deployment_state = load_deployment_state()

        prediction_count_metric = extract_prometheus_metric(metrics_text, "edge_inference_predictions_total")
        schema_mismatch_count = extract_prometheus_metric(metrics_text, "edge_inference_schema_mismatch_total")

        latency_values = [float(row["latency_ms"]) for row in inference_rows]
        if not latency_values:
            raise RuntimeError("No inference events were persisted in the experiment window.")

        cpu_values = [float(sample["cpu_pct"]) for sample in docker_samples]
        mem_usage_values = [float(sample["mem_usage_mib"]) for sample in docker_samples]
        mem_pct_values = [float(sample["mem_pct"]) for sample in docker_samples]
        if not cpu_values or not mem_usage_values or not mem_pct_values:
            raise RuntimeError("No docker stats samples were collected.")

        rows = docker_samples + inference_rows
        write_metrics_csv(rows)

        summary = {
            "experiment": {
                "started_at": iso_timestamp(started_at),
                "ended_at": iso_timestamp(ended_at),
                "event_count_requested": EVENT_COUNT,
                "event_interval_seconds": EVENT_INTERVAL,
                "stats_sample_seconds": STATS_SAMPLE_SECONDS,
                "publisher": publisher_summary,
            },
            "deployment_state": {
                "model_name": deployment_state.get("model_name"),
                "model_version": deployment_state.get("model_version"),
                "deployment_generation": deployment_state.get("deployment_generation"),
                "schema_name": deployment_state.get("feature_schema", {}).get("dataset_name"),
                "feature_count": len(deployment_state.get("feature_schema", {}).get("features", [])),
            },
            "direct_observations": {
                "persisted_inference_events": persisted_count,
                "prediction_count_metric": int(prediction_count_metric) if prediction_count_metric is not None else None,
                "schema_mismatch_count": int(schema_mismatch_count) if schema_mismatch_count is not None else None,
                "metrics_source": "edge-inference process-local /metrics endpoint",
                "latency_source": "factory_db.inference_events.latency_ms",
                "resource_source": "docker stats ind-edge-inference",
            },
            "latency_ms": summary_stats(latency_values),
            "cpu_pct": summary_stats(cpu_values),
            "memory_usage_mib": summary_stats(mem_usage_values),
            "memory_pct": summary_stats(mem_pct_values),
            "unsupported": [
                "cross-device benchmarks",
                "energy consumption",
                "hard real-time guarantees",
                "process-level CPU profiler traces",
            ],
        }
        write_summary_json(summary)
        write_report_md(summary, commands)
        return 0
    finally:
        down_cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "stop", "edge-inference", "edge-sync", "mosquitto", "db-bootstrap", "mlflow", "create_buckets", "minio", "timescale"]
        try:
            log_command(down_cmd)
            run_compose(["stop", "edge-inference", "edge-sync", "mosquitto", "db-bootstrap", "mlflow", "create_buckets", "minio", "timescale"], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
