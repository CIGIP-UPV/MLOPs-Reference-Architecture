from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time

import mlflow
import paho.mqtt.client as mqtt
from prometheus_client import Counter, Gauge, start_http_server

from industrial_mlops.config import CONFIG
from industrial_mlops.db import record_edge_sync_status
from industrial_mlops.deployment import read_deployment_state, write_deployment_state
from industrial_mlops.registry import get_feature_schema_for_version, get_production_version, get_reference_profile_for_version
from industrial_mlops.security import compute_directory_digest, verify_payload

SYNC_COMMANDS = Counter("edge_sync_commands_total", "Deployment commands processed by the edge sync agent")
SYNC_FAILURES = Counter("edge_sync_failures_total", "Deployment commands rejected or failed")
CURRENT_GENERATION = Gauge("edge_sync_generation", "Current edge deployment generation")


class EdgeSyncAgent:
    def __init__(self) -> None:
        self.current_state = read_deployment_state() or {}
        CONFIG.deployment_root.mkdir(parents=True, exist_ok=True)

    def _download_model(self, model_uri: str) -> Path:
        mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
        temp_dir = Path(tempfile.mkdtemp(prefix="edge-sync-"))
        local_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(temp_dir)))
        if local_path.is_file():
            raise RuntimeError("MLflow download returned a file, but a model directory is required.")
        return local_path

    def apply_manifest(self, manifest: dict[str, object]) -> None:
        if not verify_payload(manifest, CONFIG.shared_secret):
            SYNC_FAILURES.inc()
            raise ValueError("Invalid deployment manifest signature.")
        generation = int(manifest["deployment_generation"])
        current_generation = int(self.current_state.get("deployment_generation", 0))
        if generation <= current_generation:
            return
        started = time.time()
        downloaded = self._download_model(str(manifest["model_uri"]))
        checksum = compute_directory_digest(downloaded)
        if manifest.get("checksum") and checksum != manifest["checksum"]:
            SYNC_FAILURES.inc()
            raise ValueError("Model checksum verification failed.")
        target_dir = CONFIG.edge_model_path
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(downloaded, target_dir)
        reference_profile = get_reference_profile_for_version(
            str(manifest["model_version"]),
            model_name=str(manifest["model_name"]),
        )
        feature_schema = get_feature_schema_for_version(str(manifest["model_version"]), model_name=str(manifest["model_name"]))
        self.current_state = {
            "action": manifest["action"],
            "model_name": manifest["model_name"],
            "model_version": str(manifest["model_version"]),
            "model_uri": manifest["model_uri"],
            "checksum": checksum,
            "deployment_generation": generation,
            "source_run_id": manifest.get("source_run_id"),
            "applied_at": manifest["issued_at"],
            "reference_profile": reference_profile,
            "feature_schema": feature_schema,
            "signature": manifest["signature"],
            "local_model_path": str(target_dir),
        }
        write_deployment_state(self.current_state)
        latency_ms = (time.time() - started) * 1000.0
        CURRENT_GENERATION.set(generation)
        SYNC_COMMANDS.inc()
        record_edge_sync_status(
            {
                "deployment_generation": generation,
                "model_version": manifest["model_version"],
                "sync_state": "applied",
                "ota_latency_ms": latency_ms,
                "notes": manifest["action"],
            }
        )

    def bootstrap_current_production(self) -> None:
        production = get_production_version(CONFIG.edge_bootstrap_model_name)
        if not production:
            if self.current_state:
                CURRENT_GENERATION.set(int(self.current_state.get("deployment_generation", 0)))
            return
        expected_features = list(self.current_state.get("feature_schema", {}).get("features", [])) if self.current_state else []
        reference_profile = self.current_state.get("reference_profile", {}) if self.current_state else {}
        state_matches_production = bool(
            self.current_state
            and str(self.current_state.get("model_name")) == CONFIG.edge_bootstrap_model_name
            and str(self.current_state.get("model_version")) == str(production["version"])
            and (not production.get("tags", {}).get("sha256") or self.current_state.get("checksum") == production["tags"]["sha256"])
            and all(feature in reference_profile for feature in expected_features)
        )
        if state_matches_production:
            CURRENT_GENERATION.set(int(self.current_state.get("deployment_generation", 0)))
            return
        self.current_state = {}
        manifest = {
            "action": "promote",
            "model_name": CONFIG.edge_bootstrap_model_name,
            "model_version": production["version"],
            "model_uri": f"models:/{CONFIG.edge_bootstrap_model_name}/{production['version']}",
            "source_run_id": production.get("run_id"),
            "deployment_generation": 1,
            "checksum": production.get("tags", {}).get("sha256"),
            "previous_version": None,
            "target_stage": "Production",
            "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_by": "edge-sync-bootstrap",
            "reason": "Bootstrap existing production model",
        }
        from industrial_mlops.security import sign_payload

        manifest["signature"] = sign_payload(manifest, CONFIG.shared_secret)
        self.apply_manifest(manifest)


agent = EdgeSyncAgent()


def on_message(_client: mqtt.Client, _userdata: None, message: mqtt.MQTTMessage) -> None:
    manifest = json.loads(message.payload.decode("utf-8"))
    try:
        agent.apply_manifest(manifest)
        heartbeat = {
            "machine_id": CONFIG.edge_machine_id,
            "deployment_generation": agent.current_state["deployment_generation"],
            "model_version": agent.current_state["model_version"],
            "sync_state": "applied",
        }
        _client.publish(CONFIG.mqtt_heartbeat_topic, json.dumps(heartbeat), qos=1)
    except Exception as exc:
        SYNC_FAILURES.inc()
        record_edge_sync_status(
            {
                "deployment_generation": int(manifest.get("deployment_generation", 0)),
                "model_version": str(manifest.get("model_version", "unknown")),
                "sync_state": "failed",
                "notes": str(exc),
            }
        )


def main() -> None:
    start_http_server(CONFIG.monitoring_port_edge_sync)
    agent.bootstrap_current_production()
    client = mqtt.Client()
    if CONFIG.mqtt_username:
        client.username_pw_set(CONFIG.mqtt_username, CONFIG.mqtt_password)
    client.on_message = on_message
    client.connect(CONFIG.mqtt_host, CONFIG.mqtt_port, 60)
    client.subscribe(CONFIG.mqtt_control_topic, qos=1)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
