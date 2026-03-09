from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt

from .config import CONFIG
from .security import atomic_write_json, sign_payload


def build_manifest(
    *,
    action: str,
    model_name: str | None,
    model_version: str,
    checksum: str | None,
    generation: int,
    source_run_id: str | None,
    reason: str,
    issued_by: str,
    previous_version: str | None = None,
) -> dict[str, Any]:
    model_name = model_name or CONFIG.model_name
    manifest = {
        "action": action,
        "model_name": model_name,
        "model_version": str(model_version),
        "model_uri": f"models:/{model_name}/{model_version}",
        "source_run_id": source_run_id,
        "deployment_generation": int(generation),
        "checksum": checksum,
        "previous_version": str(previous_version) if previous_version else None,
        "target_stage": "Production",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "issued_by": issued_by,
        "reason": reason,
    }
    manifest["signature"] = sign_payload(manifest, CONFIG.shared_secret)
    return manifest


def read_deployment_state(path: Path | None = None) -> dict[str, Any] | None:
    state_path = path or CONFIG.edge_state_path
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_deployment_state(payload: dict[str, Any], path: Path | None = None) -> None:
    atomic_write_json(path or CONFIG.edge_state_path, payload)


def publish_control_command(manifest: dict[str, Any]) -> None:
    client = mqtt.Client()
    if CONFIG.mqtt_username:
        client.username_pw_set(CONFIG.mqtt_username, CONFIG.mqtt_password)
    client.connect(CONFIG.mqtt_host, CONFIG.mqtt_port, 60)
    client.publish(CONFIG.mqtt_control_topic, json.dumps(manifest), qos=1, retain=True)
    client.disconnect()
