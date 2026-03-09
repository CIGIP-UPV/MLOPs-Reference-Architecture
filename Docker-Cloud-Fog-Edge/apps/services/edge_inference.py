from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import time
from typing import Any

import mlflow.sklearn
import pandas as pd
import paho.mqtt.client as mqtt
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from industrial_mlops.config import CONFIG
from industrial_mlops.db import insert_inference_event
from industrial_mlops.deployment import read_deployment_state
from industrial_mlops.security import sign_payload, verify_payload

PREDICTIONS = Counter("edge_inference_predictions_total", "Predictions served at the edge")
CURRENT_DRIFT = Gauge("edge_inference_drift_score", "Current event-level drift score")
CURRENT_VERSION = Gauge("edge_inference_model_version", "Production model version loaded at the edge")
INFERENCE_LATENCY = Histogram("edge_inference_latency_ms", "Inference latency in milliseconds", buckets=(5, 10, 25, 50, 100, 200, 500))
SCHEMA_MISMATCHES = Counter("edge_inference_schema_mismatch_total", "Events ignored because they do not match the deployed model schema")


class EdgeInferenceService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.state: dict[str, Any] | None = None
        self.loaded_generation = 0
        self.history = deque(maxlen=128)

    def expected_features(self) -> list[str]:
        if not self.state:
            return []
        schema = self.state.get("feature_schema", {})
        return list(schema.get("features", []))

    def schema_name(self) -> str:
        if not self.state:
            return "unknown"
        schema = self.state.get("feature_schema", {})
        return str(schema.get("dataset_name", "unknown"))

    def ensure_model_loaded(self) -> None:
        state = read_deployment_state()
        if not state:
            return
        generation = int(state["deployment_generation"])
        if self.model is not None and generation == self.loaded_generation and state == self.state:
            return
        model_path = Path(state["local_model_path"])
        self.model = mlflow.sklearn.load_model(str(model_path))
        self.state = state
        self.loaded_generation = generation
        CURRENT_VERSION.set(float(str(state["model_version"])))

    def event_drift_score(self, payload: dict[str, Any]) -> float:
        if not self.state:
            return 0.0
        profile = self.state["reference_profile"]
        shifts = []
        for feature in self.expected_features():
            if feature not in profile or feature not in payload:
                continue
            mean = float(profile[feature]["mean"])
            std = max(float(profile[feature]["std"]), 1e-6)
            shifts.append(min(abs(float(payload[feature]) - mean) / std / 3.0, 1.0))
        return float(sum(shifts) / max(len(shifts), 1))

    def infer(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.ensure_model_loaded()
        if self.model is None or self.state is None:
            return None
        features = self.expected_features()
        missing = [feature for feature in features if feature not in payload]
        if missing:
            SCHEMA_MISMATCHES.inc()
            return None
        start = time.perf_counter()
        frame = pd.DataFrame([{feature: float(payload[feature]) for feature in features}])
        risk_score = float(self.model.predict_proba(frame)[0][1])
        prediction = int(risk_score >= 0.5)
        drift_score = self.event_drift_score(payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        INFERENCE_LATENCY.observe(latency_ms)
        CURRENT_DRIFT.set(drift_score)
        PREDICTIONS.inc()
        result = {
            "event_time": payload["event_time"],
            "machine_id": payload["machine_id"],
            "cycle_id": int(payload["cycle_id"]),
            "deployment_generation": int(self.state["deployment_generation"]),
            "model_name": self.state["model_name"],
            "model_version": str(self.state["model_version"]),
            "prediction": prediction,
            "risk_score": round(risk_score, 6),
            "actual_breakage": int(payload["actual_breakage"]),
            "drift_score": round(drift_score, 6),
            "latency_ms": round(latency_ms, 4),
            "outcome": "alert" if prediction else "normal",
            "command_generation": int(self.state["deployment_generation"]),
            "sync_state": "accepted",
            "sensor_snapshot": {feature: payload[feature] for feature in features},
            "schema_name": self.schema_name(),
        }
        insert_inference_event(result)
        result["signature"] = sign_payload(result, CONFIG.shared_secret)
        return result


service = EdgeInferenceService()


def on_message(client: mqtt.Client, _userdata: None, message: mqtt.MQTTMessage) -> None:
    payload = json.loads(message.payload.decode("utf-8"))
    if not verify_payload(payload, CONFIG.shared_secret):
        return
    result = service.infer(payload)
    if result is not None:
        topic = CONFIG.mqtt_company_prediction_topic if result.get("schema_name") == "company-nakamura-reference" else CONFIG.mqtt_prediction_topic
        client.publish(topic, json.dumps(result), qos=1)


def main() -> None:
    start_http_server(CONFIG.monitoring_port_edge_inference)
    client = mqtt.Client()
    if CONFIG.mqtt_username:
        client.username_pw_set(CONFIG.mqtt_username, CONFIG.mqtt_password)
    client.on_message = on_message
    client.connect(CONFIG.mqtt_host, CONFIG.mqtt_port, 60)
    client.subscribe(CONFIG.mqtt_sensor_topic, qos=1)
    client.subscribe(CONFIG.mqtt_company_sensor_topic, qos=1)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
