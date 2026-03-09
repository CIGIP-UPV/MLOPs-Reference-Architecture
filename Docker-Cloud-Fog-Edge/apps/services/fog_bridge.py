from __future__ import annotations

import json
from pathlib import Path
import time
from uuid import uuid4

from kafka import KafkaProducer
import paho.mqtt.client as mqtt
from prometheus_client import Counter, Gauge, start_http_server

from industrial_mlops.config import CONFIG
from industrial_mlops.db import insert_sensor_event
from industrial_mlops.security import verify_payload

INGESTED_EVENTS = Counter("fog_bridge_ingested_events_total", "Events ingested into fog persistence and Kafka")
SPOOLED_EVENTS = Counter("fog_bridge_spooled_events_total", "Events temporarily stored for later replay")
INVALID_SIGNATURES = Counter("fog_bridge_invalid_signature_total", "Discarded events with invalid signatures")
SPOOL_QUEUE_SIZE = Gauge("fog_bridge_spool_queue_size", "Current spool size")


class FogBridge:
    def __init__(self) -> None:
        self.producer: KafkaProducer | None = None
        self.spool_dir = CONFIG.spool_root / "fog-bridge"
        self.spool_dir.mkdir(parents=True, exist_ok=True)

    def kafka_producer(self) -> KafkaProducer:
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=CONFIG.kafka_bootstrap_servers,
                value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
            )
        return self.producer

    def persist(self, payload: dict[str, object]) -> None:
        insert_sensor_event(payload)
        producer = self.kafka_producer()
        producer.send(CONFIG.kafka_sensor_topic, payload)
        producer.flush(timeout=5)
        INGESTED_EVENTS.inc()

    def spool(self, payload: dict[str, object]) -> None:
        file_path = self.spool_dir / f"{uuid4().hex}.json"
        file_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        SPOOLED_EVENTS.inc()
        SPOOL_QUEUE_SIZE.set(len(list(self.spool_dir.glob("*.json"))))

    def flush_spool(self) -> None:
        for file_path in sorted(self.spool_dir.glob("*.json")):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            try:
                self.persist(payload)
                file_path.unlink()
            except Exception:
                break
        SPOOL_QUEUE_SIZE.set(len(list(self.spool_dir.glob("*.json"))))


bridge = FogBridge()


def on_message(_client: mqtt.Client, _userdata: None, message: mqtt.MQTTMessage) -> None:
    payload = json.loads(message.payload.decode("utf-8"))
    if not verify_payload(payload, CONFIG.shared_secret):
        INVALID_SIGNATURES.inc()
        return
    try:
        bridge.persist(payload)
        bridge.flush_spool()
    except Exception as exc:
        payload["bridge_error"] = str(exc)
        bridge.spool(payload)


def main() -> None:
    start_http_server(CONFIG.monitoring_port_fog_bridge)
    client = mqtt.Client()
    if CONFIG.mqtt_username:
        client.username_pw_set(CONFIG.mqtt_username, CONFIG.mqtt_password)
    client.on_message = on_message
    client.connect(CONFIG.mqtt_host, CONFIG.mqtt_port, 60)
    client.subscribe(CONFIG.mqtt_sensor_topic, qos=1)
    client.subscribe(CONFIG.mqtt_company_sensor_topic, qos=1)
    bridge.flush_spool()
    print(
        f"Fog bridge subscribed to {CONFIG.mqtt_sensor_topic} and "
        f"{CONFIG.mqtt_company_sensor_topic} -> {CONFIG.kafka_sensor_topic}"
    )
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
