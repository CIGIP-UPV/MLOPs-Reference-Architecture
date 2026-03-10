from __future__ import annotations

import json
import os
import time

import numpy as np
import paho.mqtt.client as mqtt

from industrial_mlops.cnc_data import CNCMachineState, generate_stream_event
from industrial_mlops.config import CONFIG
from industrial_mlops.security import sign_payload


def main() -> None:
    event_count = int(os.environ.get("EDGE_PROFILE_EVENT_COUNT", "300"))
    interval_seconds = float(os.environ.get("EDGE_PROFILE_EVENT_INTERVAL", "0.05"))
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
        for _ in range(event_count):
            payload = generate_stream_event(state, rng)
            payload["schema_name"] = "synthetic-cnc"
            payload["signature"] = sign_payload(payload, CONFIG.shared_secret)
            info = client.publish(CONFIG.mqtt_sensor_topic, json.dumps(payload), qos=1)
            info.wait_for_publish()
            sent += 1
            if interval_seconds > 0:
                time.sleep(interval_seconds)
    finally:
        client.loop_stop()

    print(
        json.dumps(
            {
                "event_count_requested": event_count,
                "event_count_sent": sent,
                "interval_seconds": interval_seconds,
                "elapsed_seconds": round(time.time() - started, 6),
                "topic": CONFIG.mqtt_sensor_topic,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
