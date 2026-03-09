from __future__ import annotations

import json
import os
import time

import numpy as np
import paho.mqtt.client as mqtt
from prometheus_client import Counter, Gauge, start_http_server

from industrial_mlops.cnc_data import CNCMachineState, generate_stream_event
from industrial_mlops.company_dataset import build_company_stream_payload, load_company_reference_dataset
from industrial_mlops.config import CONFIG
from industrial_mlops.security import sign_payload

PUBLISHED_EVENTS = Counter("cnc_simulator_events_total", "Total CNC telemetry events published")
CURRENT_TOOL_WEAR = Gauge("cnc_simulator_tool_wear", "Current simulated tool wear")
BREAKAGE_PROBABILITY = Gauge("cnc_simulator_breakage_probability", "Current breakage probability")
SECONDS_TO_TARGET = Gauge("cnc_simulator_seconds_to_target_alarm", "Seconds to the target alarm for the company replay profile")
CURRENT_PROFILE = Gauge("cnc_simulator_profile", "Simulator profile: 0 synthetic, 1 company")


def main() -> None:
    interval = float(os.environ.get("SIM_INTERVAL_SECONDS", "2.0"))
    profile = os.environ.get("SIMULATOR_PROFILE", CONFIG.simulator_profile).strip().lower()
    rng = np.random.default_rng(42)
    state = CNCMachineState(machine_id=CONFIG.edge_machine_id, cell_id=CONFIG.edge_cell_id)
    company_frame = load_company_reference_dataset() if profile == "company" else None
    company_index = 0
    start_http_server(CONFIG.monitoring_port_cnc_simulator)
    client = mqtt.Client()
    if CONFIG.mqtt_username:
        client.username_pw_set(CONFIG.mqtt_username, CONFIG.mqtt_password)
    client.connect(CONFIG.mqtt_host, CONFIG.mqtt_port, 60)
    while True:
        if profile == "company":
            if company_frame is None or company_frame.empty:
                raise RuntimeError("Company simulator profile requested but no curated company dataset is available.")
            row = company_frame.iloc[company_index % len(company_frame)]
            company_index += 1
            payload = build_company_stream_payload(
                row,
                machine_id=CONFIG.edge_machine_id,
                cell_id=CONFIG.edge_cell_id,
                cycle_id=company_index,
                tool_id=1 + int(row["actual_breakage"]),
                event_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            seconds_to_target = payload.get("seconds_to_target_alarm")
            seconds_to_target = float(seconds_to_target) if seconds_to_target not in (None, "") else None
            derived_probability = 0.05
            if int(payload["actual_breakage"]) == 1 and seconds_to_target is not None:
                derived_probability = max(0.55, 1.0 - min(seconds_to_target, 120.0) / 240.0)
            payload["breakage_probability"] = round(derived_probability, 6)
            CURRENT_TOOL_WEAR.set(0.0)
            BREAKAGE_PROBABILITY.set(float(payload["breakage_probability"]))
            SECONDS_TO_TARGET.set(seconds_to_target if seconds_to_target is not None else -1.0)
            CURRENT_PROFILE.set(1.0)
            topic = CONFIG.mqtt_company_sensor_topic
        else:
            payload = generate_stream_event(state, rng)
            payload["schema_name"] = "synthetic-cnc"
            CURRENT_TOOL_WEAR.set(float(payload["tool_wear"]))
            BREAKAGE_PROBABILITY.set(float(payload["breakage_probability"]))
            SECONDS_TO_TARGET.set(-1.0)
            CURRENT_PROFILE.set(0.0)
            topic = CONFIG.mqtt_sensor_topic
        payload["signature"] = sign_payload(payload, CONFIG.shared_secret)
        client.publish(topic, json.dumps(payload), qos=1)
        PUBLISHED_EVENTS.inc()
        print(json.dumps(payload, sort_keys=True))
        time.sleep(interval)


if __name__ == "__main__":
    main()
