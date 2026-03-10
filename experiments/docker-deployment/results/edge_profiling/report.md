# Edge Profiling Report

## Scope

This experiment measures repeated edge inference latency, container CPU usage, container memory usage, prediction count, and schema mismatch count in the local CNC validation stack.

## Minimal execution path

1. Start the minimal edge stack with the synthetic baseline model.
2. Wait until `edge-sync` materializes the deployment state and `edge-inference` exposes its metrics endpoint.
3. Publish a controlled batch of valid synthetic telemetry events over MQTT.
4. Sample `docker stats` for `ind-edge-inference` while the batch is being processed.
5. Query persisted `inference_events` from TimescaleDB and process-local Prometheus metrics from `edge-inference`.

## Directly observed results

- Requested events: `300`
- Sent events: `300`
- Persisted inference events in the experiment window: `300`
- Prediction counter observed at `edge-inference`: `300`
- Schema mismatch counter observed at `edge-inference`: `0`
- Latency summary (ms): mean `19.032088`, p50 `9.713`, p95 `75.6004`, p99 `85.764822`, min `2.6077`, max `133.7507`
- CPU summary (%): mean `293.752857`, p95 `421.759`, max `425.44`
- Memory summary (MiB): mean `127.457143`, p95 `127.74`, max `127.8`

## Caveats

- These observations come from the current local Docker environment and host hardware.
- They are suitable as operational evidence for the instantiated architecture, not as cross-device benchmarks.
- CPU and memory values are container-level observations from `docker stats`, not process-level profiler traces.
- `docker stats` reports aggregate CPU usage across host cores, so container CPU percentages may exceed `100%` on multicore hosts.
- Schema mismatches are counted from the process-local `edge-inference` metrics endpoint after a fresh service start.

## Unsupported by this experiment

- Cross-device transferability claims
- Energy consumption claims
- Hard real-time guarantees during OTA update
- Negative-path security claims beyond the counters collected here

## Commands executed

- `docker compose -f /Users/miguelangel/Documents/Tesis Doctoral/soluciones/MLOPs-Reference-Architecture-/Docker-Cloud-Fog-Edge/docker-compose.yml up -d edge-inference`
- `docker exec ind-edge-sync sh -lc test -f /var/lib/industrial-mlops/edge/deployment_state.json && echo ready`
- `docker exec ind-edge-inference python -c urllib probe localhost:8010/metrics`
- `docker stats --no-stream --format {{json .}} ind-edge-inference`
- `docker exec ind-edge-inference python -c urllib metrics fetch`
- `docker exec ind-timescale psql -U admin -d factory_db -c COPY (...) TO STDOUT WITH CSV HEADER`
- `docker exec ind-edge-sync cat /var/lib/industrial-mlops/edge/deployment_state.json`
