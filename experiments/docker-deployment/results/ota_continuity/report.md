# OTA Continuity Report

## Scope

This experiment assessed whether the edge inference path remained operational while a newly registered synthetic model generation was promoted and applied through the existing OTA path.

## Minimal execution path

1. Start the minimal edge stack with the synthetic production model.
2. Start a continuous stream of valid synthetic telemetry events over MQTT.
3. Trigger registration and manual promotion of a new model version while the stream is active.
4. Extract deployment state, edge sync status, Prometheus-format metric snapshots, logs, and persisted inference events.

## Observed continuity evidence

- Previous accepted generation: `2` / version `2`
- New accepted generation: `3` / version `3`
- Predictions associated with the previous generation in observed input order: `288`
- Predictions associated with the new generation in observed input order: `912`
- Generation switches observed in input order: `1`
- First cycle served by the new generation: `289`
- Last cycle served by the previous generation: `288`
- OTA apply latency reported by `edge_sync_status`: `167.19889640808105` ms
- Sync failures observed in metrics: `0`
- Schema mismatches observed in metrics: `0`
- Single clean generation switch observed in the persisted inference sequence: `True`
- New generation observed on persisted inference events after apply: `True`

## Observed downtime or gaps

- Update issue time: `2026-03-09T22:41:25.355873+00:00`
- Update apply time: `2026-03-09T22:41:25.530063+00:00`
- Predictions with payload timestamps before issue time: `562`
- Predictions with payload timestamps in the `issue -> apply` window: `3`
- Predictions with payload timestamps after apply time: `635`
- Boundary gap between the last pre-apply prediction and the first post-apply prediction: `0.203972` seconds
- Maximum consecutive prediction gap before update: `0.061113` seconds
- Maximum consecutive prediction gap over the full experiment: `0.061113` seconds

## Unsupported claims

- No hard real-time guarantee is established by this experiment.
- This does not prove microsecond-level continuity or deterministic scheduling.
- The experiment uses the local Docker deployment on the current host, not a cross-device benchmark.
- The promoted version was created and manually promoted for a controlled update event; this is continuity evidence, not a claim of universal promotion policy adequacy.

## Caveats

- Continuity is assessed from persisted inference timestamps, edge sync status, deployment state, and Prometheus-format counters.
- `inference_events.event_time` comes from the sensor payload, not from the database commit time of the inference result.
- For that reason, the payload-timestamp counts around the `issue -> apply` window are supportive context, not a direct proof of zero processing interruption.
- If no predictions occur in the `issue -> apply` window, the result should be interpreted as an observed short update window, not as proof of zero interruption.
- Failed/rejected sync rows observed in the experiment window: `0`

## Commands executed

- `docker compose -f /Users/miguelangel/Documents/Tesis Doctoral/soluciones/MLOPs-Reference-Architecture-/Docker-Cloud-Fog-Edge/docker-compose.yml up -d edge-inference`
- `docker exec -i ind-edge-sync python - < /Users/miguelangel/Documents/Tesis Doctoral/soluciones/MLOPs-Reference-Architecture-/experiments/publish_synthetic_edge_load.py`
- `docker exec -i ind-edge-sync python - < /Users/miguelangel/Documents/Tesis Doctoral/soluciones/MLOPs-Reference-Architecture-/experiments/promote_synthetic_generation.py`
