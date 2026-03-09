# Industrial MLOps Reference Architecture

This repository accompanies the article **"Towards a Reference Architecture for Machine Learning Operations"**.

Its primary purpose is to provide an **open-source instantiation** of the proposed industrial MLOps **Reference Architecture**. The CNC machining scenario included here should therefore be read as a **validation case for the architecture**, not as the main contribution of the work.

## What This Repository Contributes

The repository supports the paper at two complementary levels:

1. **Reference Architecture**
   A reusable decomposition of industrial MLOps responsibilities across Cloud, Fog, Edge, Governance, and Enterprise tiers.
2. **Reference Implementation**
   A vendor-neutral stack showing how those responsibilities can be instantiated with Docker or K3s/Rancher using open-source components.

The architectural claims now supported directly by the repository include:

- versioned Airflow DAGs for bootstrap, retraining, and rollback governance
- a closed-loop lifecycle combining drift detection, candidate training, promotion, and rollback
- a CNC tool-breakage validation scenario aligned with the architectural scope of the paper
- a curated company-provided Nakamura 2 reference dataset integrated into the same MLOps backbone
- an explicit Enterprise Tier with supervisory API and web UI
- OTA-style deployment manifests, checksum validation, and signed control messages
- aligned Docker and K3s artefacts representing the same logical architecture

## Architectural Scope

The repository instantiates a hierarchical Cloud-Fog-Edge architecture with explicit governance and enterprise supervision layers:

- **Cloud Tier**
  - Apache Airflow for orchestration
  - MLflow for experiment tracking and model registry
  - Jupyter Lab for reproducible experimentation
- **Fog Tier**
  - TimescaleDB for time-series, inference, and governance persistence
  - MinIO for object storage
  - Fog Bridge for MQTT -> Kafka -> Timescale mediation with spool-based store-and-forward
- **Edge Tier**
  - Mosquitto for industrial telemetry ingestion
  - Kafka for event streaming
  - Node-RED and CNC simulator services for plant-side signal generation
  - Edge Sync and Edge Inference services for controlled deployment and near-machine scoring
- **Enterprise Tier**
  - Enterprise API for promotion, rollback, and supervisory control
  - React UI for operational supervision
  - Rancher-oriented K3s management path for UI-driven cluster operations

## Validation Logic

The repository intentionally maintains two complementary experimental lines:

1. **Synthetic CNC line**
   Used for deterministic end-to-end validation of orchestration, drift handling, retraining, promotion, rollback, and OTA deployment.
2. **Company CNC reference line**
   Used to validate that the same architecture can ingest, register, deploy, and monitor a real 12-signal industrial schema.

This separation is deliberate. The synthetic line supports **reproducibility**, while the company line supports **industrial grounding**. Together, they strengthen the validation of the architecture without overstating cross-scenario generalisability.

The local artefacts also preserve a small set of auditable operational observations from the CNC validation environment, summarised below.

## Observed Validation Snapshot

The following values are directly supported by persisted local artefacts from the CNC validation environment, including Airflow task logs, TimescaleDB records, Prometheus historical metrics, and the persisted edge deployment state. They should be interpreted as **operational evidence of local execution**, not as cross-device benchmarks or evidence of broad transferability.

| Indicator | Observed value | Evidence basis |
| --- | --- | --- |
| Seeded company reference events | 4,848 | Airflow bootstrap logs and persisted Timescale rows |
| Persisted company sensor events | 21 | Timescale `company_sensor_events` and Prometheus ingestion counter |
| Persisted edge predictions | 1 | Timescale `inference_events` and Prometheus prediction counter |
| Observed edge inference latency | 99.6613 ms | Persisted inference record |
| Closed-loop drift score | 0.178127 | Persisted drift reports and Airflow drift logs |
| Closed-loop drift severity / drifted features | low / 1 | Persisted drift reports and Airflow drift logs |
| Retraining decision | `skip_retraining` | Airflow closed-loop routing logs |
| Rollback decision | `skip_rollback` | Airflow governance routing logs |
| Applied deployment generation / deployed model | `1` / `cnc_company_reference_classifier` v`1` | Persisted edge deployment state |
| Mean OTA sync latency | 248.5709 ms | Persisted edge sync status, based on 2 applied sync rows |

These observations are intentionally narrow. The current local evidence does **not** support CPU profiling, memory profiling, energy claims, latency-distribution benchmarking, real rollback execution, or observed store-and-forward disruption episodes.

Accordingly, this snapshot is best read as a compact validation trace for the instantiated CNC scenario, complementing the architecture and reproducibility material documented in the repository.

## Repository Layout

This repository offers two deployment flavours:

### 1. [Docker Development Environment](./Docker-Cloud-Fog-Edge)

Suitable for local replication, experimentation, and reviewer validation through `docker compose`.

- [Docker Guide](./Docker-Cloud-Fog-Edge/README.md)

### 2. [K3s / Kubernetes Environment](./K3s-Cloud-Fog-Edge)

Suitable for a more production-like deployment using logical tier separation and Rancher-managed K3s operations.

- [K3s Guide](./K3s-Cloud-Fog-Edge/README.md)

## Key Repository Areas

- shared industrial logic: `Docker-Cloud-Fog-Edge/apps/industrial_mlops/`
- runtime services: `Docker-Cloud-Fog-Edge/apps/services/`
- curated company CNC dataset: `Docker-Cloud-Fog-Edge/apps/industrial_mlops/data/company_cnc/`
- versioned Airflow DAGs: `Docker-Cloud-Fog-Edge/notebooks/dags/`
- enterprise supervision UI: `Docker-Cloud-Fog-Edge/enterprise-ui/`
- Docker reference stack: `Docker-Cloud-Fog-Edge/docker-compose.yml`
- K3s manifests: `K3s-Cloud-Fog-Edge/k8-manifests/`

## Quick Start

1. Choose your target environment: `Docker-Cloud-Fog-Edge` or `K3s-Cloud-Fog-Edge`.
2. Follow the environment-specific `README.md`.
3. Use the [Experiment Replication Guide](./docs/EXPERIMENT_REPLICATION_GUIDE.md) for the full step-by-step procedure.
4. Use the [Article to Repository Map](./docs/ARTICLE_TO_REPOSITORY_MAP.md) if you need article-to-code traceability.

## Continuity Note

The original project premise remains valid: this is an open reference implementation for distributed industrial AI organised around Cloud, Fog, and Edge tiers. The current version strengthens that baseline with lifecycle governance, enterprise supervision, and a CNC validation scenario aligned with the article's reference-architecture focus.
