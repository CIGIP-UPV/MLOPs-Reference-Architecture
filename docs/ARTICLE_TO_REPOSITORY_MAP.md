# Article to Repository Map

This document links the main architectural claims of the article **"Towards a Reference Architecture for Machine Learning Operations"** to concrete evidence in the repository. The intent is not to enumerate every file, but to show how each manuscript concern is instantiated in the open-source artefact.

## Reading Guide

The article should be interpreted in three layers:

1. **Evidence base**
   The PRISMA-guided review and requirement synthesis.
2. **Reference Architecture**
   The Cloud-Fog-Edge decomposition, lifecycle control points, and stakeholder-oriented responsibilities.
3. **Architectural instantiation**
   The Docker and K3s/Rancher artefacts, validated through the CNC use case.

The CNC scenario is therefore mapped below as a **validation instantiation of the architecture**, not as the primary scientific contribution.

## Architecture-to-Repository Evidence Map

| Manuscript concern | Architectural meaning in the paper | Repository evidence |
| --- | --- | --- |
| Hybrid Cloud-Fog-Edge Reference Architecture | Separation of industrial MLOps responsibilities across cloud orchestration, fog mediation, and edge execution | `README.md`, `Docker-Cloud-Fog-Edge/docker-compose.yml`, `K3s-Cloud-Fog-Edge/k8-manifests/` |
| Cloud Tier orchestration and registry | Central training, tracking, registry, and workflow control | `Docker-Cloud-Fog-Edge/notebooks/dags/cnc_bootstrap_reference_architecture_v1.py`, `Docker-Cloud-Fog-Edge/notebooks/dags/cnc_closed_loop_retraining_v1.py`, `Docker-Cloud-Fog-Edge/notebooks/dags/cnc_governance_rollback_v1.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/registry.py`, `K3s-Cloud-Fog-Edge/k8-manifests/02-cloud-tier/` |
| Fog Tier mediation and persistence | On-premises persistence, buffering, and data-plane decoupling between plant and cloud | `Docker-Cloud-Fog-Edge/apps/services/fog_bridge.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/db.py`, `Docker-Cloud-Fog-Edge/config/initdb.sql`, `K3s-Cloud-Fog-Edge/k8-manifests/01-fog-tier/` |
| Edge Tier inference and plant proximity | Near-machine scoring, controlled deployment, and local continuity under distributed constraints | `Docker-Cloud-Fog-Edge/apps/services/cnc_simulator.py`, `Docker-Cloud-Fog-Edge/apps/services/edge_inference.py`, `Docker-Cloud-Fog-Edge/apps/services/edge_sync.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/deployment.py`, `K3s-Cloud-Fog-Edge/k8-manifests/03-edge-tier/` |
| Governance and observability | Monitoring-driven lifecycle control, drift awareness, and post-deployment supervision | `Docker-Cloud-Fog-Edge/apps/industrial_mlops/drift.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/monitoring.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/orchestration.py`, `Docker-Cloud-Fog-Edge/config/prometheus.yml`, `Docker-Cloud-Fog-Edge/config/grafana/`, `K3s-Cloud-Fog-Edge/k8-manifests/04-governance/` |
| Enterprise Tier and supervisory control | Explicit user-facing supervision, promotion/rollback control, and operations-oriented governance | `Docker-Cloud-Fog-Edge/apps/services/enterprise_api.py`, `Docker-Cloud-Fog-Edge/enterprise-ui/src/App.jsx`, `K3s-Cloud-Fog-Edge/k8-manifests/05-enterprise-tier/` |
| Security, traceability, and OTA integrity | Signed control path, checksum validation, and auditable deployment state | `Docker-Cloud-Fog-Edge/apps/industrial_mlops/security.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/deployment.py`, `Docker-Cloud-Fog-Edge/apps/services/edge_sync.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/orchestration.py` |
| Closed-loop lifecycle and rollback | Drift-aware retraining, candidate evaluation, promotion, and rollback as governed lifecycle events | `Docker-Cloud-Fog-Edge/apps/industrial_mlops/orchestration.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/ml.py`, `Docker-Cloud-Fog-Edge/notebooks/dags/` |
| CNC use case as architectural instantiation | Single-cell validation scenario used to exercise the architecture end-to-end | `Docker-Cloud-Fog-Edge/apps/industrial_mlops/cnc_data.py`, `Docker-Cloud-Fog-Edge/apps/services/cnc_simulator.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/ml.py` |
| Synthetic line and company reference line coexistence | Methodological separation between reproducibility and industrial realism | `Docker-Cloud-Fog-Edge/apps/industrial_mlops/company_dataset.py`, `Docker-Cloud-Fog-Edge/apps/industrial_mlops/data/company_cnc/`, `Docker-Cloud-Fog-Edge/notebooks/CompanyDatasetTraining.ipynb`, `docs/THESIS_SECTION_SYNTHETIC_VS_COMPANY_REFERENCE.md` |

## Evidence for the Main Paper Claims

### 1. The paper proposes a reusable industrial MLOps Reference Architecture

Repository evidence:

- `README.md`
- `Docker-Cloud-Fog-Edge/docker-compose.yml`
- `K3s-Cloud-Fog-Edge/k8-manifests/`

Why it matters:

These artefacts show that the architecture is not presented only conceptually. The same logical tiering is instantiated in two deployment forms, which supports the claim that the contribution is architectural and reusable rather than tied to a single local setup.

### 2. The architecture is operationalised through a vendor-neutral open-source stack

Repository evidence:

- `Docker-Cloud-Fog-Edge/build/`
- `Docker-Cloud-Fog-Edge/apps/`
- `Docker-Cloud-Fog-Edge/docker-compose.yml`
- `K3s-Cloud-Fog-Edge/k8-manifests/`

Why it matters:

The repository demonstrates that the proposed architecture can be instantiated without proprietary platform dependence, which directly addresses the replicability gap identified in the review.

### 3. The lifecycle is governed, not only deployed

Repository evidence:

- `Docker-Cloud-Fog-Edge/apps/industrial_mlops/orchestration.py`
- `Docker-Cloud-Fog-Edge/notebooks/dags/cnc_closed_loop_retraining_v1.py`
- `Docker-Cloud-Fog-Edge/notebooks/dags/cnc_governance_rollback_v1.py`
- `Docker-Cloud-Fog-Edge/apps/services/enterprise_api.py`

Why it matters:

These files provide the strongest support for the claim that the contribution goes beyond simple serving. They make retraining, promotion, rollback, and governance explicit lifecycle concerns.

### 4. The architecture separates inference and learning loops

Repository evidence:

- `Docker-Cloud-Fog-Edge/apps/services/edge_inference.py`
- `Docker-Cloud-Fog-Edge/apps/services/edge_sync.py`
- `Docker-Cloud-Fog-Edge/apps/industrial_mlops/deployment.py`
- `docs/MANUSCRIPT_DUAL_LOOP_SYNC_AND_OTA.md`

Why it matters:

This evidence supports the argument that edge continuity and central model evolution are intentionally separated and then coordinated through controlled deployment mechanisms.

### 5. The CNC use case validates the architecture under realistic industrial constraints

Repository evidence:

- `Docker-Cloud-Fog-Edge/apps/services/cnc_simulator.py`
- `Docker-Cloud-Fog-Edge/apps/industrial_mlops/cnc_data.py`
- `Docker-Cloud-Fog-Edge/apps/industrial_mlops/data/company_cnc/`
- `docs/THESIS_SECTION_SYNTHETIC_VS_COMPANY_REFERENCE.md`

Why it matters:

This evidence should be interpreted as support for architectural validation. It shows that the same MLOps backbone can be exercised through both a reproducible synthetic line and an industrially grounded company line.

## Reviewer-Oriented Entry Points

For article review, the recommended reading order inside the repository is:

1. `README.md`
2. `docs/ARTICLE_TO_REPOSITORY_MAP.md`
3. `docs/MANUSCRIPT_NOVELTY_AND_CONTRIBUTIONS.md`
4. `docs/MANUSCRIPT_DUAL_LOOP_SYNC_AND_OTA.md`
5. `docs/THESIS_SECTION_SYNTHETIC_VS_COMPANY_REFERENCE.md`
6. `docs/EXPERIMENT_REPLICATION_GUIDE.md`

## Scope Reminder

The repository supports the paper's claims about:

- architectural decomposition
- lifecycle governance
- open-source reproducibility
- dual-data validation strategy
- operational observability

It should not be interpreted as proving:

- cross-scenario generalisability
- exhaustive modelling of tool-breakage mechanisms
- fleet-scale consensus
- formally benchmarked hard real-time guarantees
