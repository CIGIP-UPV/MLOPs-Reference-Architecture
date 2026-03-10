# Kubernetes Industrial MLOps Lab (Cloud-Fog-Edge)

This folder contains the Kubernetes flavour of the industrial MLOps reference architecture.

The repository now provides **two parallel deployment paths** for K3s:

1. the original raw manifests in `k8-manifests/`
2. a **Helm-based path** designed for **Rancher UI-driven deployment**

For the current project scope, the Helm path is the preferred option because it preserves the same logical architecture while making the stack easier to install, upgrade, and validate from Rancher without requiring direct `kubectl` access.

## What is in this folder

- `k8-manifests/`
  Original Kubernetes manifests kept for traceability and low-level inspection.
- `helm/fog-tier/`
  Fog tier chart: TimescaleDB, MinIO, Fog Bridge, and fog-side persistence.
- `helm/cloud-tier/`
  Cloud tier chart: Airflow, MLflow, and the bootstrap job.
- `helm/edge-tier/`
  Edge tier chart: Mosquitto, Kafka, Node-RED, CNC simulator, Edge Sync, and Edge Inference.
- `helm/governance/`
  Governance tier chart: Prometheus and Grafana.
- `helm/enterprise-tier/`
  Enterprise tier chart: Enterprise API and Enterprise UI.
- `helm/validation-jobs/`
  Optional validation jobs for repeated edge profiling, OTA continuity, and drift robustness.

## Recommended deployment order in Rancher

Install the charts in this order and keep the namespace names exactly as listed below so that cross-tier DNS names remain aligned with the platform defaults:

1. `fog-tier` chart in namespace `fog-tier`
2. `cloud-tier` chart in namespace `cloud-tier`
3. `edge-tier` chart in namespace `edge-tier`
4. `governance` chart in namespace `governance`
5. `enterprise-tier` chart in namespace `enterprise-tier`

After the platform is healthy, run the optional validation jobs from the `validation-jobs` chart in namespace `edge-tier`.

## Validation path

The Helm charts are intended to support the same validation families already exercised in the Docker flavour:

- drift logic robustness
- OTA/update continuity
- repeated edge inference latency
- prediction count and schema mismatch count
- CPU and memory observation for the edge inference path

The detailed Rancher-oriented procedure is documented here:

- [Rancher Helm Validation Guide](../docs/RANCHER_HELM_VALIDATION_GUIDE.md)

## Local validation performed in this repository

Because this workspace does not have access to the target K3s server, validation was limited to chart and script checks that can be executed locally:

- `helm lint` passed for all six charts
- `helm template` rendered the five platform charts and the three validation-job configurations without template errors
- Python syntax checks passed for the validation job scripts

This means the charts are locally renderable and structurally valid. It does **not** replace runtime verification inside the target Rancher-managed cluster.

## Notes

- The Helm charts preserve the same resource names and service endpoints used by the raw manifests.
- The raw manifests remain in the repository because they are still useful as low-level architectural evidence.
- The validation jobs are intentionally optional. They are meant to produce operational evidence for the article revision, not to alter the platform lifecycle.
