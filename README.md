# Industrial MLOps Reference Architecture 🏭☁️🤖

A reference implementation for **Distributed Industrial AI** based on a **Cloud-Fog-Edge** topology.

This project simulates a complete "Smart Factory" environment: from data generation on sensors (Edge) to model training and orchestration in the cloud (Cloud), passing through an intermediate persistence layer (Fog).

## 🏗 Global Architecture

The system evolves beyond a monolithic architecture to implement three distinct logical layers:

* **☁️ Cloud Tier (Orchestration & AI):**
    * **Apache Airflow:** Retraining pipeline orchestration.
    * **MLflow:** Model registry and lifecycle management.
* **🌫️ Fog Tier (Data & Storage):**
    * **TimescaleDB:** Unified database for time-series and metadata.
    * **MinIO:** Object storage (S3 compatible).
* **🏭 Edge Tier (Plant & Ingestion):**
    * **Apache Kafka:** High-throughput streaming backbone.
    * **Node-RED:** Industrial PLC and sensor simulator.
    * **Mosquitto:** MQTT Broker for lightweight IoT connectivity.

---

## 📂 Project Structure

This repository offers two deployment "flavors," each with its own detailed documentation:

### 1. [Docker Development Environment](./Docker-Cloud-Fog-Edge)
Ideal for fast local development, Proof of Concepts (PoC), and experimentation. It uses `docker-compose` to spin up the entire stack on a unified network.
* 👉 **[Go to Docker Guide](./Docker-Cloud-Fog-Edge/README.md)**

### 2. [Kubernetes Production Environment)](./K3s-Cloud-Fog-Edge)
Robust deployment using **Namespaces** to isolate layers (Cloud, Fog, Edge). Ideal for simulating a real distributed environment on Kubernetes (Minikube, K3s, Docker Desktop K8s).
* 👉 **[Go to Kubernetes Guide](./Kubernetes-Cloud-Fog-Edge/README.md)**

---

## 🚀 Key Technologies

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Kubernetes](https://img.shields.io/badge/Kubernetes-K8s-326CE5)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-Orchestration-017CEE)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-Streaming-231F20)
![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2)
![Timescale](https://img.shields.io/badge/TimescaleDB-Time%20Series-FDB515)

## 🏁 Quick Start

1. Clone this repository.
2. Choose your environment (`Docker-Cloud-Fog-Edge` or `Kubernetes-Cloud-Fog-Edge`).
3. Navigate to the corresponding folder and follow the internal `README.md`.