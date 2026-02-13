FROM jupyter/scipy-notebook:latest
LABEL authors="mmateo@cigip.upv.es"

# Switch to root to install system libs if needed
USER root

# Install Industrial Stack libraries
# - mlflow: Model Registry interaction
# - boto3: S3 (MinIO) connection
# - psycopg2-binary: DB connection (Timescale/Postgres)
# - paho-mqtt: MQTT (Mosquitto) connection
# - kafka-python: Kafka connection (New!)
# - alibi-detect & evidently: Drift detection and Monitoring
# - minio: Native MinIO client
RUN pip install \
    mlflow==2.7.1 \
    boto3 \
    psycopg2-binary \
    paho-mqtt \
    kafka-python \
    alibi-detect \
    evidently \
    minio

# Switch back to the default user to avoid permission issues
USER ${NB_UID}