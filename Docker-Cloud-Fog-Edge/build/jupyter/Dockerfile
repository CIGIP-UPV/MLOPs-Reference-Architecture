FROM jupyter/scipy-notebook:latest
LABEL authors="mmateo@cigip.upv.es"

USER root
RUN pip install --no-cache-dir \
    mlflow==2.7.1 \
    boto3 \
    fastapi \
    psycopg2-binary \
    paho-mqtt \
    kafka-python \
    alibi-detect \
    evidently \
    minio \
    prometheus-client \
    scikit-learn \
    scipy

USER ${NB_UID}
