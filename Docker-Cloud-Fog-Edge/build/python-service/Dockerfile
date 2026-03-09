FROM python:3.11-slim
LABEL authors="mmateo@cigip.upv.es"

WORKDIR /opt/industrial-mlops

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY apps/requirements-python-service.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps ./apps
ENV PYTHONPATH=/opt/industrial-mlops/apps

CMD ["python", "-m", "services.db_bootstrap"]
