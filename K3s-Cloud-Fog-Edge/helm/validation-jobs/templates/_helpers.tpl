{{- define "validation-jobs.namespace" -}}
{{- default .Release.Namespace .Values.namespaceOverride -}}
{{- end -}}

{{- define "validation-jobs.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/part-of: industrial-mlops-reference-architecture
validation.codex/run-id: {{ .Values.runId | quote }}
{{- end -}}

{{- define "validation-jobs.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{- define "validation-jobs.commonEnv" -}}
- name: TIMESCALE_HOST
  value: timescale-svc.{{ .Values.peerNamespaces.fog }}.svc.cluster.local
- name: TIMESCALE_PORT
  value: "5432"
- name: TIMESCALE_ADMIN_USER
  value: {{ .Values.credentials.timescaleUser | quote }}
- name: TIMESCALE_ADMIN_PASSWORD
  value: {{ .Values.credentials.timescalePassword | quote }}
- name: FACTORY_DB_NAME
  value: factory_db
- name: MLFLOW_TRACKING_URI
  value: http://mlflow-svc.{{ .Values.peerNamespaces.cloud }}.svc.cluster.local:5000
- name: MLFLOW_S3_ENDPOINT_URL
  value: http://minio-svc.{{ .Values.peerNamespaces.fog }}.svc.cluster.local:9000
- name: AWS_ACCESS_KEY_ID
  value: {{ .Values.credentials.minioAccessKey | quote }}
- name: AWS_SECRET_ACCESS_KEY
  value: {{ .Values.credentials.minioSecretKey | quote }}
- name: MQTT_HOST
  value: mosquitto-svc.{{ .Values.peerNamespaces.edge }}.svc.cluster.local
- name: MQTT_PORT
  value: "1883"
- name: INDUSTRIAL_SHARED_SECRET
  value: {{ .Values.sharedSecret | quote }}
- name: ENTERPRISE_API_KEY
  value: {{ .Values.enterpriseApiKey | quote }}
- name: ENTERPRISE_API_URL
  value: http://enterprise-api-svc.{{ .Values.peerNamespaces.enterprise }}.svc.cluster.local:8085
- name: EDGE_PROFILE_RESULTS_DIR
  value: /results/edge_profiling
- name: OTA_RESULTS_DIR
  value: /results/ota_continuity
- name: DRIFT_RESULTS_DIR
  value: /results/drift_robustness
- name: EDGE_BOOTSTRAP_MODEL_NAME
  value: cnc_tool_breakage_classifier
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
{{- end -}}
