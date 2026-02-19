#!/usr/bin/env bash
# F1 Worker — runs Jolpica or FastF1 ingestion for a date range, then publishes
# a completion/failure message to Pub/Sub.
#
# Environment variables (set by coordinator via Cloud Run Job execution overrides):
#   WORKER_TYPE            jolpica | fastf1
#   START                  start season/year (integer)
#   END                    end season/year (integer)
#   WORKER_ID              human-readable ID, e.g. f1-jolpica-worker-1
#   GCLOUD_PROJECT         GCP project ID       (default: f1optimizer)
#   PUBSUB_COMPLETION_TOPIC  topic for success   (default: f1-predictions-dev)
#   PUBSUB_ALERTS_TOPIC      topic for failure   (default: f1-alerts-dev)
set -uo pipefail

WORKER_TYPE="${WORKER_TYPE:-jolpica}"
START="${START:-1950}"
END="${END:-2026}"
WORKER_ID="${WORKER_ID:-worker-unknown}"
PROJECT="${GCLOUD_PROJECT:-f1optimizer}"
COMPLETION_TOPIC="${PUBSUB_COMPLETION_TOPIC:-f1-predictions-dev}"
ALERTS_TOPIC="${PUBSUB_ALERTS_TOPIC:-f1-alerts-dev}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${WORKER_ID}] $*"
}

log "=== F1 Worker starting: type=${WORKER_TYPE} start=${START} end=${END} ==="

EXIT=0

if [ "${WORKER_TYPE}" = "jolpica" ]; then
    python -m src.ingestion.ergast_ingestion \
        --start-season "${START}" \
        --end-season   "${END}"   \
        --worker-id    "${WORKER_ID}"
    EXIT=$?
elif [ "${WORKER_TYPE}" = "fastf1" ]; then
    python -m src.ingestion.fastf1_ingestion \
        --start-year "${START}" \
        --end-year   "${END}"   \
        --worker-id  "${WORKER_ID}"
    EXIT=$?
else
    log "ERROR: Unknown WORKER_TYPE='${WORKER_TYPE}'. Must be 'jolpica' or 'fastf1'."
    EXIT=1
fi

# ── Determine outcome ────────────────────────────────────────────────────────
STATUS="complete"
TOPIC="${COMPLETION_TOPIC}"
if [ "${EXIT}" -ne 0 ]; then
    STATUS="failed"
    TOPIC="${ALERTS_TOPIC}"
    log "Ingestion FAILED (exit ${EXIT})."
else
    log "Ingestion SUCCEEDED."
fi

# ── Publish to Pub/Sub via REST API (no gcloud CLI required) ─────────────────
log "Publishing ${STATUS} notification to Pub/Sub topic ${TOPIC}..."

# Get ADC token from GCE Metadata Service (available in all Cloud Run tasks)
TOKEN=$(curl -sf \
    -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null) || true

if [ -z "${TOKEN}" ]; then
    log "WARNING: Could not retrieve ADC token — Pub/Sub notification skipped."
else
    MSG_B64=$(python3 - <<PYEOF
import json, base64
payload = json.dumps({
    "worker_id":   "${WORKER_ID}",
    "status":      "${STATUS}",
    "type":        "${WORKER_TYPE}",
    "start":       ${START},
    "end":         ${END},
})
print(base64.b64encode(payload.encode()).decode())
PYEOF
)

    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST \
        "https://pubsub.googleapis.com/v1/projects/${PROJECT}/topics/${TOPIC}:publish" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"data\":\"${MSG_B64}\"}]}") || HTTP_CODE="000"

    if [ "${HTTP_CODE}" = "200" ]; then
        log "Pub/Sub notification sent (HTTP ${HTTP_CODE})."
    else
        log "WARNING: Pub/Sub publish returned HTTP ${HTTP_CODE} — continuing."
    fi
fi

log "=== Worker done (exit ${EXIT}) ==="
exit "${EXIT}"
