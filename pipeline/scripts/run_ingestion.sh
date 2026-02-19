#!/usr/bin/env bash
# Run the full F1 data ingestion pipeline: schema -> Ergast -> FastF1
# Errors in individual steps are logged; the pipeline continues to the next step.
set -uo pipefail

LOG_DIR="${LOG_DIR:-/tmp/ingestion_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ingestion_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== F1 Data Ingestion Pipeline ==="
log "Log file: $LOG_FILE"

PIPELINE_EXIT=0

# ---- 1. Apply schema -------------------------------------------------------
log "Step 1/3: Applying database schema..."
python - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import os
import pg8000.native

conn = pg8000.native.Connection(
    host=os.environ["DB_HOST"],
    port=int(os.environ.get("DB_PORT", 5432)),
    database=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    ssl_context=True,
)
schema_path = "/app/src/database/schema.sql"
with open(schema_path) as f:
    conn.run(f.read())
conn.close()
print("Schema applied successfully.")
PYEOF
SCHEMA_EXIT=${PIPESTATUS[0]}
if [ "$SCHEMA_EXIT" -ne 0 ]; then
    log "Step 1/3: Schema apply FAILED (exit $SCHEMA_EXIT) -- continuing"
    PIPELINE_EXIT=1
else
    log "Step 1/3: Schema applied."
fi

# ---- 2. Ergast historical ingestion (Jolpica, 1950-2026) -------------------
log "Step 2/3: Starting Ergast (Jolpica) ingestion..."
python -m src.ingestion.ergast_ingestion 2>&1 | tee -a "$LOG_FILE"
ERGAST_EXIT=${PIPESTATUS[0]}
if [ "$ERGAST_EXIT" -ne 0 ]; then
    log "Step 2/3: Ergast ingestion FAILED (exit $ERGAST_EXIT) -- continuing"
    PIPELINE_EXIT=1
else
    log "Step 2/3: Ergast ingestion complete."
fi

# ---- 3. FastF1 ingestion (Q + R sessions, 2018-2026) ----------------------
log "Step 3/3: Starting FastF1 ingestion (Qualifying + Race, 2018-2026, 5s delay/session, 30s pause every 10 sessions)..."
python -m src.ingestion.fastf1_ingestion 2>&1 | tee -a "$LOG_FILE"
FASTF1_EXIT=${PIPESTATUS[0]}
if [ "$FASTF1_EXIT" -ne 0 ]; then
    log "Step 3/3: FastF1 ingestion FAILED (exit $FASTF1_EXIT)"
    PIPELINE_EXIT=1
else
    log "Step 3/3: FastF1 ingestion complete."
fi

log "=== Ingestion pipeline finished (exit $PIPELINE_EXIT) ==="
exit $PIPELINE_EXIT
