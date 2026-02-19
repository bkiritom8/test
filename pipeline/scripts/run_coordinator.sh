#!/usr/bin/env bash
# F1 Multi-Agent Coordinator entrypoint.
# Delegates to coordinator.py which triggers 7 parallel workers via Cloud Run v2 API.
set -uo pipefail

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting F1 coordinator..."
exec python /app/scripts/coordinator.py
