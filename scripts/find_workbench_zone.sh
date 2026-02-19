#!/bin/bash
# Finds the first available GCP zone for the Vertex AI Workbench instance.
# Outputs a single zone name (e.g. "us-central1-b") to stdout.
# Used by infra/terraform/workbench_zone.tf and cloudbuild.yaml.

ZONES=("us-central1-a" "us-central1-b" "us-central1-c"
       "us-central1-f" "us-east1-b" "us-east1-c" "us-east1-d")

for ZONE in "${ZONES[@]}"; do
  AVAILABLE=$(gcloud compute zones describe "$ZONE" \
    --project=f1optimizer \
    --format="value(status)" 2>/dev/null)
  if [ "$AVAILABLE" = "UP" ]; then
    echo "$ZONE"
    exit 0
  fi
done

echo "us-central1-c"  # default fallback
