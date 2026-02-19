#!/usr/bin/env python3
"""
F1 Multi-Agent Ingestion Coordinator

Triggers 7 parallel Cloud Run Job executions (4 Jolpica + 3 FastF1),
polls their status every 60 seconds, prints a live progress table,
publishes a completion message to Pub/Sub, and emails the team via
Cloud Monitoring notification channels (same pattern as monitor_ingestion.py).

Workers launched:
  f1-jolpica-worker  x4 : 1950-1979 / 1980-1999 / 2000-2012 / 2013-2026
  f1-fastf1-worker   x3 : 2018-2020 / 2021-2023 / 2024-2026

Auth: Application Default Credentials (ADC) — no hardcoded secrets.
"""

import base64
import datetime
import json
import sys
import time
import urllib.parse
from typing import Optional

import google.auth
import google.auth.transport.requests
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT = "f1optimizer"
REGION = "us-central1"
POLL_INTERVAL_SECS = 60

CLOUD_RUN_BASE = "https://run.googleapis.com/v2"
PUBSUB_BASE = "https://pubsub.googleapis.com/v1"
LOGGING_BASE = "https://logging.googleapis.com/v2"
MONITORING_BASE = "https://monitoring.googleapis.com/v3"

COMPLETION_TOPIC = f"projects/{PROJECT}/topics/f1-predictions-dev"
ALERTS_TOPIC = f"projects/{PROJECT}/topics/f1-alerts-dev"

SUCCESS_EMAILS = [
    "bhargavsp01@gmail.com",
    "akshattkain0217@gmail.com",
    "ronitshetty16@gmail.com",
    "dhanyasureshnaik@gmail.com",
    "pooja.rajasimha@gmail.com",
    "ajithsri3103@gmail.com",
]
FAILED_EMAILS = ["bhargavsp01@gmail.com"]

# (worker_id, job_name, worker_type, start, end)
WORKERS = [
    ("f1-jolpica-worker-1", "f1-jolpica-worker", "jolpica", 1950, 1979),
    ("f1-jolpica-worker-2", "f1-jolpica-worker", "jolpica", 1980, 1999),
    ("f1-jolpica-worker-3", "f1-jolpica-worker", "jolpica", 2000, 2012),
    ("f1-jolpica-worker-4", "f1-jolpica-worker", "jolpica", 2013, 2026),
    ("f1-fastf1-worker-1",  "f1-fastf1-worker",  "fastf1",  2018, 2020),
    ("f1-fastf1-worker-2",  "f1-fastf1-worker",  "fastf1",  2021, 2023),
    ("f1-fastf1-worker-3",  "f1-fastf1-worker",  "fastf1",  2024, 2026),
]


# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def get_auth_session():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    _do_refresh(creds)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {creds.token}"})
    return session, creds


def _do_refresh(creds) -> None:
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)


def refresh(session: requests.Session, creds) -> None:
    _do_refresh(creds)
    session.headers.update({"Authorization": f"Bearer {creds.token}"})


# ─────────────────────────────────────────────────────────────────────────────
# Trigger + poll helpers
# ─────────────────────────────────────────────────────────────────────────────


def trigger_worker(
    session: requests.Session,
    creds,
    worker_id: str,
    job_name: str,
    worker_type: str,
    start: int,
    end: int,
) -> str:
    """Trigger a Cloud Run Job execution with env var overrides.

    Returns the fully-qualified execution resource name once the LRO resolves.
    """
    refresh(session, creds)
    url = (
        f"{CLOUD_RUN_BASE}/projects/{PROJECT}/locations/{REGION}"
        f"/jobs/{job_name}:run"
    )
    payload = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {"name": "WORKER_TYPE", "value": worker_type},
                        {"name": "START",       "value": str(start)},
                        {"name": "END",         "value": str(end)},
                        {"name": "WORKER_ID",   "value": worker_id},
                    ]
                }
            ],
            "taskCount": 1,
            "timeout": "21600s",
        }
    }
    resp = session.post(url, json=payload)
    resp.raise_for_status()
    op_name = resp.json()["name"]

    # Wait for the long-running operation to resolve and return the execution name
    op_url = f"{CLOUD_RUN_BASE}/{op_name}"
    for _ in range(36):  # up to 3 minutes
        time.sleep(5)
        refresh(session, creds)
        resp = session.get(op_url)
        resp.raise_for_status()
        op = resp.json()
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(
                    f"Worker {worker_id} failed to start: {op['error']}"
                )
            return op["response"]["name"]

    raise RuntimeError(
        f"Timeout waiting for {worker_id} execution to be assigned (>3 min)"
    )


def get_execution_status(
    session: requests.Session, creds, execution_name: str
) -> dict:
    refresh(session, creds)
    resp = session.get(f"{CLOUD_RUN_BASE}/{execution_name}")
    resp.raise_for_status()
    return resp.json()


def derive_status(data: dict) -> str:
    """Return RUNNING, SUCCEEDED, or FAILED from execution data."""
    if data.get("completionTime"):
        if data.get("succeededCount", 0) > 0:
            return "SUCCEEDED"
        if data.get("failedCount", 0) > 0:
            return "FAILED"
    for cond in data.get("conditions", []):
        if cond.get("type") == "Completed":
            state = cond.get("state", "")
            if state == "CONDITION_SUCCEEDED":
                return "SUCCEEDED"
            if state == "CONDITION_FAILED":
                return "FAILED"
    return "RUNNING"


# ─────────────────────────────────────────────────────────────────────────────
# Progress display
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {
    "RUNNING":   "🔄",
    "SUCCEEDED": "✅",
    "FAILED":    "❌",
    "PENDING":   "⏳",
}


def print_progress_table(worker_states: dict) -> None:
    print(
        f"\n[{_now()}] ─── Worker Progress ─────────────────────────────────────────"
    )
    print(f"  {'Worker ID':<26} {'Job':<22} {'Range':<12} {'Status'}")
    print(f"  {'─'*26} {'─'*22} {'─'*12} {'─'*12}")
    for wid, info in worker_states.items():
        icon = _STATUS_ICON.get(info["status"], " ")
        rng = f"{info['start']}-{info['end']}"
        print(
            f"  {wid:<26} {info['job_name']:<22} {rng:<12} "
            f"{icon} {info['status']}"
        )
    running = sum(1 for i in worker_states.values() if i["status"] == "RUNNING")
    done = sum(1 for i in worker_states.values() if i["status"] != "RUNNING")
    print(f"\n  {done}/{len(worker_states)} complete  |  {running} still running\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pub/Sub
# ─────────────────────────────────────────────────────────────────────────────


def publish_pubsub(
    session: requests.Session, creds, topic: str, message: dict
) -> None:
    refresh(session, creds)
    data = base64.b64encode(json.dumps(message).encode()).decode()
    resp = session.post(
        f"{PUBSUB_BASE}/{topic}:publish",
        json={"messages": [{"data": data}]},
    )
    resp.raise_for_status()


# ─────────────────────────────────────────────────────────────────────────────
# Email notifications (Cloud Monitoring ephemeral alert pattern)
# ─────────────────────────────────────────────────────────────────────────────


def list_channel_ids_for_emails(
    session: requests.Session, creds, target_emails: list
) -> list:
    refresh(session, creds)
    resp = session.get(
        f"{MONITORING_BASE}/projects/{PROJECT}/notificationChannels"
    )
    resp.raise_for_status()
    channels = resp.json().get("notificationChannels", [])

    matched = []
    found_emails = set()
    for ch in channels:
        if ch.get("type") == "email":
            email = ch.get("labels", {}).get("email_address", "")
            if email in target_emails:
                matched.append(ch["name"])
                found_emails.add(email)
                print(f"[{_now()}]   Channel found for: {email}")

    for m in sorted(set(target_emails) - found_emails):
        print(
            f"[{_now()}]   WARNING: no notification channel for {m} "
            "— add in Cloud Monitoring > Notification Channels"
        )
    return matched


def _build_email_content(
    status: str,
    completed_at: str,
    failed_workers: list,
    worker_states: dict,
) -> tuple:
    label = "SUCCESS ✅" if status == "SUCCEEDED" else "FAILURE ❌"
    subject = f"[F1 Parallel Ingestion] {label} — {completed_at}"

    rows = "\n".join(
        f"  - {wid}: {info['status']} ({info['start']}-{info['end']})"
        for wid, info in worker_states.items()
    )

    if status == "SUCCEEDED":
        body = f"""## F1 Parallel Ingestion — SUCCESS ✅

**Status:** All 7 workers completed successfully
**Completed:** {completed_at}
**Project:** `{PROJECT}` / `{REGION}`

### Worker Summary
{rows}

Data is now available in Cloud SQL Studio (`f1_strategy` database).
"""
    else:
        failed_list = "\n".join(f"- `{w}`" for w in failed_workers)
        body = f"""## F1 Parallel Ingestion — FAILURE ❌

**Status:** {len(failed_workers)} worker(s) failed
**Completed:** {completed_at}
**Project:** `{PROJECT}` / `{REGION}`

### Failed Workers
{failed_list}

### All Workers
{rows}

Check Cloud Logging:
https://console.cloud.google.com/run/jobs?project={PROJECT}
"""
    return subject, body


def send_email_notification(
    session: requests.Session,
    creds,
    status: str,
    failed_workers: list,
    completed_at: str,
    worker_states: dict,
) -> None:
    """Send email via ephemeral Cloud Monitoring alert policy."""
    subject, body = _build_email_content(
        status, completed_at, failed_workers, worker_states
    )
    target_emails = SUCCESS_EMAILS if status == "SUCCEEDED" else FAILED_EMAILS

    print(
        f"[{_now()}] Resolving notification channels for "
        f"{len(target_emails)} recipient(s)…"
    )
    channel_ids = list_channel_ids_for_emails(session, creds, target_emails)
    if not channel_ids:
        print(f"[{_now()}] WARNING: No matching channels — email skipped.")
        return

    run_stamp = int(time.time())
    metric_type = f"custom.googleapis.com/f1/coordinator_notify_{run_stamp}"
    encoded_type = urllib.parse.quote(metric_type, safe="")

    # 1. Create metric descriptor
    refresh(session, creds)
    session.post(
        f"{MONITORING_BASE}/projects/{PROJECT}/metricDescriptors",
        json={
            "type": metric_type,
            "metricKind": "GAUGE",
            "valueType": "INT64",
            "displayName": "F1 coordinator notification trigger (ephemeral)",
            "description": "Ephemeral metric used to fire job-completion emails.",
        },
    ).raise_for_status()
    print(f"[{_now()}] Metric descriptor created: {metric_type}")

    # 2. Write a data point (value = 1)
    now_rfc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    refresh(session, creds)
    session.post(
        f"{MONITORING_BASE}/projects/{PROJECT}/timeSeries",
        json={
            "timeSeries": [
                {
                    "metric": {"type": metric_type, "labels": {}},
                    "resource": {
                        "type": "global",
                        "labels": {"project_id": PROJECT},
                    },
                    "points": [
                        {
                            "interval": {"endTime": now_rfc},
                            "value": {"int64Value": "1"},
                        }
                    ],
                }
            ]
        },
    ).raise_for_status()
    print(f"[{_now()}] Metric data point written")

    # 3. Create alert policy
    refresh(session, creds)
    resp = session.post(
        f"{MONITORING_BASE}/projects/{PROJECT}/alertPolicies",
        json={
            "displayName": f"[EPHEMERAL] F1 Coordinator {status} {run_stamp}",
            "enabled": True,
            "notificationChannels": channel_ids,
            "documentation": {
                "subject": subject,
                "content": body,
                "mimeType": "text/markdown",
            },
            "conditions": [
                {
                    "displayName": "F1 coordinator notification trigger",
                    "conditionThreshold": {
                        "filter": (
                            f'metric.type="{metric_type}"'
                            ' AND resource.type="global"'
                        ),
                        "comparison": "COMPARISON_GT",
                        "thresholdValue": 0,
                        "duration": "0s",
                        "aggregations": [
                            {
                                "alignmentPeriod": "60s",
                                "perSeriesAligner": "ALIGN_MAX",
                            }
                        ],
                    },
                }
            ],
            "combiner": "OR",
            "alertStrategy": {"autoClose": "3600s"},
        },
    )
    resp.raise_for_status()
    policy_name = resp.json()["name"]
    print(f"[{_now()}] Alert policy created: {policy_name}")
    print(f"[{_now()}] Waiting 3 minutes for Cloud Monitoring to dispatch emails…")
    time.sleep(180)

    # 4. Clean up
    refresh(session, creds)
    session.delete(f"{MONITORING_BASE}/{policy_name}").raise_for_status()
    refresh(session, creds)
    session.delete(
        f"{MONITORING_BASE}/projects/{PROJECT}/metricDescriptors/{encoded_type}"
    ).raise_for_status()
    print(
        f"[{_now()}] Email dispatched to {len(target_emails)} recipient(s). "
        "Ephemeral resources cleaned up."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[{_now()}] === F1 Multi-Agent Coordinator starting ===")
    print(f"[{_now()}] Launching {len(WORKERS)} parallel workers…")

    session, creds = get_auth_session()

    # ── Step 1: Trigger all 7 workers ────────────────────────────────────────
    worker_states: dict = {}
    for worker_id, job_name, worker_type, start, end in WORKERS:
        print(f"[{_now()}] Triggering {worker_id} ({job_name}, {start}-{end})…")
        try:
            execution_name = trigger_worker(
                session, creds, worker_id, job_name, worker_type, start, end
            )
            exec_short = execution_name.split("/")[-1]
            worker_states[worker_id] = {
                "job_name":       job_name,
                "worker_type":    worker_type,
                "start":          start,
                "end":            end,
                "execution_name": execution_name,
                "status":         "RUNNING",
            }
            print(f"[{_now()}] ✓ {worker_id} started → execution: {exec_short}")
        except Exception as exc:
            print(f"[{_now()}] ✗ Failed to trigger {worker_id}: {exc}")
            worker_states[worker_id] = {
                "job_name":       job_name,
                "worker_type":    worker_type,
                "start":          start,
                "end":            end,
                "execution_name": None,
                "status":         "FAILED",
            }

    print_progress_table(worker_states)
    print(f"[{_now()}] All triggers sent. Polling every {POLL_INTERVAL_SECS}s…")

    # ── Step 2: Poll loop ─────────────────────────────────────────────────────
    while True:
        time.sleep(POLL_INTERVAL_SECS)

        any_still_running = False
        for worker_id, info in worker_states.items():
            if info["status"] in ("SUCCEEDED", "FAILED"):
                continue
            if info["execution_name"] is None:
                info["status"] = "FAILED"
                continue
            try:
                data = get_execution_status(session, creds, info["execution_name"])
                new_status = derive_status(data)
                if new_status != info["status"]:
                    info["status"] = new_status
                    icon = _STATUS_ICON.get(new_status, " ")
                    print(f"[{_now()}] {icon} {worker_id} → {new_status}")
            except Exception as exc:
                print(f"[{_now()}] WARNING: Could not poll {worker_id}: {exc}")
                any_still_running = True
                continue

            if info["status"] == "RUNNING":
                any_still_running = True

        print_progress_table(worker_states)

        if not any_still_running:
            break

    # ── Step 3: Summarize ─────────────────────────────────────────────────────
    completed_at = _now()
    failed_workers = [
        wid for wid, info in worker_states.items() if info["status"] == "FAILED"
    ]
    overall_status = "SUCCEEDED" if not failed_workers else "FAILED"

    print(f"\n[{_now()}] === All workers finished ===")
    print(f"[{_now()}] Overall status: {overall_status}")
    if failed_workers:
        print(f"[{_now()}] Failed workers: {failed_workers}")

    # ── Step 4: Pub/Sub completion message ────────────────────────────────────
    topic = COMPLETION_TOPIC if overall_status == "SUCCEEDED" else ALERTS_TOPIC
    try:
        publish_pubsub(
            session,
            creds,
            topic,
            {
                "coordinator":    "f1-data-coordinator",
                "status":         overall_status,
                "completed_at":   completed_at,
                "workers":        {wid: info["status"] for wid, info in worker_states.items()},
                "failed_workers": failed_workers,
            },
        )
        print(f"[{_now()}] Published completion to Pub/Sub: {topic}")
    except Exception as exc:
        print(f"[{_now()}] WARNING: Pub/Sub publish failed: {exc}")

    # ── Step 5: Email notification ────────────────────────────────────────────
    try:
        send_email_notification(
            session,
            creds,
            overall_status,
            failed_workers,
            completed_at,
            worker_states,
        )
    except Exception as exc:
        print(f"[{_now()}] WARNING: Email notification failed: {exc}")

    print(f"[{_now()}] === Coordinator done. Final status: {overall_status} ===")
    sys.exit(0 if overall_status == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()
