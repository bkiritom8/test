#!/usr/bin/env python3
"""
F1 Data Ingestion Job Monitor

STEP 1 — Triggers Cloud Run Job f1-data-ingestion
STEP 2 — Polls for completion every 30 seconds via Cloud Run v2 REST API
STEP 3 — On completion, sends email via Cloud Monitoring notification channels
STEP 4 — Email includes status, timestamp, logs link, and (if FAILED) last 20 log lines

Auth: Application Default Credentials (ADC) — no hardcoded secrets.
"""

import datetime
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
JOB_NAME = "f1-data-ingestion"
POLL_INTERVAL_SECS = 30

LOGS_LINK = (
    f"https://console.cloud.google.com/run/jobs/details/"
    f"{REGION}/{JOB_NAME}/executions?project={PROJECT}"
)

SUCCESS_EMAILS = [
    "bhargavsp01@gmail.com",
    "akshattkain0217@gmail.com",
    "ronitshetty16@gmail.com",
    "dhanyasureshnaik@gmail.com",
    "pooja.rajasimha@gmail.com",
    "ajithsri3103@gmail.com",
]

FAILED_EMAILS = ["bhargavsp01@gmail.com"]

CLOUD_RUN_BASE = "https://run.googleapis.com/v2"
LOGGING_BASE = "https://logging.googleapis.com/v2"
MONITORING_BASE = "https://monitoring.googleapis.com/v3"

# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def get_auth_session() -> tuple:
    """Create an authenticated requests.Session using ADC."""
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    _refresh(creds)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {creds.token}"})
    return session, creds


def _refresh(creds) -> None:
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)


def refresh(session: requests.Session, creds) -> None:
    """Refresh ADC token and update session Authorization header."""
    _refresh(creds)
    session.headers.update({"Authorization": f"Bearer {creds.token}"})


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Trigger Cloud Run Job
# ─────────────────────────────────────────────────────────────────────────────


def trigger_job(session: requests.Session, creds) -> str:
    """POST :run to the Cloud Run Job and return the long-running operation name."""
    refresh(session, creds)
    url = (
        f"{CLOUD_RUN_BASE}/projects/{PROJECT}/locations/{REGION}"
        f"/jobs/{JOB_NAME}:run"
    )
    print(f"[{_now()}] Triggering Cloud Run Job: {JOB_NAME} ...")
    resp = session.post(url, json={})
    resp.raise_for_status()
    op_name = resp.json()["name"]
    print(f"[{_now()}] Operation: {op_name}")
    return op_name


def resolve_execution_name(session: requests.Session, creds, op_name: str) -> str:
    """Poll the long-running operation until the execution resource name is returned."""
    op_url = f"{CLOUD_RUN_BASE}/{op_name}"
    print(f"[{_now()}] Waiting for execution to be assigned...")
    while True:
        refresh(session, creds)
        resp = session.get(op_url)
        resp.raise_for_status()
        op = resp.json()
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"Job failed to start: {op['error']}")
            execution_name = op["response"]["name"]
            print(f"[{_now()}] Execution: {execution_name}")
            return execution_name
        time.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Poll for Completion
# ─────────────────────────────────────────────────────────────────────────────


def poll_until_complete(
    session: requests.Session, creds, execution_name: str
) -> tuple:
    """Poll execution every POLL_INTERVAL_SECS until a terminal state is reached.

    Returns (status, exec_data) where status is 'SUCCEEDED' or 'FAILED'.
    """
    exec_url = f"{CLOUD_RUN_BASE}/{execution_name}"
    print(f"[{_now()}] Polling every {POLL_INTERVAL_SECS}s ...")

    while True:
        refresh(session, creds)
        resp = session.get(exec_url)
        resp.raise_for_status()
        data = resp.json()

        # Primary: completionTime indicates the job is done
        if data.get("completionTime"):
            succeeded = data.get("succeededCount", 0)
            failed = data.get("failedCount", 0)
            if succeeded > 0:
                print(f"[{_now()}] SUCCEEDED (succeededCount={succeeded})")
                return "SUCCEEDED", data
            if failed > 0:
                print(f"[{_now()}] FAILED (failedCount={failed})")
                return "FAILED", data

        # Fallback: check Completed condition
        for cond in data.get("conditions", []):
            if cond.get("type") == "Completed":
                state = cond.get("state", "")
                if state == "CONDITION_SUCCEEDED":
                    print(f"[{_now()}] SUCCEEDED (condition)")
                    return "SUCCEEDED", data
                if state == "CONDITION_FAILED":
                    print(f"[{_now()}] FAILED (condition)")
                    return "FAILED", data

        print(f"[{_now()}] Still running ... next check in {POLL_INTERVAL_SECS}s")
        time.sleep(POLL_INTERVAL_SECS)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch logs from Cloud Logging
# ─────────────────────────────────────────────────────────────────────────────


def fetch_last_logs(
    session: requests.Session, creds, execution_name: str, n: int = 20
) -> str:
    """Fetch the last N log lines for this execution from Cloud Logging."""
    exec_id = execution_name.split("/")[-1]
    refresh(session, creds)
    body = {
        "resourceNames": [f"projects/{PROJECT}"],
        "filter": (
            f'resource.type="cloud_run_job" '
            f'AND resource.labels.job_name="{JOB_NAME}" '
            f'AND labels."run.googleapis.com/execution-name"="{exec_id}"'
        ),
        "orderBy": "timestamp desc",
        "pageSize": n,
    }
    resp = session.post(f"{LOGGING_BASE}/entries:list", json=body)
    resp.raise_for_status()
    entries = resp.json().get("entries", [])

    lines = []
    for entry in reversed(entries):
        stamp = entry.get("timestamp", "")
        severity = entry.get("severity", "INFO")
        msg = entry.get("textPayload") or str(
            entry.get("jsonPayload", {}).get("message", "")
        )
        lines.append(f"[{stamp}] {severity}: {msg}")

    return "\n".join(lines) if lines else "(no log entries retrieved)"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3+4 — Send notifications via Cloud Monitoring channels
# ─────────────────────────────────────────────────────────────────────────────


def list_channel_ids_for_emails(
    session: requests.Session, creds, target_emails: list
) -> list:
    """Return notification channel resource names matching the target email addresses."""
    refresh(session, creds)
    resp = session.get(
        f"{MONITORING_BASE}/projects/{PROJECT}/notificationChannels"
    )
    resp.raise_for_status()
    channels = resp.json().get("notificationChannels", [])

    matched = []
    for ch in channels:
        if ch.get("type") == "email":
            email = ch.get("labels", {}).get("email_address", "")
            if email in target_emails:
                matched.append(ch["name"])
                print(f"[{_now()}]   Matched channel for: {email}")

    missing = set(target_emails) - {
        ch.get("labels", {}).get("email_address", "")
        for ch in channels
        if ch.get("type") == "email"
    }
    for m in sorted(missing):
        print(
            f"[{_now()}]   WARNING: no notification channel found for {m} — "
            "add it in Cloud Monitoring > Notification Channels"
        )

    return matched


def _build_email_content(
    status: str, completed_at: str, log_lines: Optional[str]
) -> tuple:
    """Return (subject, markdown_body) for the notification email."""
    label = "SUCCESS" if status == "SUCCEEDED" else "FAILURE"
    subject = f"[F1 Ingestion] {label} — {completed_at}"
    body = f"""## F1 Data Ingestion Job — {status}

**Status:** {status}
**Completed:** {completed_at}
**Job:** `{JOB_NAME}` (project: `{PROJECT}`, region: `{REGION}`)
**Logs:** [View in Cloud Console]({LOGS_LINK})
"""
    if log_lines:
        body += f"""
---

### Last 20 Log Lines

```
{log_lines}
```
"""
    return subject, body


def send_via_alert_policy(
    session: requests.Session,
    creds,
    channel_ids: list,
    subject: str,
    body: str,
    status: str,
) -> None:
    """
    Send emails through the existing notification channels by:
      1. Creating an ephemeral custom metric descriptor
      2. Writing a data point (value=1) to immediately satisfy a threshold condition
      3. Creating a temporary alert policy with the custom documentation as the email body
      4. Waiting ~3 minutes for Cloud Monitoring to evaluate and dispatch notifications
      5. Cleaning up the policy and metric descriptor

    Cloud Monitoring injects the policy's documentation.subject/content into
    the email it sends to the notification channels.
    """
    run_stamp = int(time.time())
    metric_type = f"custom.googleapis.com/f1/ingestion_notify_{run_stamp}"
    encoded_type = urllib.parse.quote(metric_type, safe="")

    # ── 1. Create custom metric descriptor ──────────────────────────────────
    refresh(session, creds)
    resp = session.post(
        f"{MONITORING_BASE}/projects/{PROJECT}/metricDescriptors",
        json={
            "type": metric_type,
            "metricKind": "GAUGE",
            "valueType": "INT64",
            "displayName": "F1 ingestion notification trigger (ephemeral)",
            "description": "Ephemeral metric used to fire job-completion emails.",
        },
    )
    resp.raise_for_status()
    print(f"[{_now()}] Custom metric descriptor created: {metric_type}")

    # ── 2. Write a time series point (value = 1) ─────────────────────────────
    refresh(session, creds)
    now_rfc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = session.post(
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
    )
    resp.raise_for_status()
    print(f"[{_now()}] Metric data point written (value=1)")

    # ── 3. Create alert policy: fires when metric > 0 ────────────────────────
    refresh(session, creds)
    resp = session.post(
        f"{MONITORING_BASE}/projects/{PROJECT}/alertPolicies",
        json={
            "displayName": f"[EPHEMERAL] F1 Ingestion {status} {run_stamp}",
            "enabled": True,
            "notificationChannels": channel_ids,
            "documentation": {
                "subject": subject,
                "content": body,
                "mimeType": "text/markdown",
            },
            "conditions": [
                {
                    "displayName": "F1 ingestion notification trigger",
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
    print(
        f"[{_now()}] Waiting 3 minutes for Cloud Monitoring to evaluate and dispatch ..."
    )
    time.sleep(180)

    # ── 4. Clean up ──────────────────────────────────────────────────────────
    refresh(session, creds)
    session.delete(f"{MONITORING_BASE}/{policy_name}").raise_for_status()
    print(f"[{_now()}] Alert policy deleted")

    refresh(session, creds)
    session.delete(
        f"{MONITORING_BASE}/projects/{PROJECT}/metricDescriptors/{encoded_type}"
    ).raise_for_status()
    print(f"[{_now()}] Custom metric descriptor deleted")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[{_now()}] === F1 Ingestion Monitor starting ===")

    session, creds = get_auth_session()

    # STEP 1 — Trigger
    op_name = trigger_job(session, creds)
    execution_name = resolve_execution_name(session, creds, op_name)

    # STEP 2 — Poll
    status, _exec_data = poll_until_complete(session, creds, execution_name)
    completed_at = _now()

    # Fetch logs on failure
    log_lines: Optional[str] = None
    if status == "FAILED":
        print(f"[{_now()}] Fetching last 20 log lines from Cloud Logging ...")
        log_lines = fetch_last_logs(session, creds, execution_name)
        print(log_lines)

    # STEP 3+4 — Notify
    subject, body = _build_email_content(status, completed_at, log_lines)
    target_emails = SUCCESS_EMAILS if status == "SUCCEEDED" else FAILED_EMAILS

    print(
        f"[{_now()}] Resolving notification channels for "
        f"{len(target_emails)} recipient(s) ..."
    )
    channel_ids = list_channel_ids_for_emails(session, creds, target_emails)

    if not channel_ids:
        print(
            f"[{_now()}] ERROR: No matching notification channels found in "
            f"project {PROJECT}. Email notifications will not be sent."
        )
    else:
        print(
            f"[{_now()}] Dispatching notifications via "
            f"{len(channel_ids)} channel(s) ..."
        )
        send_via_alert_policy(session, creds, channel_ids, subject, body, status)
        print(f"[{_now()}] Notifications dispatched")

    print(f"[{_now()}] === Done. Final status: {status} ===")
    sys.exit(0 if status == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()
