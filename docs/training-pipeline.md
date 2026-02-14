# Distributed Training Pipeline Infrastructure

**Last Updated**: 2026-02-14

## Overview

The F1 Strategy Optimizer training pipeline executes ML model training as isolated, containerized stages within the existing data processing orchestration layer. Training jobs are horizontally scalable, fault-tolerant, and secured with least-privilege access controls.

**Key Principles**:
- Training is a pipeline stage, not a separate system
- Container-native execution (framework-agnostic)
- Reads from processed storage only
- Writes artifacts to dedicated storage
- Integrated with existing orchestration
- Hardware-agnostic (CPU/GPU abstracted)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│ Orchestration Layer (Vertex AI Pipelines)             │
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │ Ingest   │──>│ Process  │──>│ Validate │          │
│  │ Stage    │   │ Stage    │   │ Stage    │          │
│  └──────────┘   └──────────┘   └──────────┘          │
│                                      │                 │
│                                      ▼                 │
│                            ┌──────────────────┐       │
│                            │ Training Stage   │       │
│                            │  (Distributed)   │       │
│                            └──────────────────┘       │
│                                      │                 │
│                                      ▼                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │ Evaluate │──>│ Register │──>│  Serve   │          │
│  │ Stage    │   │ Stage    │   │  Stage   │          │
│  └──────────┘   └──────────┘   └──────────┘          │
└────────────────────────────────────────────────────────┘
```

**Non-Blocking Design**: Training stages run independently and do not block:
- Real-time inference pipelines
- Streaming data ingestion (Dataflow)
- Dashboard queries
- Other pipeline stages

---

## Container-Native Execution

### Training Worker Container

**Base Image**:
```dockerfile
# Dockerfile.training-worker
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (framework-agnostic base)
COPY requirements-training.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-training.txt

# Copy training code
COPY pipeline/training/ /app/training/
COPY models/ /app/models/

WORKDIR /app

# Entrypoint
ENTRYPOINT ["python", "-m", "training.worker"]
```

**Container Registry**:
- **Location**: Google Container Registry (GCR)
- **Image**: `gcr.io/f1-strategy/training-worker:latest`
- **Versioning**: Semantic versioning (v1.0.0, v1.1.0, etc.)
- **Tagging**: Git commit SHA for traceability

**Container Configuration**:
```yaml
container:
  image: gcr.io/f1-strategy/training-worker:v1.0.0
  command: ["python", "-m", "training.worker"]
  args:
    - --model-type=tire_degradation
    - --data-split=train
    - --output-uri=gs://f1-strategy-artifacts/models/

  env:
    - name: GOOGLE_CLOUD_PROJECT
      value: f1-strategy
    - name: TRAINING_JOB_ID
      value: "${JOB_ID}"
    - name: WORKER_INDEX
      value: "${WORKER_INDEX}"
    - name: TOTAL_WORKERS
      value: "${TOTAL_WORKERS}"

  resources:
    requests:
      cpu: "4"
      memory: "16Gi"
    limits:
      cpu: "8"
      memory: "32Gi"
```

---

## Distributed Training Architecture

### Horizontal Scaling

**Worker Pool Configuration**:
- **Minimum Workers**: 1 (single-node training for small models)
- **Maximum Workers**: 10 (distributed training for large models)
- **Default Workers**: 4 (cost-effective for typical training)
- **Scaling Policy**: Manual specification per job (not auto-scaled during training)

**Worker Coordination**:
```
┌─────────────────────────────────────────────────┐
│ Distributed Training Job                       │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │     │
│  │ (Chief)  │  │          │  │          │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │             │            │
│       └─────────────┼─────────────┘            │
│              HTTPS/TLS 1.3                     │
│             (Encrypted Transport)              │
│                                                 │
│  Data Sharding: Each worker processes subset   │
│  Model Sync: Parameter updates via secure RPC  │
│  Checkpoints: Coordinated write to GCS         │
└─────────────────────────────────────────────────┘
```

**Inter-Worker Communication**:
- **Protocol**: gRPC over HTTPS/TLS 1.3
- **Network**: Internal VPC only (no public internet)
- **Encryption**: All traffic encrypted end-to-end
- **Service Discovery**: Kubernetes DNS or Vertex AI internal service mesh

**Data Sharding Strategy**:
```python
# Example: Each worker processes a shard of the training data
def get_data_shard(worker_index: int, total_workers: int, dataset: str):
    """Shard data across workers for distributed training."""

    query = f"""
        SELECT *
        FROM f1_strategy.{dataset}
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(race_id AS STRING))), {total_workers}) = {worker_index}
    """

    # Each worker reads only its shard from BigQuery
    df = bq_client.query(query).to_dataframe()
    return df
```

**Fault Tolerance**:
- **Worker Failure**: Job continues if <50% workers fail
- **Chief Worker Failure**: Job restarts from last checkpoint
- **Network Partition**: Job paused, retried after 5 minutes
- **Checkpoint Frequency**: Every 10 minutes or 1000 iterations

---

## Infrastructure Provisioning (Terraform)

### Compute Resources

**File**: `infrastructure/terraform/training.tf`

```hcl
# Training compute resource pool
resource "google_compute_region_instance_group_manager" "training_workers" {
  name               = "f1-training-workers"
  base_instance_name = "training-worker"
  region             = var.region

  version {
    instance_template = google_compute_instance_template.training_worker.id
  }

  target_size = 0  # Scale to zero by default

  auto_healing_policies {
    health_check      = google_compute_health_check.training_worker.id
    initial_delay_sec = 300
  }
}

# Instance template for training workers
resource "google_compute_instance_template" "training_worker" {
  name         = "f1-training-worker-template-v1"
  machine_type = var.training_machine_type

  disk {
    source_image = "cos-cloud/cos-stable"
    auto_delete  = true
    boot         = true
    disk_size_gb = 100
  }

  # Additional disk for model checkpoints (optional)
  disk {
    auto_delete  = true
    boot         = false
    disk_size_gb = 200
    type         = "pd-ssd"
  }

  network_interface {
    network    = google_compute_network.training_vpc.id
    subnetwork = google_compute_subnetwork.training_subnet.id
    # No external IP - internal only
  }

  service_account {
    email  = google_service_account.training_worker.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    google-logging-enabled = "true"
    container-vm           = module.training_container.metadata
  }

  tags = ["training-worker", "no-internet"]
}

# Autoscaler (manual trigger, not auto-scaling during training)
resource "google_compute_region_autoscaler" "training_workers" {
  name   = "f1-training-autoscaler"
  region = var.region
  target = google_compute_region_instance_group_manager.training_workers.id

  autoscaling_policy {
    min_replicas = 0
    max_replicas = var.max_training_workers

    # Scale based on custom metric (jobs in queue)
    metric {
      name   = "custom.googleapis.com/training/jobs_queued"
      target = 1
      type   = "GAUGE"
    }

    scale_in_control {
      max_scaled_in_replicas {
        fixed = 2
      }
      time_window_sec = 600
    }
  }
}

# Health check
resource "google_compute_health_check" "training_worker" {
  name               = "training-worker-health"
  check_interval_sec = 60
  timeout_sec        = 30

  http_health_check {
    port         = 8080
    request_path = "/health"
  }
}
```

### Storage Resources

**File**: `infrastructure/terraform/storage.tf`

```hcl
# GCS bucket for training artifacts
resource "google_storage_bucket" "training_artifacts" {
  name          = "f1-strategy-artifacts"
  location      = var.region
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90  # Delete artifacts older than 90 days
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.artifact_encryption.id
  }
}

# Separate buckets for different artifact types
resource "google_storage_bucket" "model_registry" {
  name          = "f1-strategy-models"
  location      = var.region
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "training_checkpoints" {
  name          = "f1-strategy-checkpoints"
  location      = var.region
  storage_class = "NEARLINE"  # Cheaper for infrequent access

  lifecycle_rule {
    condition {
      age = 7  # Delete checkpoints after 7 days
    }
    action {
      type = "Delete"
    }
  }
}
```

### IAM and Security

**File**: `infrastructure/terraform/iam.tf`

```hcl
# Training worker service account
resource "google_service_account" "training_worker" {
  account_id   = "training-worker"
  display_name = "F1 Strategy Training Worker"
  description  = "Service account for distributed training jobs"
}

# BigQuery read-only access
resource "google_project_iam_member" "training_bq_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.training_worker.email}"

  condition {
    title      = "training-data-only"
    expression = "resource.name.startsWith('projects/${var.project_id}/datasets/f1_strategy')"
  }
}

resource "google_project_iam_member" "training_bq_jobuser" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.training_worker.email}"
}

# GCS write access to artifacts bucket
resource "google_storage_bucket_iam_member" "training_artifacts_writer" {
  bucket = google_storage_bucket.training_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.training_worker.email}"
}

resource "google_storage_bucket_iam_member" "training_artifacts_reader" {
  bucket = google_storage_bucket.training_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.training_worker.email}"
}

# Vertex AI access
resource "google_project_iam_member" "training_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.training_worker.email}"
}

# Explicitly deny dangerous permissions
resource "google_project_iam_member" "training_deny_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.training_worker.email}"

  # This is a deny policy (requires Organization Policy)
  condition {
    title      = "deny-data-modification"
    expression = "false"
  }
}
```

### Networking

**File**: `infrastructure/terraform/network.tf`

```hcl
# Dedicated VPC for training workloads
resource "google_compute_network" "training_vpc" {
  name                    = "f1-training-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "training_subnet" {
  name          = "f1-training-subnet"
  ip_cidr_range = "10.10.0.0/16"
  region        = var.region
  network       = google_compute_network.training_vpc.id

  private_ip_google_access = true  # Access GCP APIs without external IP
}

# Firewall: Allow internal traffic between workers
resource "google_compute_firewall" "training_internal" {
  name    = "training-internal-allow"
  network = google_compute_network.training_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  source_tags = ["training-worker"]
  target_tags = ["training-worker"]
}

# Firewall: Deny all external ingress
resource "google_compute_firewall" "training_deny_external" {
  name     = "training-deny-external"
  network  = google_compute_network.training_vpc.name
  priority = 1000

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["no-internet"]
}

# Cloud NAT (optional, for pulling container images)
resource "google_compute_router" "training_router" {
  name    = "f1-training-router"
  region  = var.region
  network = google_compute_network.training_vpc.id
}

resource "google_compute_router_nat" "training_nat" {
  name                               = "f1-training-nat"
  router                             = google_compute_router.training_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
```

---

## Orchestration Integration

### Pipeline Definition (Vertex AI Pipelines)

**File**: `pipeline/training/orchestrator.py`

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Artifact, Model, Metrics

@component(
    base_image="gcr.io/f1-strategy/data-validator:latest",
    packages_to_install=["google-cloud-bigquery"]
)
def validate_data(split: str, min_completeness: float = 0.95) -> bool:
    """Validate training data quality before training."""
    from google.cloud import bigquery

    client = bigquery.Client()

    # Check data completeness
    query = f"""
        SELECT
            COUNT(*) as total_rows,
            COUNTIF(lap_time_ms IS NULL) as null_laps,
            COUNTIF(tire_age IS NULL) as null_tires
        FROM f1_strategy.{split}
    """

    result = list(client.query(query).result())[0]
    completeness = 1 - (result.null_laps + result.null_tires) / (result.total_rows * 2)

    if completeness < min_completeness:
        raise ValueError(f"Data completeness {completeness:.2%} < {min_completeness:.2%}")

    return True

@component(
    base_image="gcr.io/f1-strategy/training-worker:latest"
)
def train_model(
    model_type: str,
    data_split: str,
    num_workers: int,
    output_model: Output[Model],
    output_metrics: Output[Metrics]
):
    """Execute distributed training job."""
    import os
    from training.worker import DistributedTrainer

    # Initialize trainer
    trainer = DistributedTrainer(
        model_type=model_type,
        data_split=data_split,
        num_workers=num_workers,
        worker_index=int(os.getenv('WORKER_INDEX', '0')),
        job_id=os.getenv('TRAINING_JOB_ID')
    )

    # Train model
    model_uri, metrics = trainer.run()

    # Write outputs
    output_model.uri = model_uri
    output_metrics.log_metric("final_loss", metrics['loss'])
    output_metrics.log_metric("final_accuracy", metrics['accuracy'])

@component(
    base_image="gcr.io/f1-strategy/evaluator:latest"
)
def evaluate_model(
    model: Input[Model],
    test_split: str,
    output_metrics: Output[Metrics]
) -> dict:
    """Evaluate trained model on test set."""
    from evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(model.uri)
    metrics = evaluator.evaluate(test_split)

    # Log metrics
    for key, value in metrics.items():
        output_metrics.log_metric(key, value)

    return metrics

@dsl.pipeline(
    name='f1-training-pipeline',
    description='Distributed training pipeline for F1 strategy models'
)
def training_pipeline(
    model_type: str,
    data_split: str = 'train',
    num_workers: int = 4,
    test_split: str = 'test'
):
    """Complete training pipeline with validation, training, and evaluation."""

    # Stage 1: Validate data quality
    validation_task = validate_data(
        split=data_split,
        min_completeness=0.95
    )

    # Stage 2: Distributed training
    training_task = train_model(
        model_type=model_type,
        data_split=data_split,
        num_workers=num_workers
    ).after(validation_task)

    # Stage 3: Evaluate on test set
    evaluation_task = evaluate_model(
        model=training_task.outputs['output_model'],
        test_split=test_split
    ).after(training_task)

    # Set resource requirements
    training_task.set_cpu_limit('16')
    training_task.set_memory_limit('64G')
    training_task.set_retry(3)
```

### Job Submission

```python
from google.cloud import aiplatform
from kfp.v2 import compiler

# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path='training_pipeline.json'
)

# Submit to Vertex AI Pipelines
aiplatform.init(
    project='f1-strategy',
    location='us-central1'
)

job = aiplatform.PipelineJob(
    display_name='tire-degradation-training',
    template_path='training_pipeline.json',
    parameter_values={
        'model_type': 'tire_degradation',
        'data_split': 'train',
        'num_workers': 4,
        'test_split': 'test'
    },
    enable_caching=False  # Always retrain
)

job.submit(
    service_account='training-worker@f1-strategy.iam.gserviceaccount.com'
)
```

---

## Failure Handling and Retries

### Retry Policy

```yaml
Retry Configuration:
  max_retries: 3
  retry_on:
    - worker_failure
    - network_timeout
    - resource_exhaustion

  no_retry_on:
    - data_validation_failure
    - permission_denied
    - invalid_configuration

  backoff_policy:
    initial_delay: 60s
    max_delay: 600s
    multiplier: 2.0
```

### Failure Scenarios

**Worker Failure**:
```
Action: Restart failed worker from last checkpoint
Timeout: 5 minutes before retry
Max Retries: 3
Fallback: Reduce num_workers by 1, retry
```

**Chief Worker Failure**:
```
Action: Restart entire job from last checkpoint
Timeout: 10 minutes
Max Retries: 2
Fallback: Notify on-call engineer
```

**Data Validation Failure**:
```
Action: Fail immediately (no retry)
Notification: Alert data team
Logs: Full validation report to Cloud Logging
```

**Out of Memory**:
```
Action: Retry with larger instance type
Instance Upgrade: n1-standard-16 → n1-highmem-16
Max Retries: 1
Fallback: Fail, notify ML team
```

**Network Partition**:
```
Action: Pause job, wait for network recovery
Timeout: 15 minutes
Recovery: Resume from last checkpoint
```

### Checkpoint Management

```python
# Checkpoint every 10 minutes or 1000 iterations
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, frequency_sec: int = 600):
        self.checkpoint_dir = checkpoint_dir
        self.frequency_sec = frequency_sec
        self.last_checkpoint_time = time.time()

    def should_checkpoint(self, iteration: int) -> bool:
        """Check if checkpoint should be saved."""
        elapsed = time.time() - self.last_checkpoint_time

        return (
            iteration % 1000 == 0 or
            elapsed >= self.frequency_sec
        )

    def save(self, model_state: dict, iteration: int):
        """Save checkpoint to GCS."""
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{iteration}.ckpt"

        # Save to GCS
        with gcs_client.open(checkpoint_path, 'wb') as f:
            pickle.dump(model_state, f)

        self.last_checkpoint_time = time.time()

    def load_latest(self) -> dict:
        """Load latest checkpoint from GCS."""
        checkpoints = gcs_client.list_blobs(self.checkpoint_dir)
        latest = max(checkpoints, key=lambda x: x.updated)

        with gcs_client.open(latest.name, 'rb') as f:
            return pickle.load(f)
```

---

## Monitoring and Observability

### Cloud Monitoring Metrics

**Job-Level Metrics**:
```yaml
training.job.duration_seconds:
  type: gauge
  labels: [model_type, job_id]
  description: Total training job duration

training.job.status:
  type: gauge
  labels: [model_type, status]  # status: running, succeeded, failed
  description: Training job status

training.job.retry_count:
  type: counter
  labels: [model_type, failure_reason]
  description: Number of job retries
```

**Worker-Level Metrics**:
```yaml
training.worker.cpu_utilization:
  type: gauge
  labels: [job_id, worker_index]
  description: Worker CPU utilization (0-100%)

training.worker.memory_usage_bytes:
  type: gauge
  labels: [job_id, worker_index]
  description: Worker memory usage

training.worker.heartbeat:
  type: gauge
  labels: [job_id, worker_index]
  description: Worker heartbeat timestamp (for liveness check)
```

**Training Progress Metrics**:
```yaml
training.progress.iterations_completed:
  type: counter
  labels: [job_id, model_type]
  description: Training iterations completed

training.progress.records_processed:
  type: counter
  labels: [job_id, worker_index]
  description: Training records processed
```

### Alerting

**File**: `monitoring/alerts.yaml`

```yaml
alerts:
  - name: training-job-timeout
    condition: training.job.duration_seconds > 43200  # 12 hours
    severity: warning
    notification: slack

  - name: training-job-failure-rate
    condition: rate(training.job.status{status="failed"}[1h]) > 0.2
    severity: critical
    notification: pagerduty

  - name: worker-oom
    condition: training.worker.memory_usage_bytes / limits.memory > 0.95
    severity: warning
    action: auto-retry with larger instance

  - name: data-validation-failure
    condition: training.job.status{status="failed", reason="validation"}
    severity: critical
    notification: slack
    action: block future training jobs
```

### Logging

**Structured Logging**:
```python
import logging
from google.cloud import logging as cloud_logging

# Initialize Cloud Logging
logging_client = cloud_logging.Client()
logging_client.setup_logging()

# Structured log entry
logging.info(
    "Training started",
    extra={
        "job_id": os.getenv("TRAINING_JOB_ID"),
        "model_type": "tire_degradation",
        "num_workers": 4,
        "worker_index": 0,
        "data_split": "train"
    }
)
```

**Log Retention**:
- **Training Logs**: 30 days
- **Error Logs**: 90 days
- **Audit Logs**: 1 year

---

## Cost Management

### Budget Allocation

```yaml
Training Budget (Monthly):
  weekly_retraining: $80-120      # 4 models × $20-30 per week
  ad_hoc_experiments: $50         # Manual triggered jobs
  infrastructure: $30             # Storage, networking
  total: $160-200/month
```

### Cost Optimization Strategies

**1. Preemptible VMs**:
```hcl
resource "google_compute_instance_template" "training_worker_preemptible" {
  name         = "training-worker-preemptible"
  machine_type = var.training_machine_type

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  # 70% cost reduction vs standard VMs
}
```

**2. Scale to Zero**:
- Worker pool scales to 0 when no jobs running
- No cost during idle periods
- Cold start: ~2-3 minutes to provision workers

**3. Storage Lifecycle**:
- Checkpoints deleted after 7 days
- Artifacts moved to Nearline after 30 days
- Old models deleted after 90 days

**4. Resource Right-Sizing**:
```python
# Automatically adjust instance type based on model size
def select_instance_type(model_type: str) -> str:
    INSTANCE_TYPES = {
        'tire_degradation': 'n1-standard-8',   # Medium model
        'fuel_consumption': 'n1-highmem-16',   # Large LSTM model
        'brake_bias': 'n1-standard-4',         # Small linear model
        'driving_style': 'n1-standard-8'       # Medium tree model
    }
    return INSTANCE_TYPES.get(model_type, 'n1-standard-8')
```

### Cost Monitoring

```sql
-- BigQuery query for training costs
SELECT
  DATE(usage_start_time) as date,
  service.description as service,
  SUM(cost) as total_cost
FROM `f1-strategy.billing.gcp_billing_export_v1`
WHERE
  project.id = 'f1-strategy'
  AND (
    service.description LIKE '%Compute%'
    OR service.description LIKE '%Vertex AI%'
    OR service.description LIKE '%Storage%'
  )
  AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY date, service
ORDER BY date DESC;
```

---

## Security Hardening

### Principle of Least Privilege

**Training Service Account Permissions** (Minimal):
```
READ:
  ✓ BigQuery: f1_strategy.train, f1_strategy.validation
  ✓ GCS: gs://f1-strategy-models/* (read trained models)

WRITE:
  ✓ GCS: gs://f1-strategy-artifacts/* (write new artifacts)

DENIED:
  ✗ BigQuery: raw_*, processed_data (cannot modify source data)
  ✗ GCS: Delete operations on production buckets
  ✗ Compute: Create/delete infrastructure
  ✗ IAM: Modify service accounts or permissions
```

### Network Isolation

```
Training VPC:
  ├── No external IP addresses
  ├── Private Google Access enabled
  ├── Firewall: Block all ingress from internet
  ├── Firewall: Allow internal worker-to-worker traffic
  └── Cloud NAT: Only for pulling container images
```

### Secrets Management

```python
# Use Secret Manager for sensitive credentials
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    """Fetch secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/f1-strategy/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode('UTF-8')

# Never hardcode credentials
api_key = get_secret('external-api-key')
```

### Audit Logging

```yaml
Audit Events:
  - data_access: Log all BigQuery queries from training workers
  - admin_activity: Log all GCS writes
  - system_event: Log worker start/stop events

Retention: 1 year
Storage: Cloud Logging + exported to BigQuery
```

---

## Production Readiness Checklist

### Pre-Deployment

- [ ] Terraform infrastructure deployed and tested
- [ ] Service accounts created with least-privilege IAM
- [ ] Training container image built and pushed to GCR
- [ ] VPC and firewall rules configured
- [ ] Data validation pipeline tested on sample data
- [ ] Checkpoint/restore logic tested
- [ ] Monitoring dashboards configured
- [ ] Alerting rules deployed

### Post-Deployment

- [ ] Execute dry-run training job (small dataset)
- [ ] Verify artifacts written to GCS
- [ ] Validate metrics exported to Cloud Monitoring
- [ ] Test failure scenarios (worker crash, OOM, etc.)
- [ ] Verify cost tracking and budgets
- [ ] Conduct security review (IAM, network, secrets)
- [ ] Document runbook for on-call engineers

---

**See Also**:
- CLAUDE.md: High-level project overview
- docs/data.md: Training data sources and schemas
- docs/architecture.md: Overall system design
- docs/monitoring.md: Operational monitoring details
- infrastructure/terraform/: Infrastructure as code
