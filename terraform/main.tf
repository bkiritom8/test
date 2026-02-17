/**
 * F1 Strategy Optimizer - Main Terraform Configuration
 * Provisions GCP infrastructure for production deployment
 */

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "f1-optimizer-terraform-state"
    prefix = "terraform/state"
  }
}

# Configure GCP provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Local variables
locals {
  common_labels = {
    project     = "f1-strategy-optimizer"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "pubsub.googleapis.com",
    "dataflow.googleapis.com",
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "secretmanager.googleapis.com"
  ])

  service            = each.value
  disable_on_destroy = false
}

# Pub/Sub Topics and Subscriptions
module "pubsub" {
  source = "./modules/pubsub"

  project_id  = var.project_id
  environment = var.environment

  topics = [
    "f1-race-events",
    "f1-telemetry-stream",
    "f1-predictions",
    "f1-alerts"
  ]

  labels = local.common_labels

  depends_on = [google_project_service.required_apis]
}

# Dataflow Jobs
module "dataflow" {
  source = "./modules/dataflow"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  temp_location    = "gs://${var.project_id}-dataflow-temp"
  staging_location = "gs://${var.project_id}-dataflow-staging"

  labels = local.common_labels

  depends_on = [google_project_service.required_apis]
}

# Cloud Run Services
module "cloud_run" {
  source = "./modules/compute"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  service_name    = "f1-strategy-api"
  container_image = "${var.region}-docker.pkg.dev/${var.project_id}/f1-optimizer/api:latest"
  max_instances   = var.api_max_instances
  min_instances   = var.api_min_instances
  memory          = "512Mi"
  cpu             = "1"
  timeout_seconds = 60

  env_vars = {
    ENV                = var.environment
    CLOUD_SQL_INSTANCE = google_sql_database_instance.f1_db.connection_name
    PUBSUB_PROJECT_ID  = var.project_id
    ENABLE_HTTPS       = "true"
    ENABLE_IAM         = "true"
    LOG_LEVEL          = "INFO"
  }

  labels = local.common_labels

  depends_on = [google_project_service.required_apis]
}

# VPC Network
module "networking" {
  source = "./modules/networking"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  network_name = "f1-optimizer-network"

  labels = local.common_labels

  depends_on = [google_project_service.required_apis]
}

# IAM Service Accounts
resource "google_service_account" "airflow_sa" {
  account_id   = "f1-airflow-${var.environment}"
  display_name = "F1 Airflow Service Account (${var.environment})"
  description  = "Service account for Airflow DAG execution"
}

resource "google_service_account" "dataflow_sa" {
  account_id   = "f1-dataflow-${var.environment}"
  display_name = "F1 Dataflow Service Account (${var.environment})"
  description  = "Service account for Dataflow job execution"
}

resource "google_service_account" "api_sa" {
  account_id   = "f1-api-${var.environment}"
  display_name = "F1 API Service Account (${var.environment})"
  description  = "Service account for Cloud Run API service"
}

# IAM Bindings
resource "google_project_iam_member" "airflow_pubsub_admin" {
  project = var.project_id
  role    = "roles/pubsub.admin"
  member  = "serviceAccount:${google_service_account.airflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_worker" {
  project = var.project_id
  role    = "roles/dataflow.worker"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

# Cloud Storage Buckets
resource "google_storage_bucket" "data_lake" {
  name          = "${var.project_id}-data-lake"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = local.common_labels
}

resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-models"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }

  labels = local.common_labels
}

# Cloud SQL (PostgreSQL) — operational data store replacing BigQuery
resource "google_sql_database_instance" "f1_db" {
  name                = "f1-optimizer-${var.environment}"
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = false

  settings {
    tier = "db-f1-micro"

    backup_configuration {
      enabled = true
    }

    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }
  }

  labels = local.common_labels

  depends_on = [google_project_service.required_apis]
}

resource "google_sql_database" "f1_data" {
  name     = "f1_data"
  instance = google_sql_database_instance.f1_db.name
  project  = var.project_id
}

resource "google_sql_user" "api_user" {
  name     = "f1_api"
  instance = google_sql_database_instance.f1_db.name
  project  = var.project_id
  password = var.db_password
}

resource "google_project_iam_member" "api_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api_sa.email}"
}

# Monitoring and Logging
resource "google_monitoring_notification_channel" "email" {
  display_name = "F1 Optimizer Email Alerts"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

resource "google_monitoring_alert_policy" "api_error_rate" {
  display_name = "F1 API High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "API Error Rate > 5%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

# Outputs
output "cloud_sql_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.f1_db.name
}

output "api_service_url" {
  description = "Cloud Run API service URL"
  value       = module.cloud_run.service_url
}

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value       = module.pubsub.topic_names
}

output "service_accounts" {
  description = "Service account emails"
  value = {
    airflow  = google_service_account.airflow_sa.email
    dataflow = google_service_account.dataflow_sa.email
    api      = google_service_account.api_sa.email
  }
}
