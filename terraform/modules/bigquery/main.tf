/**
 * BigQuery Module
 * Creates F1 data warehouse with partitioned tables
 */

resource "google_bigquery_dataset" "f1_data" {
  dataset_id  = var.dataset_id
  project     = var.project_id
  location    = var.location
  description = "F1 historical and telemetry data (${var.environment})"

  default_table_expiration_ms = 0  # No automatic expiration

  labels = var.labels
}

# Races table
resource "google_bigquery_table" "races" {
  dataset_id = google_bigquery_dataset.f1_data.dataset_id
  table_id   = "races"
  project    = var.project_id

  description = "F1 race metadata (1950-2024)"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["year", "circuit_id"]

  schema = jsonencode([
    {
      name = "race_id"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "year"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "round"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "circuit_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "time"
      type = "TIME"
      mode = "NULLABLE"
    },
    {
      name = "url"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      default_value_expression = "CURRENT_TIMESTAMP()"
    }
  ])

  labels = var.labels
}

# Drivers table
resource "google_bigquery_table" "drivers" {
  dataset_id = google_bigquery_dataset.f1_data.dataset_id
  table_id   = "drivers"
  project    = var.project_id

  description = "F1 driver information"

  clustering = ["nationality", "driver_id"]

  schema = jsonencode([
    {
      name = "driver_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "driver_number"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "code"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "forename"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "surname"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "dob"
      type = "DATE"
      mode = "NULLABLE"
    },
    {
      name = "nationality"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "url"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      default_value_expression = "CURRENT_TIMESTAMP()"
    }
  ])

  labels = var.labels
}

# Telemetry table (partitioned and clustered for performance)
resource "google_bigquery_table" "telemetry" {
  dataset_id = google_bigquery_dataset.f1_data.dataset_id
  table_id   = "telemetry"
  project    = var.project_id

  description = "High-frequency telemetry data (2018+)"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["race_id", "driver_id", "lap"]

  schema = jsonencode([
    {
      name = "race_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "driver_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "lap"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "speed"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "throttle"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "brake"
      type = "BOOLEAN"
      mode = "NULLABLE"
    },
    {
      name = "gear"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "rpm"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "drs"
      type = "INTEGER"
      mode = "NULLABLE"
    }
  ])

  labels = var.labels
}

# Outputs
output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.f1_data.dataset_id
}

output "dataset_name" {
  description = "BigQuery dataset name"
  value       = google_bigquery_dataset.f1_data.friendly_name
}

output "tables" {
  description = "Created BigQuery tables"
  value = {
    races      = google_bigquery_table.races.table_id
    drivers    = google_bigquery_table.drivers.table_id
    telemetry  = google_bigquery_table.telemetry.table_id
  }
}
