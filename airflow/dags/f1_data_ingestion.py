"""
F1 Data Ingestion DAG with Operational Guarantees

Features:
- Task-level failure isolation
- Comprehensive logging and metrics
- Cost tracking and budget controls
- Retry logic with exponential backoff
- SLA monitoring and alerting
- Operational observability
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import psycopg2

from airflow import DAG  # type: ignore[attr-defined]
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException
from prometheus_client import Counter, Histogram, Gauge

# Import custom modules
import sys

sys.path.insert(0, "/opt/airflow/src")

from ingestion.ergast_ingestion import run_ingestion as _ergast_run
from ingestion.fastf1_ingestion import run_ingestion as _fastf1_run

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
DAG_RUN_DURATION = Histogram(
    "f1_dag_run_duration_seconds", "DAG run duration in seconds", ["dag_id", "status"]
)
TASK_DURATION = Histogram(
    "f1_task_duration_seconds",
    "Task execution duration",
    ["dag_id", "task_id", "status"],
)
TASK_RETRY_COUNT = Counter(
    "f1_task_retries_total", "Total task retries", ["dag_id", "task_id"]
)
DATA_RECORDS_INGESTED = Counter(
    "f1_data_records_ingested_total",
    "Total data records ingested",
    ["data_source", "data_type"],
)
DAG_COST = Gauge("f1_dag_cost_usd", "Estimated DAG run cost in USD", ["dag_id"])

# Cost tracking (simplified estimates)
COST_PER_TASK_RUN = 0.01  # $0.01 per task execution
COST_PER_GB_DATA = 0.05  # $0.05 per GB processed
MONTHLY_BUDGET_USD = 200.0


class OperationalMetrics:
    """Track operational metrics for DAG runs"""

    def __init__(self, dag_id: str, run_id: str):
        self.dag_id = dag_id
        self.run_id = run_id
        self.start_time = datetime.now()
        self.task_metrics: Dict[str, Any] = {}
        self.total_cost = 0.0
        self.records_processed = 0

    def record_task_start(self, task_id: str):
        """Record task start time"""
        self.task_metrics[task_id] = {"start_time": datetime.now(), "status": "running"}
        logger.info(f"[{self.dag_id}] Task {task_id} started")

    def record_task_end(self, task_id: str, status: str, records: int = 0):
        """Record task completion"""
        if task_id not in self.task_metrics:
            return

        end_time = datetime.now()
        duration = (end_time - self.task_metrics[task_id]["start_time"]).total_seconds()

        self.task_metrics[task_id].update(
            {
                "end_time": end_time,
                "duration": duration,
                "status": status,
                "records": records,
            }
        )

        # Update Prometheus metrics
        TASK_DURATION.labels(
            dag_id=self.dag_id, task_id=task_id, status=status
        ).observe(duration)

        # Update cost
        task_cost = COST_PER_TASK_RUN
        self.total_cost += task_cost
        DAG_COST.labels(dag_id=self.dag_id).set(self.total_cost)

        # Update records
        self.records_processed += records

        logger.info(
            f"[{self.dag_id}] Task {task_id} {status} - "
            f"Duration: {duration:.2f}s, Records: {records}, Cost: ${task_cost:.4f}"
        )

    def check_budget(self) -> bool:
        """Check if budget threshold exceeded"""
        if self.total_cost > (MONTHLY_BUDGET_USD / 30):  # Daily budget
            logger.warning(
                f"[{self.dag_id}] Daily budget threshold exceeded: "
                f"${self.total_cost:.2f} > ${MONTHLY_BUDGET_USD/30:.2f}"
            )
            return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get DAG run summary"""
        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "dag_id": self.dag_id,
            "run_id": self.run_id,
            "total_duration": duration,
            "total_cost": self.total_cost,
            "records_processed": self.records_processed,
            "task_count": len(self.task_metrics),
            "failed_tasks": sum(
                1 for t in self.task_metrics.values() if t.get("status") == "failed"
            ),
        }


# Global metrics instance (will be created per DAG run)
metrics: OperationalMetrics = None


def initialize_metrics(**context):
    """Initialize operational metrics for DAG run"""
    global metrics
    dag_id = context["dag"].dag_id
    run_id = context["run_id"]
    metrics = OperationalMetrics(dag_id, run_id)
    logger.info(f"Initialized metrics for DAG run {run_id}")


def ingest_ergast_data(**context):
    """Ingest Ergast data (seasons, drivers, constructors, races) into Cloud SQL."""
    global metrics
    task_id = context["task"].task_id
    metrics.record_task_start(task_id)

    try:
        current_year = datetime.now().year
        _ergast_run(
            start_season=current_year - 1,
            end_season=current_year,
            worker_id="airflow-ergast",
        )

        DATA_RECORDS_INGESTED.labels(data_source="ergast", data_type="all").inc(1)

        metrics.record_task_end(task_id, "success")
        logger.info(
            "Ergast ingestion complete for %d-%d", current_year - 1, current_year
        )
        return {"status": "ok", "seasons": f"{current_year - 1}-{current_year}"}

    except Exception as e:
        metrics.record_task_end(task_id, "failed")
        logger.error(f"Failed to ingest Ergast data: {e}")
        raise AirflowException(f"Ergast ingestion failed: {e}")


def ingest_fastf1_data(**context):
    """Ingest FastF1 telemetry (lap features, weather, driver profiles) into Cloud SQL."""
    global metrics
    task_id = context["task"].task_id
    metrics.record_task_start(task_id)

    try:
        current_year = datetime.now().year
        _fastf1_run(
            start_year=current_year - 1,
            end_year=current_year,
            worker_id="airflow-fastf1",
        )

        DATA_RECORDS_INGESTED.labels(data_source="fastf1", data_type="all").inc(1)

        metrics.record_task_end(task_id, "success")
        logger.info(
            "FastF1 ingestion complete for %d-%d", current_year - 1, current_year
        )
        return {"status": "ok", "years": f"{current_year - 1}-{current_year}"}

    except Exception as e:
        metrics.record_task_end(task_id, "failed")
        logger.error(f"Failed to ingest FastF1 data: {e}")
        raise AirflowException(f"FastF1 ingestion failed: {e}")


def verify_races_written(**context):
    """Verify race data was written to Cloud SQL after ingestion"""
    host = os.environ["DB_HOST"]
    dbname = os.environ["DB_NAME"]
    port = int(os.environ.get("DB_PORT", 5432))
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]

    try:
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            port=port,
            user=user,
            password=password,
            connect_timeout=10,
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM races")
                count = cur.fetchone()[0]
        conn.close()
    except psycopg2.Error as e:
        logger.error(f"Cloud SQL connection/query failed: {e}")
        raise AirflowException(f"Cloud SQL verification failed: {e}")

    logger.info(f"Cloud SQL races table row count: {count}")

    if count == 0:
        raise AirflowException(
            "Verification failed: races table is empty after ingestion"
        )

    return {"races_count": count}


def check_budget_threshold(**context):
    """Check if budget threshold exceeded"""
    global metrics

    if not metrics.check_budget():
        logger.warning("Budget threshold exceeded - consider throttling")
        # In production, could trigger alerts or pause non-critical tasks
        # For now, just log the warning
        pass

    summary = metrics.get_summary()
    logger.info(f"DAG run summary: {summary}")

    return summary


def publish_metrics(**context):
    """Publish final DAG run metrics"""
    global metrics
    summary = metrics.get_summary()

    # Update Prometheus metrics
    DAG_RUN_DURATION.labels(
        dag_id=summary["dag_id"],
        status="success" if summary["failed_tasks"] == 0 else "partial_failure",
    ).observe(summary["total_duration"])

    logger.info(f"Published metrics for DAG run {summary['run_id']}")
    return summary


# Default DAG arguments
default_args = {
    "owner": "f1-strategy-optimizer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=30),
}

# Create DAG
dag = DAG(
    dag_id="f1_data_ingestion",
    default_args=default_args,
    description="Ingest F1 data from Ergast and FastF1 with operational guarantees",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["f1", "data-ingestion", "production"],
    doc_md=__doc__,
    sla_miss_callback=lambda context: logger.warning(
        f"SLA missed for task {context['task'].task_id}"
    ),
)

with dag:
    # Start task
    start = EmptyOperator(task_id="start", dag=dag)

    # Initialize metrics
    init_metrics = PythonOperator(
        task_id="initialize_metrics",
        python_callable=initialize_metrics,
        provide_context=True,
        dag=dag,
    )

    # Ergast ingestion task — writes directly to Cloud SQL (recent 2 seasons)
    with TaskGroup(
        "ergast_ingestion", tooltip="Ingest data from Ergast API"
    ) as ergast_group:
        ingest_ergast = PythonOperator(
            task_id="ingest_ergast_data",
            python_callable=ingest_ergast_data,
            provide_context=True,
            execution_timeout=timedelta(hours=2),
            dag=dag,
        )

    # FastF1 ingestion task — writes directly to Cloud SQL (recent 2 seasons)
    with TaskGroup(
        "fastf1_ingestion", tooltip="Ingest data from FastF1"
    ) as fastf1_group:
        ingest_fastf1 = PythonOperator(
            task_id="ingest_fastf1_data",
            python_callable=ingest_fastf1_data,
            provide_context=True,
            execution_timeout=timedelta(hours=2),
            dag=dag,
        )

    # Verify data was written to Cloud SQL
    verify_cloud_sql = PythonOperator(
        task_id="verify_cloud_sql_races",
        python_callable=verify_races_written,
        provide_context=True,
        dag=dag,
    )

    # Budget check
    check_budget = PythonOperator(
        task_id="check_budget_threshold",
        python_callable=check_budget_threshold,
        provide_context=True,
        dag=dag,
    )

    # Publish metrics
    publish = PythonOperator(
        task_id="publish_metrics",
        python_callable=publish_metrics,
        provide_context=True,
        trigger_rule="all_done",  # Run even if upstream tasks failed
        dag=dag,
    )

    # End task
    end = EmptyOperator(task_id="end", trigger_rule="all_done", dag=dag)

    # Define DAG structure
    (
        start
        >> init_metrics
        >> [ergast_group, fastf1_group]
        >> verify_cloud_sql
        >> check_budget
        >> publish
        >> end
    )
