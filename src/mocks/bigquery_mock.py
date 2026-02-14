"""
Mock BigQuery service for local development and testing.
Provides a SQLite-backed implementation of BigQuery API.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
QUERY_COUNTER = Counter(
    'bigquery_mock_queries_total',
    'Total number of queries executed',
    ['dataset', 'table', 'status']
)
QUERY_DURATION = Histogram(
    'bigquery_mock_query_duration_seconds',
    'Query execution duration',
    ['dataset', 'table']
)
ROWS_PROCESSED = Counter(
    'bigquery_mock_rows_processed_total',
    'Total number of rows processed',
    ['dataset', 'table', 'operation']
)

# Initialize FastAPI app
app = FastAPI(
    title="Mock BigQuery Service",
    description="Local BigQuery emulator for F1 Strategy Optimizer",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    """BigQuery query request model"""
    query: str
    use_legacy_sql: bool = False
    max_results: Optional[int] = None
    timeout_ms: Optional[int] = 10000
    use_query_cache: bool = True


class QueryResponse(BaseModel):
    """BigQuery query response model"""
    job_id: str
    rows: List[Dict[str, Any]]
    total_rows: int
    schema: List[Dict[str, str]]
    query_complete: bool = True
    errors: Optional[List[str]] = None


class TableSchema(BaseModel):
    """BigQuery table schema model"""
    fields: List[Dict[str, Any]]


class InsertRequest(BaseModel):
    """BigQuery table insert request"""
    rows: List[Dict[str, Any]]
    skip_invalid_rows: bool = False
    ignore_unknown_values: bool = False


class MockBigQueryService:
    """Mock BigQuery service implementation"""

    def __init__(self, db_path: str = "/data/bigquery_mock.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_database()
        logger.info(f"Mock BigQuery service initialized with DB: {db_path}")

    def _init_database(self):
        """Initialize SQLite database and create F1 tables"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create F1 data tables
        cursor = self.conn.cursor()

        # Races table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS f1_data.races (
                race_id INTEGER PRIMARY KEY,
                year INTEGER,
                round INTEGER,
                circuit_id TEXT,
                name TEXT,
                date TEXT,
                time TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Drivers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS f1_data.drivers (
                driver_id TEXT PRIMARY KEY,
                driver_number INTEGER,
                code TEXT,
                forename TEXT,
                surname TEXT,
                dob TEXT,
                nationality TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS f1_data.results (
                result_id INTEGER PRIMARY KEY,
                race_id INTEGER,
                driver_id TEXT,
                constructor_id TEXT,
                grid INTEGER,
                position INTEGER,
                position_text TEXT,
                position_order INTEGER,
                points REAL,
                laps INTEGER,
                time TEXT,
                milliseconds INTEGER,
                fastest_lap INTEGER,
                rank INTEGER,
                fastest_lap_time TEXT,
                fastest_lap_speed REAL,
                status_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races(race_id),
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
            )
        """)

        # Lap times table (telemetry)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS f1_data.lap_times (
                lap_time_id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id INTEGER,
                driver_id TEXT,
                lap INTEGER,
                position INTEGER,
                time TEXT,
                milliseconds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races(race_id),
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
            )
        """)

        # Driver profiles table (ML features)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS f1_data.driver_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id TEXT,
                aggression_score REAL,
                consistency_score REAL,
                tire_management_score REAL,
                qualifying_pace REAL,
                race_pace REAL,
                overtaking_ability REAL,
                defensive_ability REAL,
                wet_weather_ability REAL,
                profile_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
            )
        """)

        self.conn.commit()
        logger.info("Database tables created successfully")

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            cursor = self.conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch all rows
            rows = cursor.fetchall()

            # Convert to list of dicts
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))

            ROWS_PROCESSED.labels(
                dataset='f1_data',
                table='query',
                operation='select'
            ).inc(len(result))

            return result

        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}")
            raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")

    def insert_rows(self, table: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert rows into a table"""
        try:
            cursor = self.conn.cursor()
            inserted = 0
            errors = []

            for row in rows:
                try:
                    # Build INSERT statement
                    columns = list(row.keys())
                    placeholders = ','.join(['?' for _ in columns])
                    column_names = ','.join(columns)

                    query = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"
                    cursor.execute(query, list(row.values()))
                    inserted += 1

                except sqlite3.Error as e:
                    errors.append(str(e))
                    logger.warning(f"Row insertion error: {e}")

            self.conn.commit()

            ROWS_PROCESSED.labels(
                dataset='f1_data',
                table=table.split('.')[-1],
                operation='insert'
            ).inc(inserted)

            return {
                "inserted": inserted,
                "errors": errors if errors else None
            }

        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            raise HTTPException(status_code=500, detail=f"Insert error: {str(e)}")

    def get_table_schema(self, table: str) -> List[Dict[str, str]]:
        """Get table schema"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            schema = []
            for col in columns:
                schema.append({
                    "name": col[1],
                    "type": col[2],
                    "nullable": "YES" if not col[3] else "NO",
                    "default": col[4]
                })

            return schema

        except sqlite3.Error as e:
            logger.error(f"Schema fetch error: {e}")
            raise HTTPException(status_code=404, detail=f"Table not found: {table}")

    def create_sample_data(self):
        """Create sample F1 data for testing"""
        sample_races = [
            {
                "race_id": 1,
                "year": 2024,
                "round": 1,
                "circuit_id": "bahrain",
                "name": "Bahrain Grand Prix",
                "date": "2024-03-02",
                "time": "15:00:00",
                "url": "http://en.wikipedia.org/wiki/2024_Bahrain_Grand_Prix"
            }
        ]

        sample_drivers = [
            {
                "driver_id": "max_verstappen",
                "driver_number": 1,
                "code": "VER",
                "forename": "Max",
                "surname": "Verstappen",
                "dob": "1997-09-30",
                "nationality": "Dutch",
                "url": "http://en.wikipedia.org/wiki/Max_Verstappen"
            },
            {
                "driver_id": "lewis_hamilton",
                "driver_number": 44,
                "code": "HAM",
                "forename": "Lewis",
                "surname": "Hamilton",
                "dob": "1985-01-07",
                "nationality": "British",
                "url": "http://en.wikipedia.org/wiki/Lewis_Hamilton"
            }
        ]

        self.insert_rows("f1_data.races", sample_races)
        self.insert_rows("f1_data.drivers", sample_drivers)

        logger.info("Sample data created successfully")


# Initialize service
mock_service = MockBigQueryService()


@app.on_event("startup")
async def startup_event():
    """Initialize sample data on startup"""
    try:
        mock_service.create_sample_data()
        logger.info("Mock BigQuery service started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mock-bigquery",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute BigQuery SQL query"""
    start_time = datetime.utcnow()

    try:
        with QUERY_DURATION.labels(dataset='f1_data', table='query').time():
            results = mock_service.execute_query(request.query)

        QUERY_COUNTER.labels(
            dataset='f1_data',
            table='query',
            status='success'
        ).inc()

        # Generate schema from first row
        schema = []
        if results:
            for key, value in results[0].items():
                schema.append({
                    "name": key,
                    "type": type(value).__name__
                })

        return QueryResponse(
            job_id=f"job_{datetime.utcnow().timestamp()}",
            rows=results,
            total_rows=len(results),
            schema=schema,
            query_complete=True
        )

    except Exception as e:
        QUERY_COUNTER.labels(
            dataset='f1_data',
            table='query',
            status='error'
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tables/{dataset}/{table}/insertAll")
async def insert_rows(dataset: str, table: str, request: InsertRequest):
    """Insert rows into BigQuery table"""
    try:
        full_table = f"{dataset}.{table}"
        result = mock_service.insert_rows(full_table, request.rows)

        return {
            "kind": "bigquery#tableDataInsertAllResponse",
            "insertErrors": result.get("errors")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables/{dataset}/{table}/schema")
async def get_schema(dataset: str, table: str):
    """Get table schema"""
    try:
        full_table = f"{dataset}.{table}"
        schema = mock_service.get_table_schema(full_table)

        return {
            "kind": "bigquery#table",
            "schema": {"fields": schema}
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(
        content=generate_latest().decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Mock BigQuery",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9050)
