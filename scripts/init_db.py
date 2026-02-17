#!/usr/bin/env python3
"""
Initialize database for F1 Strategy Optimizer
Creates tables, indexes, and sample data for local development
"""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_airflow_db():
    """Initialize Airflow metadata database"""
    logger.info("Initializing Airflow database...")

    import subprocess
    try:
        # Initialize Airflow database
        subprocess.run(['airflow', 'db', 'init'], check=True)

        # Create admin user
        subprocess.run([
            'airflow', 'users', 'create',
            '--username', 'admin',
            '--firstname', 'Admin',
            '--lastname', 'User',
            '--role', 'Admin',
            '--email', 'admin@f1optimizer.local',
            '--password', 'admin'
        ], check=True)

        logger.info("✅ Airflow database initialized")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to initialize Airflow database: {e}")
    except FileNotFoundError:
        logger.warning("⚠️ Airflow not found - skipping Airflow DB initialization")


def init_mock_bigquery():
    """Initialize mock BigQuery database"""
    logger.info("Initializing mock BigQuery database...")

    db_path = Path("/data/bigquery_mock.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables
    tables = [
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
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
        """
    ]

    for table_sql in tables:
        cursor.execute(table_sql)

    # Insert sample data
    sample_races = [
        (1, 2024, 1, 'bahrain', 'Bahrain Grand Prix', '2024-03-02', '15:00:00',
         'http://en.wikipedia.org/wiki/2024_Bahrain_Grand_Prix')
    ]

    sample_drivers = [
        ('max_verstappen', 1, 'VER', 'Max', 'Verstappen', '1997-09-30', 'Dutch',
         'http://en.wikipedia.org/wiki/Max_Verstappen'),
        ('lewis_hamilton', 44, 'HAM', 'Lewis', 'Hamilton', '1985-01-07', 'British',
         'http://en.wikipedia.org/wiki/Lewis_Hamilton'),
        ('charles_leclerc', 16, 'LEC', 'Charles', 'Leclerc', '1997-10-16', 'Monegasque',
         'http://en.wikipedia.org/wiki/Charles_Leclerc')
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO f1_data.races VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        sample_races
    )

    cursor.executemany(
        "INSERT OR IGNORE INTO f1_data.drivers VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        sample_drivers
    )

    conn.commit()
    conn.close()

    logger.info("✅ Mock BigQuery database initialized")


def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")

    directories = [
        "/data",
        "/app/models",
        "/tmp/fastf1_cache",
        "airflow/logs",
        "airflow/dags",
        "airflow/plugins",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.info("✅ Directories created")


def verify_environment():
    """Verify environment setup"""
    logger.info("Verifying environment...")

    checks = {
        "Python version": True,
        "Docker Compose": False,
        "PostgreSQL": False,
        "Mock services": False
    }

    # Check Python
    import sys
    if sys.version_info >= (3, 10):
        logger.info("✅ Python 3.10+ detected")
        checks["Python version"] = True
    else:
        logger.warning(f"⚠️ Python {sys.version_info.major}.{sys.version_info.minor} detected, 3.10+ recommended")

    # Check Docker Compose
    import subprocess
    try:
        subprocess.run(['docker-compose', '--version'],
                      capture_output=True, check=True)
        logger.info("✅ Docker Compose available")
        checks["Docker Compose"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("⚠️ Docker Compose not found")

    return all(checks.values())


def main():
    """Main initialization function"""
    logger.info("=" * 60)
    logger.info("F1 Strategy Optimizer - Database Initialization")
    logger.info("=" * 60)

    # Verify environment
    verify_environment()

    # Create directories
    create_directories()

    # Initialize mock BigQuery
    init_mock_bigquery()

    # Initialize Airflow (optional - only if Airflow is installed)
    # init_airflow_db()

    logger.info("=" * 60)
    logger.info("✅ Initialization complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start Docker services: docker-compose -f docker-compose.f1.yml up -d")
    logger.info("2. Access Airflow UI: http://localhost:8080 (admin/admin)")
    logger.info("3. Access API docs: http://localhost:8000/docs")
    logger.info("4. Access Grafana: http://localhost:3000 (admin/admin)")
    logger.info("")


if __name__ == "__main__":
    main()
