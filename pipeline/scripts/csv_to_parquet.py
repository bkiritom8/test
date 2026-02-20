"""
csv_to_parquet.py — Convert local CSV files to Parquet and upload to GCS.

Usage:
    python pipeline/scripts/csv_to_parquet.py \
        --input-dir /path/to/csvs \
        --gcs-prefix gs://f1optimizer-data-lake/processed/

Each CSV in --input-dir is converted to Parquet and uploaded as
<gcs-prefix><stem>.parquet (e.g. races.csv → processed/races.parquet).

Requirements: pandas, pyarrow, google-cloud-storage (all in requirements-ml.txt)
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_gcs_path(gcs_path: str):
    """Split gs://bucket/prefix into (bucket, prefix)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected gs:// path, got: {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def convert_and_upload(input_dir: str, gcs_prefix: str) -> None:
    csv_dir = Path(input_dir)
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", input_dir)
        return

    bucket_name, prefix = _parse_gcs_path(gcs_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for csv_path in csv_files:
        stem = csv_path.stem
        gcs_blob_name = f"{prefix}{stem}.parquet".lstrip("/")
        tmp_parquet = Path(f"/tmp/{stem}.parquet")

        logger.info("Reading %s", csv_path)
        df = pd.read_csv(csv_path)

        logger.info("Writing Parquet (%d rows, %d cols)", len(df), len(df.columns))
        df.to_parquet(tmp_parquet, index=False, engine="pyarrow")

        logger.info("Uploading to gs://%s/%s", bucket_name, gcs_blob_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(str(tmp_parquet))

        tmp_parquet.unlink(missing_ok=True)
        logger.info("Done: gs://%s/%s", bucket_name, gcs_blob_name)

    logger.info("Converted and uploaded %d file(s)", len(csv_files))


def main():
    parser = argparse.ArgumentParser(description="Convert CSVs to Parquet and upload to GCS")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Local directory containing CSV files",
    )
    parser.add_argument(
        "--gcs-prefix",
        required=True,
        help="GCS destination prefix, e.g. gs://f1optimizer-data-lake/processed/",
    )
    args = parser.parse_args()
    convert_and_upload(args.input_dir, args.gcs_prefix)


if __name__ == "__main__":
    main()
