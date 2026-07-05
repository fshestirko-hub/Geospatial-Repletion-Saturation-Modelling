"""Add synthetic coordinates to the telemetry stream and export GeoJSON batches."""

import json
import logging
import math
import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    cos,
    lit,
    pmod,
    radians,
    sin,
    when,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# fallback anchor in central Vienna if no per-agent anchors are provided
VIENNA_LAT = 48.20849
VIENNA_LON = 16.37208
METRES_PER_DEGREE_LAT = 111_320.0
METRES_PER_DEGREE_LON = 111_320.0 * math.cos(math.radians(VIENNA_LAT))
BASE_TIMESTAMP_MS = 1_700_000_000_000


def add_synthetic_coordinates(
    telemetry_df: DataFrame,
    anchors_df: DataFrame | None = None,
) -> DataFrame:
    """Add lat/lon columns. Uses per-agent anchors when provided."""

    elapsed_seconds = (col("Timestamp") - lit(BASE_TIMESTAMP_MS)) / lit(1000.0)

    speed_mps = (
        when(col("Activity") == "walk", lit(1.4))
        .when(col("Activity") == "bike", lit(4.5))
        .otherwise(lit(0.0))
    )

    if anchors_df is not None:
        positioned_df = telemetry_df.join(anchors_df, on="Agent_ID", how="left")
        heading_radians = col("heading_radians")
        start_lat = col("start_lat")
        start_lon = col("start_lon")
    else:
        positioned_df = telemetry_df
        heading_radians = pmod(col("Agent_ID"), lit(16)) * lit(2.0 * math.pi / 16.0)
        start_lat = lit(VIENNA_LAT)
        start_lon = lit(VIENNA_LON)

    distance_metres = elapsed_seconds * speed_mps
    east_metres = distance_metres * cos(heading_radians)
    north_metres = distance_metres * sin(heading_radians)
    metres_per_degree_lon = lit(METRES_PER_DEGREE_LAT) * cos(
        radians(start_lat)
    )

    return (
        positioned_df
        .withColumn("Latitude", start_lat + north_metres / lit(METRES_PER_DEGREE_LAT))
        .withColumn("Longitude", start_lon + east_metres / metres_per_degree_lon)
    )


def _write_geojson_batch(batch_df: DataFrame, batch_id: int, output_dir: Path) -> None:
    """Write one Spark micro-batch as one GeoJSON FeatureCollection file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"telemetry_batch_{batch_id:05d}.geojson"
    temporary_path = output_path.with_suffix(".geojson.tmp")

    ordered_df = batch_df.select(
        "Agent_ID",
        "Timestamp",
        "Activity",
        "Latitude",
        "Longitude",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
    ).orderBy("Agent_ID", "Timestamp")

    with temporary_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write('{"type":"FeatureCollection","features":[')
        first_feature = True

        for row in ordered_df.toLocalIterator():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["Longitude"]), float(row["Latitude"])],
                },
                "properties": {
                    "Agent_ID": int(row["Agent_ID"]),
                    "Timestamp": int(row["Timestamp"]),
                    "Activity": str(row["Activity"]),
                    "ax": float(row["ax"]),
                    "ay": float(row["ay"]),
                    "az": float(row["az"]),
                    "gx": float(row["gx"]),
                    "gy": float(row["gy"]),
                    "gz": float(row["gz"]),
                    "coordinate_source": "synthetic",
                },
            }

            if not first_feature:
                file_handle.write(",")
            json.dump(feature, file_handle, separators=(",", ":"))
            first_feature = False

        file_handle.write("]}")

    temporary_path.replace(output_path)
    logging.info("GeoJSON micro-batch written to %s", output_path)


def run_geospatial_streaming_simulation(spark: SparkSession, project_root: Path) -> Path:
    """Read synthetic telemetry as Spark micro-batches and export GeoJSON files.

    Returns the output directory containing the generated GeoJSON files.
    """

    project_root = Path(project_root)
    processed_dir = project_root / "data" / "processed"
    master_parquet_path = processed_dir / "synthetic_telemetry.parquet"
    streaming_source_dir = project_root / "data" / "geospatial_streaming_source"
    checkpoint_dir = project_root / "data" / "geospatial_streaming_checkpoint"
    output_dir = project_root / "data" / "geospatial_output"

    if not master_parquet_path.exists():
        raise FileNotFoundError(
            f"Synthetic telemetry not found at {master_parquet_path}. "
            "Run the bootstrapping notebook section first."
        )

    for directory in (streaming_source_dir, checkpoint_dir, output_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    master_df = spark.read.parquet(str(master_parquet_path))
    master_df.repartition(10).write.mode("overwrite").parquet(str(streaming_source_dir))

    telemetry_stream = (
        spark.readStream
        .schema(master_df.schema)
        .option("maxFilesPerTrigger", 1)
        .parquet(str(streaming_source_dir))
    )

    geospatial_stream = add_synthetic_coordinates(telemetry_stream)

    query = (
        geospatial_stream.writeStream
        .foreachBatch(lambda df, batch_id: _write_geojson_batch(df, batch_id, output_dir))
        .option("checkpointLocation", str(checkpoint_dir))
        .start()
    )

    try:
        query.processAllAvailable()
    finally:
        query.stop()

    logging.info("Geospatial streaming export completed. Output: %s", output_dir)
    return output_dir
