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

# Reference anchor in central Vienna used when per-agent anchors are unavailable.
VIENNA_LAT = 48.20849
VIENNA_LON = 16.37208

# Standard metres-to-degrees scaling conversions.
METRES_PER_DEGREE_LAT = 111_320.0
# Longitude metres shrink relative to latitude based on the cosine of the latitude angle.
METRES_PER_DEGREE_LON = 111_320.0 * math.cos(math.radians(VIENNA_LAT))
BASE_TIMESTAMP_MS = 1_700_000_000_000


def add_synthetic_coordinates(
    telemetry_df: DataFrame,
    anchors_df: DataFrame | None = None,
) -> DataFrame:
    """Add spatial latitude and longitude columns based on temporal progression and velocity constraints."""

    # Convert milliseconds timestamp difference to elapsed seconds for coordinate translation.
    elapsed_seconds = (col("Timestamp") - lit(BASE_TIMESTAMP_MS)) / lit(1000.0)

    # Velocity constraint models mapped to specific physical activities.
    speed_mps = (
        when(col("Activity") == "walk", lit(1.4))
        .when(col("Activity") == "bike", lit(4.5))
        .otherwise(lit(0.0))
    )

    if anchors_df is not None:
        # Join telemetry stream with the spatial anchor properties on agent state.
        positioned_df = telemetry_df.join(anchors_df, on=["Agent_ID", "Activity"], how="left")

        # Determine coordinates along pre-calculated OSMnx paths.
        if "route_lats" in positioned_df.columns:
            from pyspark.sql.functions import element_at, size, least, greatest, floor

            # Index mapping: Convert elapsed seconds to a 1-based index for Spark arrays.
            idx = floor(elapsed_seconds).cast("int") + 1

            # Index clamping: Limit the index between 1 and the array size to prevent index-out-of-bounds errors.
            clamped_idx = greatest(lit(1), least(idx, size(col("route_lats"))))

            return (
                positioned_df
                .withColumn("Latitude", element_at(col("route_lats"), clamped_idx))
                .withColumn("Longitude", element_at(col("route_lons"), clamped_idx))
                .drop("route_lats", "route_lons")
            )

        else:
            # Straight-line spatial fallback configuration.
            heading_radians = col("heading_radians")
            start_lat = col("start_lat")
            start_lon = col("start_lon")
    else:
        # Generate default heading vectors using unique Agent IDs.
        positioned_df = telemetry_df
        heading_radians = pmod(col("Agent_ID"), lit(16)) * lit(2.0 * math.pi / 16.0)
        start_lat = lit(VIENNA_LAT)
        start_lon = lit(VIENNA_LON)

    # Displacements calculations based on simple kinematics.
    distance_metres = elapsed_seconds * speed_mps
    east_metres = distance_metres * cos(heading_radians)
    north_metres = distance_metres * sin(heading_radians)
    
    # Scale longitude metres based on start latitude location to maintain projection accuracy.
    metres_per_degree_lon = lit(METRES_PER_DEGREE_LAT) * cos(
        radians(start_lat)
    )

    return (
        positioned_df
        .withColumn("Latitude", start_lat + north_metres / lit(METRES_PER_DEGREE_LAT))
        .withColumn("Longitude", start_lon + east_metres / metres_per_degree_lon)
    )


def _write_geojson_batch(batch_df: DataFrame, batch_id: int, output_dir: Path) -> None:
    """Serialize one Structured Streaming micro-batch as a GeoJSON FeatureCollection file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"telemetry_batch_{batch_id:05d}.geojson"
    temporary_path = output_path.with_suffix(".geojson.tmp")

    # Order rows to guarantee deterministic micro-batch structural records.
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

    # Write file using a temporary path to ensure atomic file writing boundaries.
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write('{"type":"FeatureCollection","features":[')
        first_feature = True

        # Use toLocalIterator() to safely stream rows to the driver without causing driver out-of-memory errors.
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

    # Atomically replace temporary file with complete target file.
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

    # Minimal diagnostics fallback generator checks.
    if not master_parquet_path.exists():
        from src._create_minimal_telemetry import create_minimal_telemetry
        create_minimal_telemetry(project_root, num_agents=30, samples_per_agent=500)
        spark.catalog.clearCache()

    # Clear directories to guarantee write-collision safety (Pipeline output idempotency).
    for directory in (streaming_source_dir, checkpoint_dir, output_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    # Ingest synthetic telemetry Parquet database and re-write to source folder to mimic streaming updates.
    master_df = spark.read.parquet(str(master_parquet_path))
    master_df.repartition(10).write.mode("overwrite").parquet(str(streaming_source_dir))

    # Build agent anchors based on OGD geographic layers.
    from src.step_12_vienna_spatial_layers import (
        build_agent_anchors_spark,
        download_vienna_layers,
        load_vienna_bike_paths,
        load_vienna_districts,
        load_vienna_pedestrian_zones,
    )
    spatial_dir = project_root / "data" / "spatial"
    layer_paths = download_vienna_layers(spatial_dir)
    districts_gdf = load_vienna_districts(layer_paths["districts"])
    pedestrian_gdf = load_vienna_pedestrian_zones(layer_paths["pedestrian_zones"])
    bike_gdf = load_vienna_bike_paths(layer_paths["bike_paths"])
    anchors_df = build_agent_anchors_spark(
        master_df, districts_gdf, pedestrian_gdf, bike_gdf
    )

    # structured streaming ingestion setup.
    # We restrict trigger to maxFilesPerTrigger=1 to ensure micro-batch simulation flows frame-by-frame.
    telemetry_stream = (
        spark.readStream
        .schema(master_df.schema)
        .option("maxFilesPerTrigger", 1)
        .parquet(str(streaming_source_dir))
    )

    # Project coordinates dynamically on streaming records.
    geospatial_stream = add_synthetic_coordinates(telemetry_stream, anchors_df)
    
    # Process micro-batches via foreachBatch to write GeoJSON files.
    query = (
        geospatial_stream.writeStream
        .foreachBatch(lambda df, batch_id: _write_geojson_batch(df, batch_id, output_dir))
        .option("checkpointLocation", str(checkpoint_dir))
        .start()
    )

    try:
        # Await finalization of all available streaming data in staging area.
        query.processAllAvailable()
    finally:
        # Terminate write query execution.
        query.stop()

    logging.info("Geospatial streaming export completed. Output: %s", output_dir)
    return output_dir
