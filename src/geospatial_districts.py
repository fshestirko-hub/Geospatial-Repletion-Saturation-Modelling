"""Spatial joins between synthetic telemetry and Vienna map layers."""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, floor, lit
from pyspark.sql.types import StringType, StructField, StructType
from shapely import from_wkb
from shapely.geometry import Point
from shapely.strtree import STRtree

from src.geospatial_streaming import add_synthetic_coordinates
from src.vienna_spatial_layers import (
    build_agent_anchors_spark,
    download_vienna_layers,
    load_vienna_bike_paths,
    load_vienna_districts,
    load_vienna_pedestrian_zones,
    write_spatial_provenance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TIME_WINDOW_SECONDS = 60
BIKE_MATCH_TOLERANCE_METRES = 150
METRES_PER_DEGREE_LAT = 111_320.0


def _build_district_lookup(districts_gdf: gpd.GeoDataFrame) -> list[dict]:
    lookup = []
    for row in districts_gdf.itertuples(index=False):
        lookup.append(
            {
                "district_name": row.district_name,
                "district_number": row.district_number,
                "geometry_wkb": row.geometry.wkb,
            }
        )
    return lookup


def _build_pedestrian_lookup(pedestrian_gdf: gpd.GeoDataFrame) -> list[dict]:
    lookup = []
    for row in pedestrian_gdf.itertuples(index=False):
        lookup.append(
            {
                "infrastructure_label": row.infrastructure_label,
                "geometry_wkb": row.geometry.wkb,
            }
        )
    return lookup


def _build_bike_lookup(bike_gdf: gpd.GeoDataFrame) -> list[dict]:
    lookup = []
    for row in bike_gdf.itertuples(index=False):
        lookup.append(
            {
                "infrastructure_label": row.infrastructure_label,
                "geometry_wkb": row.geometry.wkb,
            }
        )
    return lookup


def assign_spatial_context(
    telemetry_df: DataFrame,
    districts_gdf: gpd.GeoDataFrame,
    pedestrian_gdf: gpd.GeoDataFrame,
    bike_gdf: gpd.GeoDataFrame,
) -> DataFrame:
    """Assign district and activity-specific infrastructure labels in Spark."""

    spark_context = telemetry_df.sparkSession.sparkContext
    district_lookup = spark_context.broadcast(_build_district_lookup(districts_gdf))
    pedestrian_lookup = spark_context.broadcast(_build_pedestrian_lookup(pedestrian_gdf))
    bike_lookup = spark_context.broadcast(_build_bike_lookup(bike_gdf))

    def assign_partition(iterator):
        districts = district_lookup.value
        pedestrian_zones = pedestrian_lookup.value
        bike_paths = bike_lookup.value
        bike_geometries = [from_wkb(item["geometry_wkb"]) for item in bike_paths]
        bike_tree = STRtree(bike_geometries)
        bike_labels = [item["infrastructure_label"] for item in bike_paths]

        def lookup_district(longitude: float, latitude: float) -> tuple[str, str]:
            point = Point(longitude, latitude)
            for district in districts:
                if from_wkb(district["geometry_wkb"]).contains(point):
                    return district["district_name"], district["district_number"]
            return "Outside Vienna", "outside"

        def lookup_infrastructure(activity: str, longitude: float, latitude: float) -> tuple[str, str]:
            point = Point(longitude, latitude)

            if activity == "walk":
                for zone in pedestrian_zones:
                    if from_wkb(zone["geometry_wkb"]).contains(point):
                        return "pedestrian_zone", zone["infrastructure_label"]
                return "pedestrian_zone", "unassigned"

            if activity == "bike":
                if bike_geometries:
                    nearest_indices = bike_tree.query_nearest(point)
                    if len(nearest_indices) > 0:
                        nearest_index = int(nearest_indices[0])
                        nearest_geometry = bike_geometries[nearest_index]
                        distance_metres = point.distance(nearest_geometry) * METRES_PER_DEGREE_LAT
                        if distance_metres <= BIKE_MATCH_TOLERANCE_METRES:
                            return "bike_path", bike_labels[nearest_index]
                return "bike_path", "unassigned"

            district_name, _ = lookup_district(longitude, latitude)
            return "district_centroid", district_name

        for pdf in iterator:
            district_values = pdf.apply(
                lambda row: lookup_district(row["Longitude"], row["Latitude"]),
                axis=1,
                result_type="expand",
            )
            infrastructure_values = pdf.apply(
                lambda row: lookup_infrastructure(
                    row["Activity"], row["Longitude"], row["Latitude"]
                ),
                axis=1,
                result_type="expand",
            )
            pdf["district_name"] = district_values[0]
            pdf["district_number"] = district_values[1]
            pdf["infrastructure_type"] = infrastructure_values[0]
            pdf["infrastructure_label"] = infrastructure_values[1]
            yield pdf

    output_schema = StructType(
        list(telemetry_df.schema.fields)
        + [
            StructField("district_name", StringType(), True),
            StructField("district_number", StringType(), True),
            StructField("infrastructure_type", StringType(), True),
            StructField("infrastructure_label", StringType(), True),
        ]
    )
    return telemetry_df.mapInPandas(assign_partition, schema=output_schema)


def aggregate_district_activity_counts(assigned_df: DataFrame) -> DataFrame:
    windowed_df = assigned_df.withColumn(
        "time_window_index",
        floor((col("Timestamp") - lit(1_700_000_000_000)) / lit(TIME_WINDOW_SECONDS * 1000)),
    )
    return (
        windowed_df
        .groupBy("district_name", "district_number", "Activity", "time_window_index")
        .count()
        .orderBy("district_number", "Activity", "time_window_index")
    )


def aggregate_infrastructure_activity_counts(assigned_df: DataFrame) -> DataFrame:
    windowed_df = assigned_df.withColumn(
        "time_window_index",
        floor((col("Timestamp") - lit(1_700_000_000_000)) / lit(TIME_WINDOW_SECONDS * 1000)),
    )
    return (
        windowed_df
        .groupBy(
            "infrastructure_type",
            "infrastructure_label",
            "Activity",
            "time_window_index",
        )
        .count()
        .orderBy("infrastructure_type", "Activity", "time_window_index")
    )


def save_choropleth_map(
    districts_gdf: gpd.GeoDataFrame,
    summary_pdf: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = (
        summary_pdf.groupby(["district_name", "district_number"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "simulated_record_count"})
    )
    map_gdf = districts_gdf.merge(
        totals,
        on=["district_name", "district_number"],
        how="left",
    )
    map_gdf["simulated_record_count"] = map_gdf["simulated_record_count"].fillna(0)

    figure, axis = plt.subplots(figsize=(10, 8))
    map_gdf.plot(
        column="simulated_record_count",
        cmap="YlOrRd",
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=axis,
        legend_kwds={"label": "Simulated record count"},
    )
    axis.set_title("Simulated Telemetry Count by Vienna District")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    logging.info("Saved district choropleth map to %s", output_path)
    return output_path


def save_scatter_map(
    districts_gdf: gpd.GeoDataFrame,
    points_pdf: pd.DataFrame,
    output_path: Path,
    max_points: int = 5000,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(points_pdf) > max_points:
        points_pdf = points_pdf.sample(max_points, random_state=42)

    activity_colors = {
        "walk": "#2ca25f",
        "bike": "#3182bd",
        "stand": "#756bb1",
    }

    figure, axis = plt.subplots(figsize=(10, 8))
    districts_gdf.boundary.plot(ax=axis, linewidth=0.6, color="#444444")
    for activity, color in activity_colors.items():
        activity_points = points_pdf[points_pdf["Activity"] == activity]
        if activity_points.empty:
            continue
        axis.scatter(
            activity_points["Longitude"],
            activity_points["Latitude"],
            s=8,
            alpha=0.5,
            c=color,
            label=activity,
        )

    axis.set_title("Synthetic Agent Locations by Activity")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend(title="Activity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    logging.info("Saved scatter map to %s", output_path)
    return output_path


def run_district_assignment(spark: SparkSession, project_root: Path) -> dict[str, Path]:
    """Run the full geospatial assignment pipeline."""

    project_root = Path(project_root)
    spatial_dir = project_root / "data" / "spatial"
    output_dir = project_root / "data" / "geospatial_output"
    telemetry_path = project_root / "data" / "processed" / "synthetic_telemetry.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"Synthetic telemetry not found at {telemetry_path}. "
            "Run bootstrapping or src/create_minimal_telemetry.py first."
        )

    layer_paths = download_vienna_layers(spatial_dir)
    districts_gdf = load_vienna_districts(layer_paths["districts"])
    pedestrian_gdf = load_vienna_pedestrian_zones(layer_paths["pedestrian_zones"])
    bike_gdf = load_vienna_bike_paths(layer_paths["bike_paths"])

    telemetry_df = spark.read.parquet(str(telemetry_path))
    full_anchors_df = build_agent_anchors_spark(
        telemetry_df,
        districts_gdf,
        pedestrian_gdf,
        bike_gdf,
    )
    full_anchors_df.write.mode("overwrite").parquet(str(spatial_dir / "agent_anchors.parquet"))

    anchors_df = full_anchors_df.select(
        "Agent_ID",
        "start_lat",
        "start_lon",
        "heading_radians",
        "start_infrastructure_type",
        "start_infrastructure_label",
    )

    geospatial_df = add_synthetic_coordinates(telemetry_df, anchors_df)
    assigned_df = assign_spatial_context(
        geospatial_df,
        districts_gdf,
        pedestrian_gdf,
        bike_gdf,
    )

    district_summary_df = aggregate_district_activity_counts(assigned_df)
    infrastructure_summary_df = aggregate_infrastructure_activity_counts(assigned_df)

    district_summary_path = output_dir / "district_activity_counts.csv"
    infrastructure_summary_path = output_dir / "infrastructure_activity_counts.csv"
    district_summary_pdf = district_summary_df.toPandas()
    infrastructure_summary_pdf = infrastructure_summary_df.toPandas()
    district_summary_pdf.to_csv(district_summary_path, index=False)
    infrastructure_summary_pdf.to_csv(infrastructure_summary_path, index=False)

    map_path = save_choropleth_map(
        districts_gdf,
        district_summary_pdf,
        output_dir / "district_choropleth.png",
    )

    points_pdf = assigned_df.select(
        "Agent_ID", "Activity", "Latitude", "Longitude", "district_name"
    ).toPandas()
    scatter_path = save_scatter_map(
        districts_gdf,
        points_pdf,
        output_dir / "telemetry_scatter_map.png",
    )

    metadata_path = write_spatial_provenance(output_dir / "spatial_data_provenance.txt")

    logging.info("Geospatial assignment complete.")
    return {
        "summary_csv": district_summary_path,
        "infrastructure_csv": infrastructure_summary_path,
        "map_png": map_path,
        "scatter_png": scatter_path,
        "districts_geojson": layer_paths["districts"],
        "pedestrian_geojson": layer_paths["pedestrian_zones"],
        "bike_geojson": layer_paths["bike_paths"],
        "agent_anchors": spatial_dir / "agent_anchors.parquet",
        "metadata": metadata_path,
    }
