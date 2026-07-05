"""Download Vienna WFS layers and build agent start points."""

import logging
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

WFS_BASE = (
    "https://data.wien.gv.at/daten/geo?"
    "service=WFS&request=GetFeature&version=1.1.0"
    "&srsName=EPSG:4326&outputFormat=json"
)

SPATIAL_LAYERS = {
    "districts": {
        "type_name": "ogdwien:BEZIRKSGRENZEOGD",
        "filename": "vienna_districts.geojson",
        "english_name": "Vienna district boundaries (Bezirksgrenzen)",
        "licence": "CC BY 4.0 AT",
    },
    "pedestrian_zones": {
        "type_name": "ogdwien:FUSSGEHERZONEOGD",
        "filename": "vienna_pedestrian_zones.geojson",
        "english_name": "Vienna pedestrian zones (Fußgängerzonen)",
        "licence": "CC BY 4.0 AT",
    },
    "bike_paths": {
        "type_name": "ogdwien:RADWEGEOGD",
        "filename": "vienna_bike_paths.geojson",
        "english_name": "Vienna bike path network (Radwege)",
        "licence": "CC BY 4.0 AT",
    },
}

PROVIDER = "City of Vienna – Open Government Data"


def _layer_url(type_name: str) -> str:
    return f"{WFS_BASE}&typeName={type_name}"


def download_vienna_layers(spatial_dir: Path) -> dict[str, Path]:
    """Download official Vienna layers if they are not cached locally."""

    spatial_dir = Path(spatial_dir)
    spatial_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: dict[str, Path] = {}

    for layer_key, layer_info in SPATIAL_LAYERS.items():
        output_path = spatial_dir / layer_info["filename"]
        downloaded_paths[layer_key] = output_path

        if output_path.exists() and output_path.stat().st_size > 0:
            logging.info("Layer already cached: %s", output_path.name)
            continue

        logging.info("Downloading %s...", layer_info["english_name"])
        layer_gdf = gpd.read_file(_layer_url(layer_info["type_name"]))
        layer_gdf.to_file(output_path, driver="GeoJSON")
        logging.info("Saved %s features to %s", len(layer_gdf), output_path)

    return downloaded_paths


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def load_vienna_districts(path: Path) -> gpd.GeoDataFrame:
    districts_gdf = _to_wgs84(gpd.read_file(path))

    name_column = next(
        (column for column in ("NAMEK", "NAME", "BEZNAME") if column in districts_gdf.columns),
        None,
    )
    number_column = next(
        (column for column in ("BEZ", "BEZIRK", "GKZ") if column in districts_gdf.columns),
        None,
    )

    districts_gdf["district_name"] = (
        districts_gdf[name_column].astype(str) if name_column else districts_gdf.index.astype(str)
    )
    districts_gdf["district_number"] = (
        districts_gdf[number_column].astype(str) if number_column else "unknown"
    )
    return districts_gdf[["district_name", "district_number", "geometry"]]


def load_vienna_pedestrian_zones(path: Path) -> gpd.GeoDataFrame:
    pedestrian_gdf = _to_wgs84(gpd.read_file(path))
    label_column = "ADRESSE" if "ADRESSE" in pedestrian_gdf.columns else "OBJECTID"
    pedestrian_gdf["infrastructure_id"] = pedestrian_gdf[label_column].astype(str)
    pedestrian_gdf["infrastructure_label"] = pedestrian_gdf["infrastructure_id"]
    return pedestrian_gdf[["infrastructure_id", "infrastructure_label", "geometry"]]


def load_vienna_bike_paths(path: Path) -> gpd.GeoDataFrame:
    bike_gdf = _to_wgs84(gpd.read_file(path))
    if "MERKMAL" in bike_gdf.columns:
        bike_gdf["infrastructure_id"] = bike_gdf["MERKMAL"].astype(str)
    else:
        bike_gdf["infrastructure_id"] = bike_gdf.index.astype(str)
    bike_gdf["infrastructure_label"] = bike_gdf["infrastructure_id"]
    return bike_gdf[["infrastructure_id", "infrastructure_label", "geometry"]]


def build_agent_anchor_pdf(
    agents_pdf: pd.DataFrame,
    districts_gdf: gpd.GeoDataFrame,
    pedestrian_gdf: gpd.GeoDataFrame,
    bike_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign each synthetic agent a reproducible start point on city infrastructure."""

    district_geometries = list(districts_gdf.geometry)
    district_labels = districts_gdf["district_name"].tolist()
    pedestrian_geometries = list(pedestrian_gdf.geometry)
    pedestrian_labels = pedestrian_gdf["infrastructure_label"].tolist()
    bike_geometries = list(bike_gdf.geometry)
    bike_labels = bike_gdf["infrastructure_label"].tolist()

    anchor_records = []
    for agent_row in agents_pdf.itertuples(index=False):
        agent_id = int(agent_row.Agent_ID)
        activity = str(agent_row.Activity)
        heading_radians = (agent_id % 16) * (2.0 * math.pi / 16.0)

        if activity == "walk":
            feature_index = agent_id % len(pedestrian_geometries)
            start_point = pedestrian_geometries[feature_index].representative_point()
            infrastructure_type = "pedestrian_zone"
            infrastructure_label = pedestrian_labels[feature_index]
        elif activity == "bike":
            feature_index = agent_id % len(bike_geometries)
            line = bike_geometries[feature_index]
            start_point = line.interpolate(0.5, normalized=True) if line.length > 0 else line.centroid
            infrastructure_type = "bike_path"
            infrastructure_label = bike_labels[feature_index]
        else:
            feature_index = agent_id % len(district_geometries)
            start_point = district_geometries[feature_index].representative_point()
            infrastructure_type = "district_centroid"
            infrastructure_label = district_labels[feature_index]

        anchor_records.append(
            {
                "Agent_ID": agent_id,
                "Activity": activity,
                "start_lat": float(start_point.y),
                "start_lon": float(start_point.x),
                "heading_radians": heading_radians,
                "start_infrastructure_type": infrastructure_type,
                "start_infrastructure_label": infrastructure_label,
            }
        )

    return pd.DataFrame(anchor_records)


def build_agent_anchors_spark(
    telemetry_df: DataFrame,
    districts_gdf: gpd.GeoDataFrame,
    pedestrian_gdf: gpd.GeoDataFrame,
    bike_gdf: gpd.GeoDataFrame,
) -> DataFrame:
    """Build a Spark anchor table for all agents in the telemetry dataset."""

    agents_pdf = (
        telemetry_df.select("Agent_ID", "Activity")
        .dropDuplicates()
        .orderBy("Agent_ID")
        .toPandas()
    )
    anchor_pdf = build_agent_anchor_pdf(
        agents_pdf,
        districts_gdf,
        pedestrian_gdf,
        bike_gdf,
    )

    anchor_schema = StructType(
        [
            StructField("Agent_ID", LongType(), False),
            StructField("Activity", StringType(), False),
            StructField("start_lat", DoubleType(), False),
            StructField("start_lon", DoubleType(), False),
            StructField("heading_radians", DoubleType(), False),
            StructField("start_infrastructure_type", StringType(), False),
            StructField("start_infrastructure_label", StringType(), False),
        ]
    )
    spark = telemetry_df.sparkSession
    return spark.createDataFrame(anchor_pdf, schema=anchor_schema)


def write_spatial_provenance(output_path: Path) -> Path:
    """Document all official spatial datasets used in the geospatial stage."""

    lines = [
        f"Provider: {PROVIDER}",
        "Licence for all City of Vienna layers: CC BY 4.0 AT",
        "",
    ]
    for layer_info in SPATIAL_LAYERS.values():
        lines.extend(
            [
                f"Dataset: {layer_info['english_name']}",
                f"WFS typeName: {layer_info['type_name']}",
                f"Source URL: {_layer_url(layer_info['type_name'])}",
                "",
            ]
        )
    lines.extend(
        [
            "Note: telemetry coordinates are synthetic (HHAR has no GPS).",
            "Walk agents start on pedestrian zones, bike agents on bike paths.",
            "Stand agents use a district reference point.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
