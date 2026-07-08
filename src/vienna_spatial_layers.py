"""Download Vienna WFS layers and build agent start points without Pandas or GeoPandas."""

import logging
import math
import os
import random
import json
import urllib.request
from pathlib import Path

import osmnx as ox
import networkx as nx
from shapely.geometry import LineString, shape
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame

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

PROVIDER = "City of Vienna (Stadt Wien) - data.wien.gv.at"


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

        url = _layer_url(layer_info["type_name"])
        logging.info("Downloading %s from %s...", layer_info["english_name"], url)
        try:
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response:
                content = response.read()
                with open(output_path, "wb") as f:
                    f.write(content)
            logging.info("Saved %s features to %s", layer_info["english_name"], output_path)
        except Exception as e:
            logging.error("Failed to download layer %s: %s", layer_key, e)
            raise e

    return downloaded_paths


def load_vienna_districts(path: Path) -> list[dict]:
    """Load Vienna districts GeoJSON into a list of dictionaries with Shapely geometries."""
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    districts = []
    for feature in geojson["features"]:
        props = feature["properties"]
        geom = shape(feature["geometry"])

        name_column = next(
            (col for col in ("NAMEK", "NAME", "BEZNAME", "district_name") if col in props),
            None,
        )
        number_column = next(
            (col for col in ("BEZ", "BEZIRK", "GKZ", "district_number") if col in props),
            None,
        )

        name = str(props[name_column]) if name_column else "unknown"
        number = str(props[number_column]) if number_column else "unknown"

        districts.append({
            "district_name": name,
            "district_number": number,
            "geometry": geom,
        })
    return districts


def load_vienna_pedestrian_zones(path: Path) -> list[dict]:
    """Load pedestrian zones GeoJSON into a list of dictionaries with Shapely geometries."""
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    pedestrian_zones = []
    for feature in geojson["features"]:
        props = feature["properties"]
        geom = shape(feature["geometry"])

        label_column = next(
            (col for col in ("ADRESSE", "OBJECTID", "infrastructure_label") if col in props),
            None,
        )
        label = str(props[label_column]) if label_column else "unassigned"

        pedestrian_zones.append({
            "infrastructure_id": label,
            "infrastructure_label": label,
            "geometry": geom,
        })
    return pedestrian_zones


def load_vienna_bike_paths(path: Path) -> list[dict]:
    """Load bike paths GeoJSON into a list of dictionaries with Shapely geometries."""
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    bike_paths = []
    for feature in geojson["features"]:
        props = feature["properties"]
        geom = shape(feature["geometry"])

        label_column = next(
            (col for col in ("MERKMAL", "infrastructure_label") if col in props),
            None,
        )
        label = str(props[label_column]) if label_column else "unassigned"

        bike_paths.append({
            "infrastructure_id": label,
            "infrastructure_label": label,
            "geometry": geom,
        })
    return bike_paths


STATION_ATTRACTION_RATIO = 0.6

STATION_COORDS = [
    (48.20849, 16.37208),  # Stephansplatz
    (48.20026, 16.36988),  # Karlsplatz
    (48.1969, 16.3377),    # Westbahnhof
    (48.1852, 16.3762),    # Hauptbahnhof
    (48.2154, 16.3618),    # Schottentor
]


def load_vienna_streets_graph(project_root: Path) -> nx.MultiDiGraph | None:
    """Load the Vienna walk network graph, downloading and caching it if needed."""
    spatial_dir = project_root / "data" / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    graph_path = spatial_dir / "vienna_walk_network.graphml"

    if graph_path.exists() and graph_path.stat().st_size > 0:
        logging.info("Loading cached street graph: %s", graph_path.name)
        try:
            return ox.load_graphml(filepath=graph_path)
        except Exception as e:
            logging.warning("Failed to load cached graph: %s. Re-downloading...", e)

    logging.info("Downloading Vienna walk network graph via OSMnx (one-time cache operation)...")
    try:
        G = ox.graph_from_place("Vienna, Austria", network_type="walk")
        ox.save_graphml(G, filepath=graph_path)
        logging.info("Saved street graph to %s", graph_path)
        return G
    except Exception as e:
        logging.error("Failed to download street graph: %s", e)
        return None


def build_agent_anchor_list(
    agents: list[dict],
    districts: list[dict],
    pedestrian_zones: list[dict],
    bike_paths: list[dict],
) -> list[dict]:
    """Assign each synthetic agent a reproducible start point and sampled route coordinates."""

    district_geometries = [d["geometry"] for d in districts]
    district_labels = [d["district_name"] for d in districts]
    pedestrian_geometries = [p["geometry"] for p in pedestrian_zones]
    pedestrian_labels = [p["infrastructure_label"] for p in pedestrian_zones]
    bike_geometries = [b["geometry"] for b in bike_paths]
    bike_labels = [b["infrastructure_label"] for b in bike_paths]

    # Load routing graph.
    project_root = Path(__file__).resolve().parent.parent
    G = load_vienna_streets_graph(project_root)
    graph_nodes = list(G.nodes) if G is not None else []

    anchor_records = []
    for agent in agents:
        agent_id = int(agent["Agent_ID"])
        activity = str(agent["Activity"])
        heading_radians = (agent_id % 16) * (2.0 * math.pi / 16.0)

        # 1. Establish start point.
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

        # 2. Establish route path and destination.
        route_lats = []
        route_lons = []
        dest_lat, dest_lon = float(start_point.y), float(start_point.x)

        if activity in ("walk", "bike") and G is not None:
            try:
                # Snap start point to nearest graph node.
                start_node = ox.distance.nearest_nodes(G, float(start_point.x), float(start_point.y))

                # Route destination based on attraction ratio.
                # Reproducible randomness per agent.
                random.seed(agent_id + 200)
                if random.random() < STATION_ATTRACTION_RATIO:
                    # Target a transit station.
                    station_lat, station_lon = random.choice(STATION_COORDS)
                    dest_node = ox.distance.nearest_nodes(G, station_lon, station_lat)
                    dest_lat, dest_lon = station_lat, station_lon
                else:
                    # Target a random node in the street graph.
                    dest_node = random.choice(graph_nodes)
                    dest_lat = float(G.nodes[dest_node]["y"])
                    dest_lon = float(G.nodes[dest_node]["x"])

                # Calculate shortest path along the graph.
                path = nx.shortest_path(G, source=start_node, target=dest_node, weight="length")
                coords = [[float(G.nodes[n]["y"]), float(G.nodes[n]["x"])] for n in path]
                
                # Sample coordinates at 1-second intervals.
                if len(coords) > 1:
                    coords_meters = []
                    curr_x, curr_y = 0.0, 0.0
                    coords_meters.append((curr_x, curr_y))
                    for i in range(1, len(coords)):
                        lat1, lon1 = coords[i-1]
                        lat2, lon2 = coords[i]
                        dy = (lat2 - lat1) * 111320.0
                        dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat1))
                        curr_x += dx
                        curr_y += dy
                        coords_meters.append((curr_x, curr_y))
                    
                    line_meters = LineString(coords_meters)
                    total_length = line_meters.length
                    
                    step_distance = 1.4 if activity == "walk" else 4.5
                    d = 0.0
                    m_lon = 111320.0 * math.cos(math.radians(coords[0][0]))
                    while d < total_length:
                        pt = line_meters.interpolate(d)
                        lat = coords[0][0] + pt.y / 111320.0
                        lon = coords[0][1] + pt.x / m_lon
                        route_lats.append(float(lat))
                        route_lons.append(float(lon))
                        d += step_distance
                    
                    # Add final destination coordinate.
                    route_lats.append(float(dest_lat))
                    route_lons.append(float(dest_lon))
                else:
                    route_lats = [float(start_point.y)]
                    route_lons = [float(start_point.x)]
            except Exception as e:
                logging.warning(
                    f"Failed to find route for Agent {agent_id} ({activity}): {e}. Using straight-line fallback."
                )
                station_lat, station_lon = STATION_COORDS[agent_id % len(STATION_COORDS)]
                dest_lat, dest_lon = station_lat, station_lon
                
                dy = (station_lat - start_point.y) * 111320.0
                dx = (station_lon - start_point.x) * 111320.0 * math.cos(math.radians(start_point.y))
                dist = math.sqrt(dx*dx + dy*dy)
                step_distance = 1.4 if activity == "walk" else 4.5
                
                if dist > 0:
                    steps = int(dist / step_distance)
                    for s in range(steps + 1):
                        frac = (s * step_distance) / dist
                        route_lats.append(float(start_point.y + frac * (station_lat - start_point.y)))
                        route_lons.append(float(start_point.x + frac * (station_lon - start_point.x)))
                else:
                    route_lats = [float(start_point.y)]
                    route_lons = [float(start_point.x)]
        else:
            route_lats = [float(start_point.y)]
            route_lons = [float(start_point.x)]

        anchor_records.append(
            {
                "Agent_ID": agent_id,
                "Activity": activity,
                "start_lat": float(start_point.y),
                "start_lon": float(start_point.x),
                "heading_radians": heading_radians,
                "start_infrastructure_type": infrastructure_type,
                "start_infrastructure_label": infrastructure_label,
                "route_lats": route_lats,
                "route_lons": route_lons,
                "dest_lat": float(dest_lat),
                "dest_lon": float(dest_lon),
            }
        )

    return anchor_records


def build_agent_anchors_spark(
    telemetry_df: DataFrame,
    districts: list[dict],
    pedestrian_zones: list[dict],
    bike_paths: list[dict],
) -> DataFrame:
    """Build a Spark anchor table for all agents in the telemetry dataset without using Pandas."""

    agents = (
        telemetry_df.select("Agent_ID", "Activity")
        .dropDuplicates()
        .orderBy("Agent_ID")
        .collect()
    )
    agents_list = [{"Agent_ID": int(r.Agent_ID), "Activity": str(r.Activity)} for r in agents]

    anchor_records = build_agent_anchor_list(
        agents_list,
        districts,
        pedestrian_zones,
        bike_paths,
    )

    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    temp_file_path = temp_dir / "temp_agent_anchors.parquet"

    # Define schema explicitly for PyArrow pylist conversion (0% Pandas!).
    schema = pa.schema([
        ("Agent_ID", pa.int64()),
        ("Activity", pa.string()),
        ("start_lat", pa.float64()),
        ("start_lon", pa.float64()),
        ("heading_radians", pa.float64()),
        ("start_infrastructure_type", pa.string()),
        ("start_infrastructure_label", pa.string()),
        ("route_lats", pa.list_(pa.float64())),
        ("route_lons", pa.list_(pa.float64())),
        ("dest_lat", pa.float64()),
        ("dest_lon", pa.float64()),
    ])

    table = pa.Table.from_pylist(anchor_records, schema=schema)
    pq.write_table(table, str(temp_file_path))

    spark = telemetry_df.sparkSession
    return spark.read.parquet(str(temp_file_path))


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
