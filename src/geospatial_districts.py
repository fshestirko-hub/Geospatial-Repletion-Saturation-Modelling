import logging
from pathlib import Path
import math
from collections import defaultdict
from pyspark.sql import DataFrame, SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TIME_WINDOW_SECONDS = 60
BIKE_MATCH_TOLERANCE_METRES = 150
METRES_PER_DEGREE_LAT = 111_320.0


def assign_spatial_context(
    telemetry_df: DataFrame,
    districts: list[dict],
    pedestrian_zones: list[dict],
    bike_paths: list[dict],
) -> list[dict]:
    """Assign district and activity-specific infrastructure labels natively on the Spark driver using Shapely."""
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    logging.info("Collecting telemetry data to driver for spatial context assignment...")
    rows = telemetry_df.collect()

    logging.info("Building spatial search indices on the driver...")
    # 1. District Index.
    dist_geoms = [d["geometry"] for d in districts]
    dist_tree = STRtree(dist_geoms)
    dist_names = [d["district_name"] for d in districts]
    dist_numbers = [d["district_number"] for d in districts]

    # 2. Pedestrian Index.
    ped_geoms = [p["geometry"] for p in pedestrian_zones]
    ped_tree = STRtree(ped_geoms) if ped_geoms else None
    ped_labels = [p["infrastructure_label"] for p in pedestrian_zones]

    # 3. Bike Index.
    bike_geoms = [b["geometry"] for b in bike_paths]
    bike_tree = STRtree(bike_geoms) if bike_geoms else None
    bike_labels = [b["infrastructure_label"] for b in bike_paths]

    logging.info("Assigning spatial contexts to %d records...", len(rows))
    assigned_rows = []

    for r in rows:
        x, y = float(r.Longitude), float(r.Latitude)
        activity = r.Activity
        point = Point(x, y)

        # 1. District Lookup.
        d_name, d_num = "Outside Vienna", "outside"
        if dist_tree is not None:
            cand_indices = dist_tree.query(point)
            for idx in cand_indices:
                if dist_geoms[int(idx)].contains(point):
                    d_name, d_num = dist_names[int(idx)], dist_numbers[int(idx)]
                    break

        # 2. Infrastructure Lookup.
        infra_type, infra_label = "district_centroid", d_name

        if activity == "walk":
            if ped_tree is not None:
                cand_indices = ped_tree.query(point)
                found = False
                for idx in cand_indices:
                    if ped_geoms[int(idx)].contains(point):
                        infra_type, infra_label = "pedestrian_zone", ped_labels[int(idx)]
                        found = True
                        break
                if not found:
                    infra_type, infra_label = "pedestrian_zone", "unassigned"
            else:
                infra_type, infra_label = "pedestrian_zone", "unassigned"

        elif activity == "bike":
            if bike_tree is not None:
                nearest_idx = bike_tree.query_nearest(point)
                if nearest_idx is not None:
                    if hasattr(nearest_idx, "__len__") or hasattr(nearest_idx, "__iter__"):
                        idx_val = int(list(nearest_idx)[0])
                    else:
                        idx_val = int(nearest_idx)

                    nearest_geom = bike_geoms[idx_val]
                    distance_metres = point.distance(nearest_geom) * METRES_PER_DEGREE_LAT
                    if distance_metres <= BIKE_MATCH_TOLERANCE_METRES:
                        infra_type, infra_label = "bike_path", bike_labels[idx_val]
                    else:
                        infra_type, infra_label = "bike_path", "unassigned"
                else:
                    infra_type, infra_label = "bike_path", "unassigned"
            else:
                infra_type, infra_label = "bike_path", "unassigned"

        row_dict = r.asDict()
        row_dict["district_name"] = d_name
        row_dict["district_number"] = d_num
        row_dict["infrastructure_type"] = infra_type
        row_dict["infrastructure_label"] = infra_label
        assigned_rows.append(row_dict)

    return assigned_rows


def aggregate_district_activity_counts(assigned_rows: list[dict]) -> list[tuple]:
    """Aggregate activity records by district and time window using pure Python."""
    counts = defaultdict(int)
    for r in assigned_rows:
        ts = r["Timestamp"]
        time_window_index = int(math.floor((ts - 1_700_000_000_000) / (TIME_WINDOW_SECONDS * 1000)))
        key = (r["district_name"], r["district_number"], r["Activity"], time_window_index)
        counts[key] += 1

    def sort_key(item):
        d_name, d_num, activity, win_idx = item[0]
        try:
            num_val = int(d_num)
        except ValueError:
            num_val = 999999
        return (num_val, activity, win_idx)

    return sorted(counts.items(), key=sort_key)


def aggregate_infrastructure_activity_counts(assigned_rows: list[dict]) -> list[tuple]:
    """Aggregate activity records by infrastructure type/label and time window using pure Python."""
    counts = defaultdict(int)
    for r in assigned_rows:
        ts = r["Timestamp"]
        time_window_index = int(math.floor((ts - 1_700_000_000_000) / (TIME_WINDOW_SECONDS * 1000)))
        key = (r["infrastructure_type"], r["infrastructure_label"], r["Activity"], time_window_index)
        counts[key] += 1

    def sort_key(item):
        infra_type, infra_label, activity, win_idx = item[0]
        return (infra_type, activity, win_idx)

    return sorted(counts.items(), key=sort_key)


def save_choropleth_map(
    districts: list[dict],
    district_counts_list: list,
    output_path: Path,
) -> Path:
    """Plot Vienna districts as a choropleth map using pure Matplotlib patches (0% GeoPandas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = defaultdict(int)
    for (d_name, d_num, activity, win_idx), count in district_counts_list:
        totals[d_name] += count

    figure, axis = plt.subplots(figsize=(10, 8))
    patches = []
    values = []

    for d in districts:
        geom = d["geometry"]
        count = totals.get(d["district_name"], 0)
        
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            polys = []
            
        for poly in polys:
            coords = np.array(poly.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
            values.append(count)

    p_col = PatchCollection(patches, cmap="YlOrRd", edgecolors="black", linewidths=0.8)
    p_col.set_array(np.array(values))
    axis.add_collection(p_col)

    # Automatically set plot bounds.
    all_lons = []
    all_lats = []
    for d in districts:
        geom = d["geometry"]
        minx, miny, maxx, maxy = geom.bounds
        all_lons.extend([minx, maxx])
        all_lats.extend([miny, maxy])
    if all_lons and all_lats:
        axis.set_xlim(min(all_lons) - 0.01, max(all_lons) + 0.01)
        axis.set_ylim(min(all_lats) - 0.01, max(all_lats) + 0.01)

    figure.colorbar(p_col, ax=axis, label="Simulated record count")
    axis.set_title("Simulated Telemetry Count by Vienna District")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    logging.info("Saved district choropleth map to %s", output_path)
    return output_path


def save_scatter_map(
    districts: list[dict],
    assigned_rows: list[dict],
    output_path: Path,
    max_points: int = 5000,
) -> Path:
    """Plot synthetic agent locations over district boundary lines using pure Matplotlib (0% GeoPandas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(assigned_rows) > max_points:
        points_sample = random.sample(assigned_rows, max_points)
    else:
        points_sample = assigned_rows

    activity_colors = {
        "walk": "#2ca25f",
        "bike": "#3182bd",
        "stand": "#756bb1",
    }

    figure, axis = plt.subplots(figsize=(10, 8))
    
    # Plot district boundaries.
    for d in districts:
        geom = d["geometry"]
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            polys = []
        for poly in polys:
            coords = np.array(poly.exterior.coords)
            axis.plot(coords[:, 0], coords[:, 1], color="#444444", linewidth=0.6)

    # Group coordinates by activity.
    by_activity = {act: ([], []) for act in activity_colors}
    for r in points_sample:
        activity = r["Activity"]
        if activity in by_activity:
            by_activity[activity][0].append(float(r["Longitude"]))
            by_activity[activity][1].append(float(r["Latitude"]))

    for activity, color in activity_colors.items():
        lons, lats = by_activity[activity]
        if not lons:
            continue
        axis.scatter(
            lons,
            lats,
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
    from src.geospatial_streaming import add_synthetic_coordinates
    from src.geospatial_visualization import create_interactive_playback_map
    from src.vienna_spatial_layers import (
        build_agent_anchors_spark,
        download_vienna_layers,
        load_vienna_bike_paths,
        load_vienna_districts,
        load_vienna_pedestrian_zones,
        write_spatial_provenance,
    )

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
    districts = load_vienna_districts(layer_paths["districts"])
    pedestrian_zones = load_vienna_pedestrian_zones(layer_paths["pedestrian_zones"])
    bike_paths = load_vienna_bike_paths(layer_paths["bike_paths"])

    telemetry_df = spark.read.parquet(str(telemetry_path))
    full_anchors_df = build_agent_anchors_spark(
        telemetry_df,
        districts,
        pedestrian_zones,
        bike_paths,
    )
    full_anchors_df.write.mode("overwrite").parquet(str(spatial_dir / "agent_anchors.parquet"))

    anchors_df = full_anchors_df.select(
        "Agent_ID",
        "Activity",
        "start_lat",
        "start_lon",
        "heading_radians",
        "start_infrastructure_type",
        "start_infrastructure_label",
        "route_lats",
        "route_lons",
        "dest_lat",
        "dest_lon",
    )

    geospatial_df = add_synthetic_coordinates(telemetry_df, anchors_df)
    
    # Run spatial context assignment loop on the driver.
    assigned_rows = assign_spatial_context(
        geospatial_df,
        districts,
        pedestrian_zones,
        bike_paths,
    )

    # Perform aggregations on the driver using pure Python.
    district_counts_list = aggregate_district_activity_counts(assigned_rows)
    infra_counts_list = aggregate_infrastructure_activity_counts(assigned_rows)

    district_summary_path = output_dir / "district_activity_counts.csv"
    infrastructure_summary_path = output_dir / "infrastructure_activity_counts.csv"

    # Write CSV summaries directly using standard Python CSV (0% Pandas).
    import csv
    logging.info("Writing district summary CSV...")
    with open(district_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["district_name", "district_number", "Activity", "time_window_index", "count"])
        for (d_name, d_num, activity, win_idx), count in district_counts_list:
            writer.writerow([d_name, d_num, activity, win_idx, count])

    logging.info("Writing infrastructure summary CSV...")
    with open(infrastructure_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["infrastructure_label", "infrastructure_type", "Activity", "time_window_index", "count"])
        for (infra_type, infra_label, activity, win_idx), count in infra_counts_list:
            writer.writerow([infra_label, infra_type, activity, win_idx, count])

    map_path = save_choropleth_map(
        districts,
        district_counts_list,
        output_dir / "district_choropleth.png",
    )

    scatter_path = save_scatter_map(
        districts,
        assigned_rows,
        output_dir / "telemetry_scatter_map.png",
    )

    playback_html_path = output_dir / "interactive_citizens_map.html"
    logging.info("Generating interactive playback map...")
    create_interactive_playback_map(assigned_rows, layer_paths["districts"], playback_html_path)

    metadata_path = write_spatial_provenance(output_dir / "spatial_data_provenance.txt")

    logging.info("Geospatial assignment complete.")
    return {
        "summary_csv": district_summary_path,
        "infrastructure_csv": infrastructure_summary_path,
        "map_png": map_path,
        "scatter_png": scatter_path,
        "playback_html": playback_html_path,
        "districts_geojson": layer_paths["districts"],
        "pedestrian_geojson": layer_paths["pedestrian_zones"],
        "bike_geojson": layer_paths["bike_paths"],
        "agent_anchors": spatial_dir / "agent_anchors.parquet",
        "metadata": metadata_path,
    }
