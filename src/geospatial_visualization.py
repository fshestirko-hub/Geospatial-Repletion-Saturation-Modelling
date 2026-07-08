"""Geospatial interactive visualizations using Leaflet and Folium without Pandas or GeoPandas."""

import json
from pathlib import Path
from datetime import datetime, timedelta
import folium
from folium.plugins import TimestampedGeoJson

def create_interactive_playback_map(
    assigned_rows: list[dict],
    districts_geojson_path: Path,
    output_html_path: Path,
) -> Path:
    """Generate an interactive Leaflet HTML map animating agents over time."""
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialise Folium Map centred on Vienna city centre.
    vienna_center = [48.20849, 16.37208]
    m = folium.Map(
        location=vienna_center,
        zoom_start=13,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # 2. Add District Boundaries as a subtle background layer directly from GeoJSON.
    with open(districts_geojson_path, "r", encoding="utf-8") as f:
        districts_geojson = json.load(f)

    # Standardise district fields in GeoJSON properties for the tooltip.
    for feature in districts_geojson["features"]:
        props = feature["properties"]
        name_col = next((c for c in ("NAMEK", "NAME", "BEZNAME", "district_name") if c in props), None)
        num_col = next((c for c in ("BEZ", "BEZIRK", "GKZ", "district_number") if c in props), None)
        props["district_name"] = str(props[name_col]) if name_col else "unknown"
        props["district_number"] = str(props[num_col]) if num_col else "unknown"

    folium.GeoJson(
        districts_geojson,
        name="Vienna Districts",
        style_function=lambda x: {
            "fillColor": "#333333",
            "color": "#666666",
            "weight": 1.2,
            "fillOpacity": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["district_name", "district_number"],
            aliases=["District:", "No.:"],
        )
    ).add_to(m)

    # 3. Build Timestamped GeoJSON Features for agent coordinates.
    base_time = datetime(2026, 7, 7, 20, 0, 0)
    
    # Group by (Agent_ID, Activity) using a standard Python dictionary.
    grouped = {}
    for r in assigned_rows:
        key = (r["Agent_ID"], r["Activity"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    activity_colors = {
        "walk": "#2ca25f",  # green
        "bike": "#3182bd",  # blue
        "stand": "#756bb1", # purple
    }

    features = []
    for (agent_id, activity), group in grouped.items():
        color = activity_colors.get(activity, "#ff7f0e")
        
        # Sort group chronologically.
        group.sort(key=lambda r: r["Timestamp"])
        
        min_ts = group[0]["Timestamp"]
        
        # Sample coordinates at 2-second intervals (2000ms threshold) to keep Leaflet highly responsive.
        last_sampled_ts = -999999
        for r in group:
            ts = r["Timestamp"]
            if ts - last_sampled_ts < 2000:
                continue
            last_sampled_ts = ts
            
            elapsed_sec = (ts - min_ts) / 1000.0
            feature_time = (base_time + timedelta(seconds=elapsed_sec)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(r["Longitude"]), float(r["Latitude"])]
                },
                "properties": {
                    "time": feature_time,
                    "popup": f"Agent {agent_id} ({activity})",
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": color,
                        "fillOpacity": 0.8,
                        "stroke": "true",
                        "color": "#ffffff",
                        "weight": 1.0,
                        "radius": 6
                    }
                }
            })

    # Wrap features in a FeatureCollection.
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    # 4. Add the TimestampedGeoJson plugin to the map.
    TimestampedGeoJson(
        feature_collection,
        period="PT2S",        # 2-second step size
        add_last_point=True,
        auto_play=False,
        loop=True,
        max_speed=5,
        min_speed=0.1,
        duration="PT1S",
        time_slider_drag_update=True
    ).add_to(m)

    # 5. Save and return the HTML file path.
    m.save(str(output_html_path))
    return output_html_path
