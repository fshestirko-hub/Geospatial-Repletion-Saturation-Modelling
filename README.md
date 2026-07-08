# Geospatial Repletion & Saturation Modelling

Group project for **Data Processing 2** (City 5).

Spark pipeline for UCI HHAR sensor data, synthetic telemetry, and Vienna spatial layers.

## Setup (macOS)

### Java

```bash
brew install openjdk@17
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

### Python

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run geospatial notebook

```text
notebooks/03_district_assignment_routed.ipynb
```

Needs `data/processed/synthetic_telemetry.parquet` (from bootstrapping). If missing, the notebook creates a small test file.

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_part_master_notebook.ipynb` | EDA, bootstrapping, streaming |
| `02_geospatial_streaming_routed.ipynb` | GeoJSON export |
| `03_district_assignment_routed.ipynb` | Spatial joins, maps, CSV output |

## Outputs (phase 3)

- `data/geospatial_output/district_activity_counts.csv`
- `data/geospatial_output/infrastructure_activity_counts.csv`
- `data/geospatial_output/district_choropleth.png`
- `data/geospatial_output/telemetry_scatter_map.png`
- `data/spatial/` — Vienna OGD layers + agent anchors

## Notes

- HHAR has no GPS; Vienna coordinates are generated for the simulation.
- Vienna open data: CC BY 4.0 AT (City of Vienna).

## Team

- Anna-Nadiia Rabich
- Fedir Shestirko
- Paul Andreas Sommer
- Daniil Volkov
