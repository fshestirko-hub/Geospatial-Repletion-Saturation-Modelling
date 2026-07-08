# Geospatial Repletion & Saturation Modelling in Vienna

This repository implements a distributed, scalable big data architecture capable of simulating and forecasting urban spatio-temporal crowd dynamics across Vienna's municipal transit networks. The system ingests raw high-frequency mobile sensor signals, resamples them to synthetically scale the agent population, Snaps the coordinates to municipal Open Government Data (OGD) infrastructure layers, and trains supervised regression models in Apache Spark to forecast urban bottlenecks 30–60 minutes in advance.

---

## 1. Pipeline and Notebook Mapping

The execution workflow is structured sequentially into five production notebooks, located in the `notebooks/` directory:

1.  **[`01_part_master_notebook.ipynb`](notebooks/01_part_master_notebook.ipynb)**: Exploratory data analysis, 3D signal phase-space attractor visualization, signal de-noising, and nonparametric block bootstrap resampling.
2.  **[`02_geospatial_streaming_routed.ipynb`](notebooks/02_geospatial_streaming_routed.ipynb)**: Structured Streaming coordinate projection using local tangent plane flat-earth approximations and real-time micro-batch GeoJSON exports.
3.  **[`03_district_assignment_routed.ipynb`](notebooks/03_district_assignment_routed.ipynb)**: Spatial joining, snapping biking/walking agent vectors onto Vienna's infrastructure, and performing point-in-polygon district clustering utilizing driver-side packed R-Trees (`STRtree`).
4.  **[`04_predictive_modelling_new.ipynb`](notebooks/04_predictive_modelling_new.ipynb)**: Predictive modeling using PySpark MLlib. Engineering feature lag windows and rolling stats, preventing temporal data leakage, and training a Gradient Boosted Trees (GBT) Regressor to forecast saturation indexes.
5.  **[`05_geospatial_forecasting_dashboard.ipynb`](notebooks/05_geospatial_forecasting_dashboard.ipynb)**: Interactive Folium/Leaflet visualization map dashboard plotting predicted saturation bottlenecks over time.

---

## 2. Technical Stack and Core Components

*   **Data Processing**: Apache Spark SQL & PySpark MLlib (for big data transformations and model training).
*   **Real-time Processing**: Spark Structured Streaming (micro-batch pipeline mapping kinetic offsets).
*   **Geospatial Tracking**: Shapely STRtrees & Folium/Leaflet mapping.
*   **Resampling Engine**: Overlapping Block Bootstrap Resampling ($B=100$, overlap $p=5$) to preserve temporal correlation.
*   **Format**: Columnar, compressed Parquet storage formats.

---

## 3. Installation and Setup

### 3.1. Prerequisites
Ensure Python 3.10+ and Java JDK 17 (required for Spark JVM) are installed. 

### 3.2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3.3. Windows Bootstrapping (Winutils)
For local Windows execution, Spark requires native Hadoop binaries. The bootstrap configuration is handled automatically by:
```python
from src.step_08_bootstrapping import setup_winutils
setup_winutils(PROJECT_ROOT)
```
This maps the required `winutils.exe` and `hadoop.dll` binaries located in `data/winutils/bin/`.

---

## 4. Academic Citations and Licensing
Refer to the **[`dataset_licenses.md`](dataset_licenses.md)** file for full details of academic datasets, public OGD municipal sources, and licensing designations (CC BY 4.0, CC BY 4.0 AT, ODbL).
