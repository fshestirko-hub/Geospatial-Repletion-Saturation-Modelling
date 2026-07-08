#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}"
export PATH="$JAVA_HOME/bin:$PATH"
export MPLCONFIGDIR="$ROOT/.matplotlib_cache"

source "$ROOT/.venv/bin/activate"

python -m src._create_minimal_telemetry
python - <<'PY'
from pathlib import Path
from pyspark.sql import SparkSession
from src.step_13_geospatial_districts import run_district_assignment

spark = (
    SparkSession.builder
    .appName("Vienna-District-Assignment")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
try:
    outputs = run_district_assignment(spark, Path("."))
    for name, path in outputs.items():
        print(f"{name}: {path}")
finally:
    spark.stop()
PY

echo "Done. Open notebooks/03_district_assignment_routed.ipynb or check data/geospatial_output/"
