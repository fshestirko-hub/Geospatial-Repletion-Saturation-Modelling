"""Small test parquet when bootstrapping output is not available yet."""

import logging
from pathlib import Path

from pyspark.sql import SparkSession
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

BASE_TIMESTAMP_MS = 1_700_000_000_000
ACTIVITIES = ["walk", "bike", "stand"]


def create_minimal_telemetry(
    project_root: Path,
    num_agents: int = 10,
    samples_per_agent: int = 200,
) -> Path:
    """Write a small Parquet file compatible with the geospatial pipeline."""

    project_root = Path(project_root)
    output_path = project_root / "data" / "processed" / "synthetic_telemetry.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean existing path if it exists to avoid writing conflicts
    if output_path.exists():
        import shutil
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    import pandas as pd

    records = []
    for agent_id in range(num_agents):
        activity = ACTIVITIES[agent_id % len(ACTIVITIES)]
        for sample_idx in range(samples_per_agent):
            timestamp = BASE_TIMESTAMP_MS + sample_idx * 10
            phase = sample_idx / 100.0
            records.append(
                {
                    "Agent_ID": int(agent_id),
                    "Timestamp": int(timestamp),
                    "ax": float(0.1 * phase),
                    "ay": float(0.2 * phase),
                    "az": float(9.8 + 0.05 * phase),
                    "gx": float(0.01 * phase),
                    "gy": float(0.02 * phase),
                    "gz": float(0.03 * phase),
                    "Activity": str(activity),
                }
            )

    df = pd.DataFrame(records)
    
    # Force specific schemas to match exactly what Spark expects
    df["Agent_ID"] = df["Agent_ID"].astype("int64")
    df["Timestamp"] = df["Timestamp"].astype("int64")
    df["ax"] = df["ax"].astype("float64")
    df["ay"] = df["ay"].astype("float64")
    df["az"] = df["az"].astype("float64")
    df["gx"] = df["gx"].astype("float64")
    df["gy"] = df["gy"].astype("float64")
    df["gz"] = df["gz"].astype("float64")
    df["Activity"] = df["Activity"].astype("string")

    df.to_parquet(str(output_path), engine="pyarrow", index=False)

    logging.info("Wrote %s rows to %s", len(df), output_path)
    return output_path


if __name__ == "__main__":
    from src.step_08_bootstrapping import get_project_root

    create_minimal_telemetry(get_project_root())
