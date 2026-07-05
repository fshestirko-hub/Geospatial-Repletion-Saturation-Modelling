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

    spark = (
        SparkSession.builder
        .appName("Minimal-Telemetry-Generator")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    records = []
    for agent_id in range(num_agents):
        activity = ACTIVITIES[agent_id % len(ACTIVITIES)]
        for sample_idx in range(samples_per_agent):
            timestamp = BASE_TIMESTAMP_MS + sample_idx * 10
            phase = sample_idx / 100.0
            records.append(
                (
                    agent_id,
                    timestamp,
                    0.1 * phase,
                    0.2 * phase,
                    9.8 + 0.05 * phase,
                    0.01 * phase,
                    0.02 * phase,
                    0.03 * phase,
                    activity,
                )
            )

    schema = StructType(
        [
            StructField("Agent_ID", LongType(), False),
            StructField("Timestamp", LongType(), False),
            StructField("ax", DoubleType(), False),
            StructField("ay", DoubleType(), False),
            StructField("az", DoubleType(), False),
            StructField("gx", DoubleType(), False),
            StructField("gy", DoubleType(), False),
            StructField("gz", DoubleType(), False),
            StructField("Activity", StringType(), False),
        ]
    )

    telemetry_df = spark.createDataFrame(records, schema=schema)
    telemetry_df.write.mode("overwrite").parquet(str(output_path))

    row_count = telemetry_df.count()
    logging.info("Wrote %s rows to %s", row_count, output_path)
    return output_path


if __name__ == "__main__":
    from src.bootstrapping import get_project_root

    create_minimal_telemetry(get_project_root())
