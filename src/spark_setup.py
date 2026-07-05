"""Java/Matplotlib setup for local Spark notebooks."""

import os
from pathlib import Path
import sys

JAVA_CANDIDATES = (
    Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    Path("/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"),
    Path("/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    Path("/usr/local/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"),
)


def configure_spark_environment(project_root: Path | None = None) -> str | None:
    """Set JAVA_HOME and MPLCONFIGDIR when they are missing."""

    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        for candidate in JAVA_CANDIDATES:
            java_binary = candidate / "bin" / "java"
            if java_binary.exists():
                java_home = str(candidate)
                os.environ["JAVA_HOME"] = java_home
                os.environ["PATH"] = (
                    str(candidate / "bin") + os.pathsep + os.environ.get("PATH", "")
                )
                break

    if project_root is not None:
        project_root_path = str(Path(project_root).resolve())
        matplotlib_cache = Path(project_root_path) / ".matplotlib_cache"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [
            entry for entry in existing_pythonpath.split(os.pathsep) if entry
        ]
        if project_root_path not in pythonpath_entries:
            pythonpath_entries.insert(0, project_root_path)
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return java_home


def spark_pythonpath_configs(project_root: Path) -> dict[str, str]:
    """Spark configs so Python workers can import project modules."""

    project_root_path = str(Path(project_root).resolve())
    return {
        "spark.executorEnv.PYTHONPATH": project_root_path,
        "spark.driverEnv.PYTHONPATH": project_root_path,
        "spark.pyspark.python": os.environ.get("PYSPARK_PYTHON", sys.executable),
        "spark.pyspark.driver.python": os.environ.get(
            "PYSPARK_DRIVER_PYTHON", sys.executable
        ),
    }
