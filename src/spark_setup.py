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

    try:
        import sys
        debug_path = Path("c:/Users/fedka/Documents/GitHub/Geospatial Repletion & Saturation Modelling/notebook_debug.log")
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- configure_spark_environment called ---\n")
            f.write(f"sys.executable: {sys.executable}\n")
            f.write(f"sys.version: {sys.version}\n")
            f.write(f"JAVA_HOME initially: {os.environ.get('JAVA_HOME')}\n")
            f.write(f"PATH first 3: {os.environ.get('PATH', '').split(os.pathsep)[:3]}\n")
    except Exception as ex:
        pass

    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        candidates = list(JAVA_CANDIDATES)
        if os.name == 'nt':
            for base_dir in (
                Path("C:/Program Files/Eclipse Adoptium"),
                Path("C:/Program Files/Java"),
                Path("C:/Program Files (x86)/Java"),
            ):
                if base_dir.exists() and base_dir.is_dir():
                    for child in base_dir.iterdir():
                        if child.is_dir():
                            candidates.append(child)

        for candidate in candidates:
            java_binary = candidate / "bin" / ("java.exe" if os.name == 'nt' else "java")
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

        # Resolve winutils configuration dynamically for Windows systems
        if os.name == 'nt':
            from src.bootstrapping import setup_winutils
            setup_winutils(Path(project_root))

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [
            entry for entry in existing_pythonpath.split(os.pathsep) if entry
        ]
        if project_root_path not in pythonpath_entries:
            pythonpath_entries.insert(0, project_root_path)
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    try:
        debug_path = Path("c:/Users/fedka/Documents/GitHub/Geospatial Repletion & Saturation Modelling/notebook_debug.log")
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"java_home resolved to: {java_home}\n")
            f.write(f"JAVA_HOME in env finally: {os.environ.get('JAVA_HOME')}\n")
            f.write(f"PATH finally: {os.environ.get('PATH')}\n")
    except Exception:
        pass
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
