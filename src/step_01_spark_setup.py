"""Java and Matplotlib initialisation for local Spark notebooks."""

import os
from pathlib import Path
import sys

# Define default installation paths for Java on macOS systems.
# These candidates are checked sequentially if the JAVA_HOME variable is not preset.
JAVA_CANDIDATES = (
    Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    Path("/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"),
    Path("/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    Path("/usr/local/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"),
)


def configure_spark_environment(project_root: Path | None = None) -> str | None:
    """Set JAVA_HOME and MPLCONFIGDIR environment variables when they are missing.

    This ensures that Spark can locate the Java runtime and Matplotlib can write
    to a project-local cache directory, preventing write permission conflicts.
    """

    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        candidates = list(JAVA_CANDIDATES)
        # On Windows host systems, scan standard installation directories for Java JDKs.
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
                # Prepend Java binary folder to PATH to ensure Spark sub-processes execute correctly.
                os.environ["PATH"] = (
                    str(candidate / "bin") + os.pathsep + os.environ.get("PATH", "")
                )
                break

    if project_root is not None:
        project_root_path = str(Path(project_root).resolve())
        # Establish a local Matplotlib cache directory inside the repository to avoid server permission errors.
        matplotlib_cache = Path(project_root_path) / ".matplotlib_cache"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        # Initialise native Windows binaries (winutils.exe and hadoop.dll) dynamically.
        if os.name == 'nt':
            from src.step_08_bootstrapping import setup_winutils
            setup_winutils(Path(project_root))

        # Insert project root path into PYTHONPATH to allow executor nodes to locate local packages.
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [
            entry for entry in existing_pythonpath.split(os.pathsep) if entry
        ]
        if project_root_path not in pythonpath_entries:
            pythonpath_entries.insert(0, project_root_path)
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return java_home


def spark_pythonpath_configs(project_root: Path) -> dict[str, str]:
    """Provide Spark configurations so Python executor processes can import project modules.

    This binds the project root path to Spark's driver and executor environments.
    """

    project_root_path = str(Path(project_root).resolve())
    return {
        "spark.executorEnv.PYTHONPATH": project_root_path,
        "spark.driverEnv.PYTHONPATH": project_root_path,
        "spark.pyspark.python": os.environ.get("PYSPARK_PYTHON", sys.executable),
        "spark.pyspark.driver.python": os.environ.get(
            "PYSPARK_DRIVER_PYTHON", sys.executable
        ),
    }

