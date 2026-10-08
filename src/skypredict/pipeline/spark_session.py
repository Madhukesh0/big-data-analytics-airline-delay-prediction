"""Spark session management.

PySpark runs in local mode by default (``local[*]``); the master can be
changed with the ``SKYPREDICT_SPARK_MASTER`` environment variable. A
project-local Java 17 runtime (downloaded into ``<project>/runtime``) is
picked up automatically when no system Java is configured.
"""

from __future__ import annotations

import os
import shutil
import sys

from ..config import LOCAL_JRE_DIR, SPARK_MASTER

JAVA_INSTRUCTIONS = (
    "Java was not found. PySpark needs Java 17. "
    "Either install Java 17 system-wide, or run the provided setup to place a "
    "JRE under the project's runtime/ directory "
    "(see README, section 'Java / PySpark requirements')."
)


class SparkUnavailableError(RuntimeError):
    """Raised when Spark cannot start, with clear setup instructions."""


def _configure_local_java() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    candidates = sorted(d for d in LOCAL_JRE_DIR.glob("jdk*") if (d / "bin").exists())
    if candidates:
        java_home = str(candidates[0].resolve())
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ.get("PATH", "")


def _configure_local_hadoop() -> None:
    """Point Hadoop at the bundled winutils on Windows.

    Writing Parquet on Windows requires winutils.exe/hadoop.dll; without them
    Spark fails with 'HADOOP_HOME and hadoop.home.dir are unset'.
    """
    if os.name != "nt":
        return
    hadoop_dir = LOCAL_JRE_DIR / "hadoop"
    winutils = hadoop_dir / "bin" / "winutils.exe"
    if not winutils.exists():
        return
    os.environ.setdefault("HADOOP_HOME", str(hadoop_dir.resolve()))
    os.environ.setdefault("hadoop.home.dir", str(hadoop_dir.resolve()))
    bin_dir = str((hadoop_dir / "bin").resolve())
    path = os.environ.get("PATH", "")
    if bin_dir not in path:
        os.environ["PATH"] = bin_dir + os.pathsep + path


def java_available() -> bool:
    _configure_local_java()
    return shutil.which("java") is not None


def check_spark_available() -> tuple[bool, str]:
    """Return (available, message). Never raises."""
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False, (
            "PySpark is not installed in this Python environment. "
            "Install it with: pip install pyspark"
        )
    if not java_available():
        return False, JAVA_INSTRUCTIONS
    return True, "Spark available"


def get_spark(master: str | None = None, app_name: str = "SkyPredict"):
    """Create (or reuse) a local SparkSession, or raise with instructions."""
    available, message = check_spark_available()
    if not available:
        raise SparkUnavailableError(message)

    # Spark's Python workers must use THIS interpreter (on Windows a bare
    # `python` on PATH often resolves to the Microsoft Store stub).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    _configure_local_hadoop()

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master(master or SPARK_MASTER)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
