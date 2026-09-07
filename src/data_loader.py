"""Load RiverGuard's raw CSV datasets without modifying them."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


DATASET_FILENAMES = {
    "sensors": "sensors.csv",
    "sensor_readings": "sensor_readings.csv",
    "flow_direction": "flow_direction.csv",
    "industrial_units": "industrial_units.csv",
    "discharge_schedule": "discharge_schedule.csv",
    "unit_operations": "unit_operations.csv",
    "citizen_observations": "citizen_observations.csv",
}


def raw_data_dir() -> Path:
    """Return the raw-data directory relative to the project package."""
    return Path(__file__).resolve().parents[1] / "data" / "raw"


def load_dataset(dataset_name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load one known raw dataset by its logical name.

    Raises:
        ValueError: If ``dataset_name`` is not one of the known datasets.
        FileNotFoundError: If the expected CSV file is missing.
    """
    if dataset_name not in DATASET_FILENAMES:
        available = ", ".join(sorted(DATASET_FILENAMES))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}."
        )

    directory = Path(data_dir) if data_dir is not None else raw_data_dir()
    path = directory / DATASET_FILENAMES[dataset_name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing RiverGuard dataset: {path}. "
            f"Expected '{DATASET_FILENAMES[dataset_name]}' under {directory}."
        )

    return pd.read_csv(path)


def load_all_datasets(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load every known raw dataset keyed by its logical dataset name."""
    return {
        dataset_name: load_dataset(dataset_name, data_dir=data_dir)
        for dataset_name in DATASET_FILENAMES
    }


def load_sensors(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load monitoring-station metadata."""
    return load_dataset("sensors", data_dir=data_dir)


def load_sensor_readings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load sensor readings."""
    return load_dataset("sensor_readings", data_dir=data_dir)


def load_flow_direction(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load flow-direction records."""
    return load_dataset("flow_direction", data_dir=data_dir)


def load_industrial_units(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load industrial-unit metadata."""
    return load_dataset("industrial_units", data_dir=data_dir)


def load_discharge_schedule(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load industrial discharge schedules."""
    return load_dataset("discharge_schedule", data_dir=data_dir)


def load_unit_operations(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load unit operating events."""
    return load_dataset("unit_operations", data_dir=data_dir)


def load_citizen_observations(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load citizen observations."""
    return load_dataset("citizen_observations", data_dir=data_dir)
