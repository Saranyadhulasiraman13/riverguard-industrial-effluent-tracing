"""Reusable cleaning pipeline for RiverGuard's raw CSV datasets."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .data_loader import DATASET_FILENAMES, load_all_datasets


TIMESTAMP_COLUMNS = {
	"sensor_readings": ["timestamp"],
	"flow_direction": ["timestamp"],
	"discharge_schedule": ["discharge_start", "discharge_end"],
	"unit_operations": ["timestamp"],
	"citizen_observations": ["timestamp"],
}

SORT_COLUMNS = {
	"sensor_readings": ["timestamp", "station_id", "parameter"],
	"flow_direction": ["timestamp", "from_station", "to_station"],
	"discharge_schedule": ["discharge_start", "unit_id"],
	"unit_operations": ["timestamp", "unit_id"],
	"citizen_observations": ["timestamp", "location"],
}

UPPERCASE_COLUMNS = {
	"sensors": ["station_id"],
	"sensor_readings": ["station_id"],
	"flow_direction": ["from_station", "to_station"],
	"industrial_units": ["unit_id", "station_id"],
	"discharge_schedule": ["unit_id"],
	"unit_operations": ["unit_id"],
}

LOWERCASE_COLUMNS = {
	"sensor_readings": ["parameter"],
	"flow_direction": ["direction"],
	"discharge_schedule": ["discharge_status"],
	"unit_operations": ["operation_status"],
	"citizen_observations": ["observation_type"],
}


def processed_data_dir() -> Path:
	"""Return the project-relative directory for cleaned CSV output."""
	return Path(__file__).resolve().parents[1] / "data" / "processed"


def _empty_stats(dataset_name: str, frame: pd.DataFrame) -> Dict[str, int | str]:
	"""Create the audit fields recorded for one dataset."""
	return {
		"dataset": dataset_name,
		"input_rows": len(frame),
		"output_rows": 0,
		"missing_before": int(frame.isna().sum().sum()),
		"missing_after": 0,
		"duplicates_before": int(frame.duplicated().sum()),
		"duplicates_removed": 0,
		"duplicates_after": 0,
		"timestamps_fixed": 0,
		"timestamps_removed": 0,
		"numerical_values_imputed": 0,
		"invalid_values_detected": 0,
		"suspicious_extreme_values_retained": 0,
	}


def _normalize_categories(dataset_name: str, frame: pd.DataFrame) -> pd.DataFrame:
	"""Trim text and canonicalize only known IDs and categorical fields."""
	cleaned = frame.copy()
	for column in cleaned.select_dtypes(include="object"):
		cleaned[column] = cleaned[column].map(
			lambda value: value.strip() if isinstance(value, str) else value
		)
	for column in UPPERCASE_COLUMNS.get(dataset_name, []):
		cleaned[column] = cleaned[column].str.upper()
	for column in LOWERCASE_COLUMNS.get(dataset_name, []):
		cleaned[column] = cleaned[column].str.lower()
	return cleaned


def _clean_timestamps(
	dataset_name: str, frame: pd.DataFrame, stats: Dict[str, int | str]
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
	"""Parse timestamps, remove rows missing required time values, and normalize format."""
	cleaned = frame.copy()
	parsed_columns: Dict[str, pd.Series] = {}
	invalid_rows = pd.Series(False, index=cleaned.index)
	for column in TIMESTAMP_COLUMNS.get(dataset_name, []):
		original = cleaned[column].copy()
		parsed = pd.to_datetime(cleaned[column], utc=True, errors="coerce")
		parsed_columns[column] = parsed
		invalid_rows |= parsed.isna()
		normalized = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
		stats["timestamps_fixed"] += int(
			((original.notna()) & parsed.notna() & (original != normalized)).sum()
		)
		cleaned[column] = normalized

	stats["timestamps_removed"] = int(invalid_rows.sum())
	if invalid_rows.any():
		cleaned = cleaned.loc[~invalid_rows].copy()
		parsed_columns = {
			column: values.loc[~invalid_rows] for column, values in parsed_columns.items()
		}
	return cleaned, parsed_columns


def _clean_numeric(
	dataset_name: str, frame: pd.DataFrame, stats: Dict[str, int | str]
) -> pd.DataFrame:
	"""Validate numeric columns and impute only missing sensor readings."""
	cleaned = frame.copy()
	if dataset_name == "sensors":
		numeric_columns = ["latitude", "longitude"]
	elif dataset_name == "sensor_readings":
		numeric_columns = ["value"]
	else:
		numeric_columns = []

	for column in numeric_columns:
		original = cleaned[column].copy()
		numeric = pd.to_numeric(original, errors="coerce")
		stats["invalid_values_detected"] += int(
			(original.notna() & numeric.isna()).sum()
		)
		cleaned[column] = numeric

	if dataset_name == "sensors":
		invalid_latitude = ~cleaned["latitude"].between(-90, 90) & cleaned[
			"latitude"
		].notna()
		invalid_longitude = ~cleaned["longitude"].between(-180, 180) & cleaned[
			"longitude"
		].notna()
		invalid_coordinates = invalid_latitude | invalid_longitude
		stats["invalid_values_detected"] += int(invalid_coordinates.sum())
		cleaned.loc[invalid_coordinates, ["latitude", "longitude"]] = pd.NA

	if dataset_name == "sensor_readings":
		invalid_sensor_values = (cleaned["value"] < 0) & cleaned["value"].notna()
		stats["invalid_values_detected"] += int(invalid_sensor_values.sum())
		cleaned.loc[invalid_sensor_values, "value"] = pd.NA
		missing_before_imputation = int(cleaned["value"].isna().sum())
		if missing_before_imputation:
			group_medians = cleaned.groupby("parameter")["value"].transform("median")
			cleaned["value"] = cleaned["value"].fillna(group_medians)
			cleaned["value"] = cleaned["value"].fillna(cleaned["value"].median())
			stats["numerical_values_imputed"] = int(
				missing_before_imputation - cleaned["value"].isna().sum()
			)

	return cleaned


def _count_extremes(dataset_name: str, frame: pd.DataFrame) -> int:
	"""Count IQR outliers for reporting without removing them."""
	if dataset_name != "sensor_readings" or frame.empty:
		return 0
	values = frame["value"].dropna()
	if values.empty:
		return 0
	lower_quartile = values.quantile(0.25)
	upper_quartile = values.quantile(0.75)
	upper_fence = upper_quartile + 1.5 * (upper_quartile - lower_quartile)
	return int((values > upper_fence).sum())


def clean_dataset(
	dataset_name: str, frame: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int | str]]:
	"""Clean one known RiverGuard dataset and return its audit summary."""
	if dataset_name not in DATASET_FILENAMES:
		raise ValueError(f"Unknown RiverGuard dataset: {dataset_name}")

	stats = _empty_stats(dataset_name, frame)
	cleaned = _normalize_categories(dataset_name, frame)
	cleaned = cleaned.drop_duplicates().copy()
	stats["duplicates_removed"] = int(stats["duplicates_before"])
	cleaned, _ = _clean_timestamps(dataset_name, cleaned, stats)
	cleaned = _clean_numeric(dataset_name, cleaned, stats)

	sort_columns = [
		column for column in SORT_COLUMNS.get(dataset_name, []) if column in cleaned
	]
	if sort_columns:
		cleaned = cleaned.sort_values(sort_columns, kind="stable")
	cleaned = cleaned.reset_index(drop=True)

	stats["output_rows"] = len(cleaned)
	stats["missing_after"] = int(cleaned.isna().sum().sum())
	stats["duplicates_after"] = int(cleaned.duplicated().sum())
	stats["suspicious_extreme_values_retained"] = _count_extremes(
		dataset_name, cleaned
	)
	return cleaned, stats


def run_cleaning_pipeline(
	data_dir: Optional[Path] = None,
	output_dir: Optional[Path] = None,
	summary_path: Optional[Path] = None,
) -> pd.DataFrame:
	"""Clean all raw datasets, write processed copies, and return the summary."""
	destination = Path(output_dir) if output_dir is not None else processed_data_dir()
	destination.mkdir(parents=True, exist_ok=True)
	summary_destination = (
		Path(summary_path)
		if summary_path is not None
		else Path(__file__).resolve().parents[1] / "reports" / "cleaning_summary.csv"
	)
	summary_destination.parent.mkdir(parents=True, exist_ok=True)

	summaries = []
	for dataset_name, frame in load_all_datasets(data_dir=data_dir).items():
		cleaned, stats = clean_dataset(dataset_name, frame)
		cleaned.to_csv(destination / DATASET_FILENAMES[dataset_name], index=False)
		summaries.append(stats)

	summary = pd.DataFrame(summaries)
	summary.to_csv(summary_destination, index=False)
	return summary
