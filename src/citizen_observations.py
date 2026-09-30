"""Optional ingestion and event association for verified citizen observations."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATIONS_PATH = (
	PROJECT_ROOT / "data" / "raw" / "citizen_science_observations.csv"
)
OBSERVATION_COLUMNS = [
	"observation_id",
	"observation_time",
	"station_id",
	"observation_location",
	"observation_type",
	"observation_value",
	"observation_description",
	"latitude",
	"longitude",
]
REQUIRED_COLUMNS = {"observation_id", "observation_time", "observation_type"}


def _present(values: pd.Series) -> pd.Series:
	return values.notna() & values.astype("string").str.strip().ne("").fillna(False)


def validate_observations(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
	"""Validate optional observation rows while retaining all submitted records."""
	missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
	if missing_columns:
		raise ValueError(
			f"Citizen observations are missing required columns: {sorted(missing_columns)}"
		)
	location_columns = {"station_id", "observation_location"}.intersection(frame.columns)
	coordinate_columns = {"latitude", "longitude"}.issubset(frame.columns)
	if not location_columns and not coordinate_columns:
		raise ValueError(
			"Citizen observations require station_id, observation_location, or both latitude and longitude."
		)

	validated = frame.copy()
	for column in OBSERVATION_COLUMNS:
		if column not in validated:
			validated[column] = pd.NA

	issues = [[] for _ in range(len(validated))]
	missing_required_values = pd.Series(False, index=validated.index)
	for column in sorted(REQUIRED_COLUMNS):
		missing = ~_present(validated[column])
		missing_required_values |= missing
		for index in validated.index[missing]:
			issues[validated.index.get_loc(index)].append(f"missing_{column}")

	validated["observation_time_utc"] = pd.to_datetime(
		validated["observation_time"], utc=True, errors="coerce"
	)
	invalid_timestamps = validated["observation_time_utc"].isna()
	for position in range(len(validated)):
		if invalid_timestamps.iloc[position]:
			issues[position].append("invalid_observation_time")

	observation_ids = validated["observation_id"].astype("string").str.strip()
	duplicate_ids = observation_ids.notna() & observation_ids.ne("") & observation_ids.duplicated(keep=False)
	for position in range(len(validated)):
		if duplicate_ids.iloc[position]:
			issues[position].append("duplicate_observation_id")

	station_present = _present(validated["station_id"])
	location_present = _present(validated["observation_location"])
	latitude_present = _present(validated["latitude"])
	longitude_present = _present(validated["longitude"])
	latitude = pd.to_numeric(validated["latitude"], errors="coerce")
	longitude = pd.to_numeric(validated["longitude"], errors="coerce")
	coordinates_present = latitude_present | longitude_present
	coordinates_valid = (
		latitude_present
		& longitude_present
		& latitude.between(-90, 90)
		& longitude.between(-180, 180)
	)
	invalid_coordinates = coordinates_present & ~coordinates_valid
	missing_location = ~(station_present | location_present | coordinates_valid)
	for position in range(len(validated)):
		if invalid_coordinates.iloc[position]:
			issues[position].append("invalid_coordinates")
		if missing_location.iloc[position]:
			issues[position].append("missing_location")

	value_present = _present(validated["observation_value"])
	value_numeric = pd.to_numeric(validated["observation_value"], errors="coerce")
	invalid_values = value_present & value_numeric.isna()
	for position in range(len(validated)):
		if invalid_values.iloc[position]:
			issues[position].append("invalid_observation_value")

	validated["observation_value_numeric"] = value_numeric
	validated["latitude_numeric"] = latitude
	validated["longitude_numeric"] = longitude
	validated["validation_issues"] = [";".join(row_issues) for row_issues in issues]
	validated["is_valid"] = validated["validation_issues"].eq("")
	validated = validated.sort_values(
		"observation_time_utc", kind="stable", na_position="last"
	).reset_index(drop=True)

	duplicate_id_count = int(duplicate_ids.sum())
	invalid_value_count = int(invalid_values.sum())
	invalid_coordinate_count = int(invalid_coordinates.sum())
	status: Dict[str, object] = {
		"status": "available",
		"observations_available": True,
		"input_rows": len(frame),
		"valid_rows": int(validated["is_valid"].sum()),
		"invalid_rows": int((~validated["is_valid"]).sum()),
		"missing_required_values": int(missing_required_values.sum()),
		"invalid_timestamps": int(invalid_timestamps.sum()),
		"duplicate_observation_id_rows": duplicate_id_count,
		"invalid_numeric_values": invalid_value_count,
		"invalid_coordinate_rows": invalid_coordinate_count,
		"missing_location_rows": int(missing_location.sum()),
		"message": "Observations are supporting context, not proof of a pollution source.",
	}
	return validated, status


def load_optional_observations(
	path: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
	"""Load the optional citizen-science file, or report its absence without failing."""
	source = Path(path) if path is not None else DEFAULT_OBSERVATIONS_PATH
	if not source.is_file():
		return None, {
			"status": "unavailable",
			"observations_available": False,
			"input_rows": 0,
			"valid_rows": 0,
			"invalid_rows": 0,
			"source_file": str(source),
			"message": "Citizen-science observations are not currently available.",
		}

	try:
		frame = pd.read_csv(source)
		validated, status = validate_observations(frame)
	except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
		return None, {
			"status": "invalid",
			"observations_available": False,
			"input_rows": 0,
			"valid_rows": 0,
			"invalid_rows": 0,
			"source_file": str(source),
			"message": str(error),
		}

	status["source_file"] = str(source)
	return validated, status


def associate_observations_with_events(
	observations: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
	"""Associate valid station observations falling inside an event's time window."""
	validated, _ = validate_observations(observations)
	required_event_columns = {"event_id", "station_id", "start_time", "end_time"}
	missing_columns = required_event_columns.difference(events.columns)
	if missing_columns:
		raise ValueError(
			f"Pollution events are missing required columns: {sorted(missing_columns)}"
		)

	parsed_events = events.copy()
	parsed_events["start_time"] = pd.to_datetime(
		parsed_events["start_time"], utc=True, errors="coerce"
	)
	parsed_events["end_time"] = pd.to_datetime(
		parsed_events["end_time"], utc=True, errors="coerce"
	)
	output_rows = []
	for observation in validated.to_dict("records"):
		matches = pd.DataFrame()
		if observation["is_valid"] and pd.notna(observation["station_id"]):
			matches = parsed_events[
				(parsed_events["station_id"].astype(str) == str(observation["station_id"]))
				& (parsed_events["start_time"] <= observation["observation_time_utc"])
				& (parsed_events["end_time"] >= observation["observation_time_utc"])
				& parsed_events["start_time"].notna()
				& parsed_events["end_time"].notna()
			]
		if matches.empty:
			row = dict(observation)
			row["event_id"] = pd.NA
			row["association_status"] = "not associated with an event"
			output_rows.append(row)
		else:
			for event_id in matches["event_id"]:
				row = dict(observation)
				row["event_id"] = event_id
				row["association_status"] = "supporting observation; not proof of pollution source"
				output_rows.append(row)
	if not output_rows:
		result = validated.copy()
		result["event_id"] = pd.Series(dtype="object")
		result["association_status"] = pd.Series(dtype="object")
		return result
	return pd.DataFrame(output_rows)


def write_observation_reports(
	events: pd.DataFrame,
	reports_dir: Optional[Path] = None,
	observations_path: Optional[Path] = None,
) -> Dict[str, Path]:
	"""Write an availability report and, only when present, validated observations."""
	destination = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "reports"
	destination.mkdir(parents=True, exist_ok=True)
	observations, status = load_optional_observations(observations_path)
	status_path = destination / "citizen_observations_status.csv"
	pd.DataFrame([status]).to_csv(status_path, index=False)
	outputs = {"status_path": status_path}
	if observations is not None:
		associated = associate_observations_with_events(observations, events)
		observations_report_path = destination / "citizen_observations.csv"
		associated.to_csv(observations_report_path, index=False)
		outputs["observations_path"] = observations_report_path
	return outputs