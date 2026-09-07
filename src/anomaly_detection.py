"""Explainable baseline and historical anomaly detection for sensor readings."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


BASELINE_TURBIDITY_THRESHOLD = 10.0
ROLLING_WINDOW = 3
ROLLING_MIN_PERIODS = 2
EVENT_MAX_GAP_HOURS = 6


def _project_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _default_sensor_readings_path() -> Path:
	processed_path = _project_root() / "data" / "processed" / "sensor_readings.csv"
	if processed_path.is_file():
		return processed_path
	return _project_root() / "data" / "raw" / "sensor_readings.csv"


def load_sensor_readings(path: Optional[Path] = None) -> pd.DataFrame:
	"""Load cleaned sensor readings, falling back to the raw CSV if needed."""
	readings_path = Path(path) if path is not None else _default_sensor_readings_path()
	if not readings_path.is_file():
		raise FileNotFoundError(f"Sensor readings file not found: {readings_path}")
	readings = pd.read_csv(readings_path)
	required = {"timestamp", "station_id", "parameter", "value"}
	missing_columns = required.difference(readings.columns)
	if missing_columns:
		raise ValueError(
			f"Sensor readings are missing required columns: {sorted(missing_columns)}"
		)
	readings["timestamp"] = pd.to_datetime(
		readings["timestamp"], utc=True, errors="coerce"
	)
	readings["value"] = pd.to_numeric(readings["value"], errors="coerce")
	readings = readings.dropna(subset=["timestamp", "station_id", "parameter", "value"])
	return readings.sort_values(
		["station_id", "parameter", "timestamp"], kind="stable"
	).reset_index(drop=True)


def select_pollution_parameter(readings: pd.DataFrame) -> str:
	"""Select turbidity when available, otherwise the most populated numeric parameter."""
	parameters = readings["parameter"].dropna().astype(str)
	turbidity = parameters[parameters.str.contains("turbidity", case=False)]
	if not turbidity.empty:
		return turbidity.iloc[0]

	numeric_counts = (
		readings.assign(value_numeric=pd.to_numeric(readings["value"], errors="coerce"))
		.dropna(subset=["value_numeric"])
		.groupby("parameter")
		.size()
	)
	if numeric_counts.empty:
		raise ValueError("No numeric pollution-related sensor parameter is available")
	return str(numeric_counts.sort_values(ascending=False).index[0])


def _baseline_threshold(parameter: str, values: pd.Series) -> float:
	"""Return a transparent threshold for the selected parameter."""
	if parameter.lower().find("turbidity") >= 0:
		return BASELINE_TURBIDITY_THRESHOLD
	lower_quartile = values.quantile(0.25)
	upper_quartile = values.quantile(0.75)
	return float(upper_quartile + 1.5 * (upper_quartile - lower_quartile))


def run_baseline_detector(
	readings: Optional[pd.DataFrame] = None,
	output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str, float]:
	"""Run a fixed or historical threshold baseline without contextual data."""
	data = load_sensor_readings() if readings is None else readings.copy()
	parameter = select_pollution_parameter(data)
	selected = data[data["parameter"] == parameter].copy()
	threshold = _baseline_threshold(parameter, selected["value"])
	selected["observed_value"] = selected["value"]
	selected["anomaly"] = selected["observed_value"] > threshold
	selected["anomaly_score"] = (
		(selected["observed_value"] - threshold) / threshold
	).clip(lower=0.0)
	result = selected[
		["timestamp", "station_id", "parameter", "observed_value", "anomaly", "anomaly_score"]
	].sort_values(["timestamp", "station_id"], kind="stable")
	if output_path is not None:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		result.assign(timestamp=result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(
			output_path, index=False
		)
	return result.reset_index(drop=True), parameter, threshold


def _severity(score: float) -> str:
	if score >= 10:
		return "critical"
	if score >= 5:
		return "high"
	if score >= 3:
		return "moderate"
	return "low"


def _historical_anomalies(readings: pd.DataFrame, parameter: str) -> pd.DataFrame:
	"""Calculate shifted rolling-IQR anomalies so current values do not set their baseline."""
	selected = readings[readings["parameter"] == parameter].copy()
	grouped = selected.groupby(["station_id", "parameter"], group_keys=False)
	selected["expected_baseline"] = grouped["value"].transform(
		lambda values: values.shift(1).rolling(
			ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
		).median()
	)
	selected["rolling_q1"] = grouped["value"].transform(
		lambda values: values.shift(1).rolling(
			ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
		).quantile(0.25)
	)
	selected["rolling_q3"] = grouped["value"].transform(
		lambda values: values.shift(1).rolling(
			ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
		).quantile(0.75)
	)
	selected["rolling_iqr"] = selected["rolling_q3"] - selected["rolling_q1"]
	selected["scale"] = selected["rolling_iqr"].where(
		selected["rolling_iqr"] > 0,
		selected["expected_baseline"].abs() * 0.1,
	)
	selected["scale"] = selected["scale"].fillna(1.0).clip(lower=1e-9)
	iqr_fence = selected["rolling_q3"] + 1.5 * selected["rolling_iqr"]
	relative_fence = selected["expected_baseline"] * 1.5
	selected["upper_fence"] = pd.concat(
		[iqr_fence, relative_fence], axis=1
	).max(axis=1)
	selected["anomaly"] = (
		selected["expected_baseline"].notna()
		& (selected["value"] > selected["upper_fence"])
	)
	selected["anomaly_score"] = (
		(selected["value"] - selected["expected_baseline"]) / selected["scale"]
	).where(selected["anomaly"], 0.0)
	selected["severity"] = selected["anomaly_score"].map(_severity)
	selected["reason"] = selected.apply(
		lambda row: (
			f"Observed {row['parameter']} is substantially above the recent "
			"historical baseline."
			if row["anomaly"]
			else "Within the recent historical baseline."
		),
		axis=1,
	)
	return selected


def _assign_pollution_events(anomalies: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Group nearby anomalies at the same station and parameter into events."""
	if anomalies.empty:
		empty_events = pd.DataFrame(
			columns=[
				"event_id", "start_time", "end_time", "station_id", "peak_value",
				"affected_parameter", "duration", "severity",
			]
		)
		return anomalies.assign(event_id=pd.Series(dtype="object")), empty_events

	ordered = anomalies.sort_values(
		["station_id", "parameter", "timestamp"], kind="stable"
	).copy()
	gaps = ordered.groupby(["station_id", "parameter"])["timestamp"].diff()
	starts_new_event = gaps.isna() | (gaps > pd.Timedelta(hours=EVENT_MAX_GAP_HOURS))
	ordered["event_number"] = starts_new_event.groupby(
		[ordered["station_id"], ordered["parameter"]]
	).cumsum()
	event_groups = ordered.groupby(
		["station_id", "parameter", "event_number"], sort=False
	)
	event_rows = []
	event_key = {}
	for event_index, ((station_id, parameter, _), group) in enumerate(
		event_groups, start=1
	):
		event_number = int(group["event_number"].iloc[0])
		event_id = f"PE{event_index:04d}"
		peak = group.loc[group["observed_value"].idxmax()]
		start_time = group["timestamp"].min()
		end_time = group["timestamp"].max()
		event_key[(station_id, parameter, event_number)] = event_id
		event_rows.append(
			{
				"event_id": event_id,
				"start_time": start_time,
				"end_time": end_time,
				"station_id": station_id,
				"peak_value": group["observed_value"].max(),
				"affected_parameter": parameter,
				"duration": end_time - start_time,
				"severity": peak["severity"],
			}
		)
	events = pd.DataFrame(event_rows)
	ordered["event_id"] = [
		event_key[(row.station_id, row.parameter, row.event_number)]
		for row in ordered.itertuples()
	]
	return ordered.drop(columns=["event_number"]), events


def run_anomaly_pipeline(
	readings: Optional[pd.DataFrame] = None,
	report_path: Optional[Path] = None,
) -> Dict[str, object]:
	"""Run baseline detection, historical detection, event grouping, and reporting."""
	data = load_sensor_readings() if readings is None else readings.copy()
	parameter = select_pollution_parameter(data)
	baseline, _, threshold = run_baseline_detector(data)
	historical = _historical_anomalies(data, parameter)
	anomalies = historical[historical["anomaly"]].copy()
	anomalies["observed_value"] = anomalies["value"]
	anomalies, events = _assign_pollution_events(anomalies)
	report_columns = [
		"event_id", "timestamp", "station_id", "parameter", "observed_value",
		"expected_baseline", "anomaly_score", "severity", "reason",
	]
	report = anomalies[report_columns].sort_values(
		["timestamp", "station_id"], kind="stable"
	)
	if report_path is None:
		report_path = _project_root() / "reports" / "anomaly_report.csv"
	report_path = Path(report_path)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report.assign(timestamp=report["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(
		report_path, index=False
	)
	events_path = report_path.parent / "pollution_events.csv"
	events.assign(
		start_time=events["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
		end_time=events["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
		duration=events["duration"].astype(str),
	).to_csv(events_path, index=False)
	baseline_path = report_path.parent / "baseline_anomalies.csv"
	baseline.assign(
		timestamp=baseline["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
	).to_csv(baseline_path, index=False)
	return {
		"baseline": baseline,
		"improved_anomalies": report,
		"pollution_events": events,
		"parameter": parameter,
		"baseline_threshold": threshold,
		"baseline_path": baseline_path,
		"report_path": report_path,
		"events_path": events_path,
	}
