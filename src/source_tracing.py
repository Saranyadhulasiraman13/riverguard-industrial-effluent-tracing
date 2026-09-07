"""Explainable ranking of plausible industrial sources for inspection."""

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


REQUIRED_WEIGHTS = {
	"upstream_weight",
	"discharge_timing_weight",
	"operation_weight",
	"sensor_relationship_weight",
}
DEFAULT_THRESHOLDS = {
	"high_confidence_threshold": 0.75,
	"medium_confidence_threshold": 0.45,
}


def _project_root() -> Path:
	return Path(__file__).resolve().parents[1]


def load_weights(path: Optional[Path] = None) -> Dict[str, float]:
	"""Load and validate configurable source-ranking weights and thresholds."""
	weights_path = Path(path) if path is not None else _project_root() / "config" / "weights.yaml"
	if not weights_path.is_file():
		raise FileNotFoundError(f"Source-ranking weights file not found: {weights_path}")
	with weights_path.open("r", encoding="utf-8") as stream:
		configuration = yaml.safe_load(stream) or {}
	missing = REQUIRED_WEIGHTS.difference(configuration)
	if missing:
		raise ValueError(f"Weights file is missing keys: {sorted(missing)}")
	weights = {key: float(configuration[key]) for key in REQUIRED_WEIGHTS}
	total = sum(weights.values())
	if total <= 0 or abs(total - 1.0) > 1e-9:
		raise ValueError("Source-ranking weights must be positive and sum to 1.0")
	weights.update(
		{
			key: float(configuration.get(key, value))
			for key, value in DEFAULT_THRESHOLDS.items()
		}
	)
	return weights


def _load_csv(data_dir: Path, filename: str, required_columns: Iterable[str]) -> pd.DataFrame:
	path = data_dir / filename
	if not path.is_file():
		raise FileNotFoundError(f"Required source-tracing dataset not found: {path}")
	frame = pd.read_csv(path)
	missing = set(required_columns).difference(frame.columns)
	if missing:
		raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
	return frame


def load_source_tracing_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
	"""Load the actual processed CSV inputs used by source tracing."""
	directory = Path(data_dir) if data_dir is not None else _project_root() / "data" / "processed"
	datasets = {
		"flow": _load_csv(
			directory,
			"flow_direction.csv",
			["timestamp", "from_station", "to_station", "direction"],
		),
		"units": _load_csv(
			directory, "industrial_units.csv", ["unit_id", "unit_name", "station_id", "industry_type"]
		),
		"schedules": _load_csv(
			directory,
			"discharge_schedule.csv",
			["unit_id", "discharge_start", "discharge_end", "discharge_status"],
		),
		"operations": _load_csv(
			directory, "unit_operations.csv", ["unit_id", "timestamp", "operation_status"]
		),
	}
	datasets["flow"]["timestamp"] = pd.to_datetime(datasets["flow"]["timestamp"], utc=True)
	datasets["schedules"]["discharge_start"] = pd.to_datetime(
		datasets["schedules"]["discharge_start"], utc=True
	)
	datasets["schedules"]["discharge_end"] = pd.to_datetime(
		datasets["schedules"]["discharge_end"], utc=True
	)
	datasets["operations"]["timestamp"] = pd.to_datetime(
		datasets["operations"]["timestamp"], utc=True
	)
	return datasets


def load_pollution_events(path: Optional[Path] = None) -> pd.DataFrame:
	"""Load the event report produced by the anomaly detector."""
	events_path = Path(path) if path is not None else _project_root() / "reports" / "pollution_events.csv"
	if not events_path.is_file():
		raise FileNotFoundError(f"Pollution event report not found: {events_path}")
	events = pd.read_csv(events_path)
	required = {
		"event_id",
		"start_time",
		"end_time",
		"station_id",
		"peak_value",
		"affected_parameter",
	}
	missing = required.difference(events.columns)
	if missing:
		raise ValueError(f"Pollution events are missing columns: {sorted(missing)}")
	events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
	events["end_time"] = pd.to_datetime(events["end_time"], utc=True)
	return events


def _flow_snapshot(flow: pd.DataFrame, event_time: pd.Timestamp) -> Tuple[pd.DataFrame, bool]:
	"""Select the nearest flow snapshot within six hours and flag missing/conflicting flow."""
	differences = (flow["timestamp"] - event_time).abs()
	if differences.empty or differences.min() > pd.Timedelta(hours=6):
		return flow.iloc[0:0], True
	snapshot_time = flow.loc[differences.idxmin(), "timestamp"]
	snapshot = flow[flow["timestamp"] == snapshot_time].copy()
	conflicting = not snapshot["direction"].str.lower().eq("downstream").all()
	return snapshot, conflicting


def _station_path(snapshot: pd.DataFrame, source_station: str, affected_station: str) -> Optional[List[str]]:
	"""Find a directed downstream path through one flow snapshot."""
	if source_station == affected_station:
		return [source_station]
	graph: Dict[str, List[str]] = {}
	for row in snapshot.itertuples(index=False):
		if str(row.direction).lower() == "downstream":
			graph.setdefault(row.from_station, []).append(row.to_station)
	queue = deque([(source_station, [source_station])])
	visited = {source_station}
	while queue:
		station, path = queue.popleft()
		for next_station in graph.get(station, []):
			if next_station == affected_station:
				return path + [next_station]
			if next_station not in visited:
				visited.add(next_station)
				queue.append((next_station, path + [next_station]))
	return None


def _discharge_match(
	schedules: pd.DataFrame, unit_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[float, str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]]:
	"""Score an overlapping or recently completed scheduled discharge."""
	unit_schedules = schedules[schedules["unit_id"] == unit_id]
	for row in unit_schedules.itertuples(index=False):
		if row.discharge_start <= end and row.discharge_end >= start:
			return 1.0, "Discharge schedule overlaps the pollution event.", (
				row.discharge_start,
				row.discharge_end,
			)
		if row.discharge_end <= start and start - row.discharge_end <= pd.Timedelta(hours=6):
			return 0.8, "Scheduled discharge ended shortly before the pollution event.", (
				row.discharge_start,
				row.discharge_end,
			)
	return 0.0, "No matching scheduled discharge was found.", None


def _operation_match(
	operations: pd.DataFrame,
	unit_id: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	discharge_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Tuple[float, str]:
	"""Score the latest relevant operating status, prioritizing a matched discharge window."""
	unit_operations = operations[operations["unit_id"] == unit_id]
	if discharge_window is not None:
		window_start, window_end = discharge_window
	else:
		window_start = start - pd.Timedelta(hours=6)
		window_end = end + pd.Timedelta(hours=2)
	relevant = unit_operations[
		(unit_operations["timestamp"] >= window_start)
		& (unit_operations["timestamp"] <= window_end)
	].sort_values("timestamp")
	if relevant.empty:
		return 0.0, "No unit operating event was recorded during the relevant period."
	latest = relevant.iloc[-1]
	status = str(latest["operation_status"]).lower()
	if status == "active":
		return 1.0, "Unit operation record shows active status during the relevant period."
	if status == "maintenance":
		return 0.2, "Unit was in maintenance during the relevant period."
	return 0.0, f"Latest unit operation status was '{status}', not active."


def _confidence(score: float, weights: Dict[str, float], flow_uncertain: bool) -> str:
	if flow_uncertain:
		return "Low" if score < weights["medium_confidence_threshold"] else "Medium"
	if score >= weights["high_confidence_threshold"]:
		return "High"
	if score >= weights["medium_confidence_threshold"]:
		return "Medium"
	return "Low"


def _rank_event_candidates(
	event: pd.Series,
	data: Dict[str, pd.DataFrame],
	weights: Dict[str, float],
) -> List[Dict[str, object]]:
	"""Build explainable candidate rows for one pollution event."""
	start = pd.Timestamp(event["start_time"])
	end = pd.Timestamp(event["end_time"])
	snapshot, flow_uncertain = _flow_snapshot(data["flow"], start)
	candidates = []
	for unit in data["units"].itertuples(index=False):
		path = _station_path(snapshot, unit.station_id, event["station_id"])
		if flow_uncertain:
			upstream_score = 0.25
			upstream_reason = "Flow direction was unavailable or conflicting; manual inspection is required."
		elif path is not None and len(path) > 1:
			upstream_score = 1.0
			upstream_reason = "Unit station is upstream of the affected station under the observed flow."
		elif path == [unit.station_id]:
			upstream_score = 0.5
			upstream_reason = "Unit shares the affected monitoring station; no upstream path was established."
		else:
			upstream_score = 0.0
			upstream_reason = "Observed flow does not support transport from the unit station to the affected station."

		discharge_score, discharge_reason, discharge_window = _discharge_match(
			data["schedules"], unit.unit_id, start, end
		)
		operation_score, operation_reason = _operation_match(
			data["operations"], unit.unit_id, start, end, discharge_window
		)
		if path is not None and len(path) > 1 and not flow_uncertain:
			sensor_score = 1.0
			sensor_reason = "Flow snapshot and event timing are consistent with downstream transport."
		elif path == [unit.station_id] and not flow_uncertain:
			sensor_score = 0.5
			sensor_reason = "The unit and affected station share a location, but downstream transport was not established."
		else:
			sensor_score = 0.0
			sensor_reason = "Sensor timing does not establish a supported transport relationship."

		score = (
			weights["upstream_weight"] * upstream_score
			+ weights["discharge_timing_weight"] * discharge_score
			+ weights["operation_weight"] * operation_score
			+ weights["sensor_relationship_weight"] * sensor_score
		)
		confidence = _confidence(score, weights, flow_uncertain or upstream_score < 1.0)
		reasons = [upstream_reason, discharge_reason, operation_reason, sensor_reason]
		if confidence in {"High", "Medium"} and upstream_score == 1.0 and not flow_uncertain:
			conclusion = f"{unit.unit_id} is a {confidence.lower()}-priority plausible source for inspection."
		else:
			conclusion = f"{unit.unit_id} remains a lower-confidence candidate for manual inspection."
		candidates.append(
			{
				"event_id": event["event_id"],
				"source_id": unit.unit_id,
				"affected_station": event["station_id"],
				"upstream_compatibility": upstream_score,
				"discharge_timing_match": discharge_score,
				"operating_event_match": operation_score,
				"sensor_timing_relationship": sensor_score,
				"score": round(score, 6),
				"confidence": confidence,
				"reasons": conclusion + " " + " ".join(reasons),
			}
		)
	return candidates


def run_source_tracing(
	events: Optional[pd.DataFrame] = None,
	data_dir: Optional[Path] = None,
	weights_path: Optional[Path] = None,
	output_path: Optional[Path] = None,
) -> pd.DataFrame:
	"""Rank plausible industrial sources for every pollution event."""
	event_data = load_pollution_events() if events is None else events.copy()
	source_data = load_source_tracing_data(data_dir=data_dir)
	weights = load_weights(weights_path)
	rows = []
	for _, event in event_data.sort_values("event_id").iterrows():
		rows.extend(_rank_event_candidates(event, source_data, weights))

	ranking = pd.DataFrame(rows)
	ranking["rank"] = ranking.groupby("event_id")["score"].rank(
		method="first", ascending=False
	).astype(int)
	ranking = ranking.sort_values(["event_id", "rank"], kind="stable").reset_index(drop=True)
	destination = Path(output_path) if output_path is not None else _project_root() / "reports" / "source_ranking.csv"
	destination.parent.mkdir(parents=True, exist_ok=True)
	ranking.to_csv(destination, index=False)
	return ranking
