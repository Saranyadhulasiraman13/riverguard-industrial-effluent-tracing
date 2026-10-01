"""Explainable ranking of plausible industrial sources for inspection."""

from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from .citizen_observations import (
	associate_observations_with_events,
	load_optional_observations,
)
from .environmental_observations import (
	associate_environmental_observations_with_events,
	load_optional_environmental_observations,
)


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
	"""Load source-tracing inputs, leaving absent contextual sources empty."""
	directory = Path(data_dir) if data_dir is not None else _project_root() / "data" / "processed"
	datasets = {
		"flow": _load_optional_csv(
			directory,
			"flow_direction.csv",
			["timestamp", "from_station", "to_station", "direction"],
		),
		"units": _load_csv(
			directory,
			"industrial_units.csv",
			["unit_id", "unit_name", "station_id", "industry_type"],
		),
		"schedules": _load_optional_csv(
			directory,
			"discharge_schedule.csv",
			["unit_id", "discharge_start", "discharge_end", "discharge_status"],
		),
		"operations": _load_optional_csv(
			directory,
			"unit_operations.csv",
			["unit_id", "timestamp", "operation_status"],
		),
	}
	datasets["flow"]["timestamp"] = pd.to_datetime(
		datasets["flow"]["timestamp"], utc=True, errors="coerce"
	)
	datasets["schedules"]["discharge_start"] = pd.to_datetime(
		datasets["schedules"]["discharge_start"], utc=True, errors="coerce"
	)
	datasets["schedules"]["discharge_end"] = pd.to_datetime(
		datasets["schedules"]["discharge_end"], utc=True, errors="coerce"
	)
	datasets["operations"]["timestamp"] = pd.to_datetime(
		datasets["operations"]["timestamp"], utc=True, errors="coerce"
	)
	return datasets


def _load_optional_csv(
	data_dir: Path, filename: str, required_columns: Iterable[str]
) -> pd.DataFrame:
	"""Load a contextual dataset, using an empty schema when it is absent."""
	path = data_dir / filename
	if not path.is_file():
		return pd.DataFrame(columns=list(required_columns))
	return _load_csv(data_dir, filename, required_columns)


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
	flow = flow.dropna(subset=["timestamp"])
	differences = (flow["timestamp"] - event_time).abs()
	if differences.empty or differences.min() > pd.Timedelta(hours=6):
		return flow.iloc[0:0], True
	snapshot_time = flow.loc[differences.idxmin(), "timestamp"]
	snapshot = flow[flow["timestamp"] == snapshot_time].copy()
	directions = snapshot["direction"].astype("string").str.strip().str.lower()
	conflicting = directions.isna().any() or not directions.eq("downstream").fillna(False).all()
	return snapshot, conflicting


def _flow_snapshot(flow: pd.DataFrame, event_time: pd.Timestamp) -> Tuple[pd.DataFrame, bool]:
	"""Select the nearest flow snapshot within six hours and flag missing/conflicting flow."""
	flow = flow.dropna(subset=["timestamp"])
	differences = (flow["timestamp"] - event_time).abs()
	if differences.empty or differences.min() > pd.Timedelta(hours=6):
		return flow.iloc[0:0], True
	snapshot_time = flow.loc[differences.idxmin(), "timestamp"]
	snapshot = flow[flow["timestamp"] == snapshot_time].copy()
	directions = snapshot["direction"].astype("string").str.strip().str.lower()
	conflicting = directions.isna().any() or not directions.eq("downstream").fillna(False).all()
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
) -> Tuple[float, str, Optional[Tuple[pd.Timestamp, pd.Timestamp]], bool]:
	"""Score an overlapping or recently completed scheduled discharge."""
	unit_schedules = schedules[schedules["unit_id"] == unit_id]
	unit_schedules = unit_schedules.dropna(subset=["discharge_start", "discharge_end"])
	if unit_schedules.empty:
		return 0.0, "Discharge schedule evidence is unavailable for this unit.", None, False
	for row in unit_schedules.itertuples(index=False):
		if row.discharge_start <= end and row.discharge_end >= start:
			return 1.0, "Discharge schedule overlaps the pollution event.", (
				row.discharge_start,
				row.discharge_end,
			), True
		if row.discharge_end <= start and start - row.discharge_end <= pd.Timedelta(hours=6):
			return 0.8, "Scheduled discharge ended shortly before the pollution event.", (
				row.discharge_start,
				row.discharge_end,
			), True
	return 0.0, "No matching scheduled discharge was found in the available schedule.", None, True


def _operation_match(
	operations: pd.DataFrame,
	unit_id: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	 discharge_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Tuple[float, str, bool]:
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
	].dropna(subset=["timestamp", "operation_status"])
	relevant = relevant[
		relevant["operation_status"].astype("string").str.strip().ne("")
	].sort_values("timestamp")
	if relevant.empty:
		return 0.0, "Operating-event evidence is unavailable for the relevant period.", False
	latest = relevant.iloc[-1]
	status = str(latest["operation_status"]).lower()
	if status == "active":
		return 1.0, "Unit operation record shows active status during the relevant period.", True
	if status == "maintenance":
		return 0.2, "Unit was in maintenance during the relevant period.", True
	return 0.0, f"Latest unit operation status was '{status}', not active.", True


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
	flow_available = not snapshot.empty and not flow_uncertain
	if snapshot.empty:
		flow_status = "unavailable"
	elif flow_uncertain:
		flow_status = "conflicting or incomplete"
	else:
		flow_status = "available"
	flow_reason = (
		"River-flow evidence is available within six hours of the event."
		if flow_available
		else f"River-flow evidence is {flow_status} within six hours of the event."
	)
	sensor_peak = event.get("peak_value")
	sensor_parameter = event.get("affected_parameter")
	sensor_anomaly_available = (
		pd.notna(event.get("event_id"))
		and pd.notna(sensor_peak)
		and pd.notna(sensor_parameter)
	)
	citizen_count = _event_observation_count(event.get("citizen_observation_match_count", 0))
	satellite_count = _event_observation_count(
		event.get("satellite_environmental_match_count", 0)
	)
	citizen_data_status = _status_text(
		event.get("citizen_observation_data_status", "unavailable")
	)
	satellite_data_status = _status_text(
		event.get("satellite_environmental_data_status", "unavailable")
	)
	candidates = []
	for unit in data["units"].itertuples(index=False):
		path = _station_path(snapshot, unit.station_id, event["station_id"])
		if flow_uncertain:
			upstream_score = 0.0
			upstream_reason = f"Upstream relationship unavailable because river-flow evidence is {flow_status}."
		elif path is not None and len(path) > 1:
			upstream_score = 1.0
			upstream_reason = "Unit station is upstream of the affected station under the observed flow."
		elif path == [unit.station_id]:
			upstream_score = 0.5
			upstream_reason = "Unit shares the affected monitoring station; no upstream path was established."
		else:
			upstream_score = 0.0
			upstream_reason = "Observed flow does not support transport from the unit station to the affected station."

		discharge_score, discharge_reason, discharge_window, discharge_available = _discharge_match(
			data["schedules"], unit.unit_id, start, end
		)
		operation_score, operation_reason, operation_available = _operation_match(
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

		flow_component_available = flow_available
		upstream_contribution = round(weights["upstream_weight"] * upstream_score, 6)
		discharge_contribution = round(
			weights["discharge_timing_weight"] * discharge_score, 6
		)
		operation_contribution = round(
			weights["operation_weight"] * operation_score, 6
		)
		sensor_contribution = round(
			weights["sensor_relationship_weight"] * sensor_score, 6
		)
		score = round(
			upstream_contribution
			+ discharge_contribution
			+ operation_contribution
			+ sensor_contribution,
			6,
		)
		available_weight = (
			(weights["upstream_weight"] + weights["sensor_relationship_weight"])
			* float(flow_component_available)
			+ weights["discharge_timing_weight"] * float(discharge_available)
			+ weights["operation_weight"] * float(operation_available)
		)
		configured_weight = sum(
			weights[key]
			for key in (
				"upstream_weight",
				"discharge_timing_weight",
				"operation_weight",
				"sensor_relationship_weight",
			)
		)
		score_coverage = available_weight / configured_weight if configured_weight else 0.0
		confidence = _confidence(score, weights, flow_uncertain or upstream_score < 1.0)
		if confidence == "High" and score_coverage < 1.0:
			confidence = "Medium"
		reasons = [upstream_reason, discharge_reason, operation_reason, sensor_reason]
		if confidence in {"High", "Medium"} and upstream_score == 1.0 and not flow_uncertain:
			conclusion = f"{unit.unit_id} is a {confidence.lower()}-priority plausible source for inspection."
		else:
			conclusion = f"{unit.unit_id} remains a lower-confidence candidate for manual inspection."
		available_components = []
		unavailable_components = []
		if sensor_anomaly_available:
			available_components.append("sensor anomaly event (event context, not source attribution)")
		else:
			unavailable_components.append("sensor anomaly event details")
		if flow_available:
			available_components.append("river flow/upstream relationship")
		else:
			unavailable_components.append(f"river flow/upstream relationship ({flow_status})")
		if discharge_available:
			available_components.append("discharge schedule")
		else:
			unavailable_components.append("discharge schedule")
		if operation_available:
			available_components.append("operating event")
		else:
			unavailable_components.append("operating event")
		if citizen_count:
			available_components.append(
				f"{citizen_count} citizen observation(s) matched to event (event context only)"
			)
		elif citizen_data_status == "available":
			unavailable_components.append("citizen observation with a valid event match")
		else:
			unavailable_components.append("citizen observation data")
		if satellite_count:
			available_components.append(
				f"{satellite_count} satellite/environmental observation(s) matched to event (event context only)"
			)
		elif satellite_data_status == "available":
			unavailable_components.append(
				"satellite/environmental observation with a valid event match"
			)
		else:
			unavailable_components.append("satellite/environmental observation data")
		evidence_available = any(
			(flow_available, discharge_available, operation_available)
		)
		evidence_summary = (
			"Available evidence: "
			+ ("; ".join(available_components) if available_components else "none")
			+ ". Unavailable evidence: "
			+ ("; ".join(unavailable_components) if unavailable_components else "none")
			+ "."
		)
		candidates.append(
			{
				"event_id": event["event_id"],
				"source_id": unit.unit_id,
				"affected_station": event["station_id"],
				"sensor_anomaly_evidence_available": bool(sensor_anomaly_available),
				"sensor_anomaly_value": sensor_peak,
				"sensor_anomaly_parameter": sensor_parameter,
				"sensor_anomaly_severity": event.get("severity", pd.NA),
				"upstream_compatibility": upstream_score,
				"upstream_evidence_available": bool(flow_available),
				"discharge_timing_match": discharge_score,
				"discharge_evidence_available": bool(discharge_available),
				"operating_event_match": operation_score,
				"operating_event_evidence_available": bool(operation_available),
				"sensor_timing_relationship": sensor_score,
				"sensor_timing_evidence_available": bool(flow_available),
				"citizen_observation_evidence_available": bool(citizen_count),
				"citizen_observation_match_count": citizen_count,
				"citizen_observation_data_status": citizen_data_status,
				"satellite_environmental_evidence_available": bool(satellite_count),
				"satellite_environmental_match_count": satellite_count,
				"satellite_environmental_data_status": satellite_data_status,
				"evidence_available": (
					"Evidence available" if evidence_available else "Evidence unavailable"
				),
				"evidence_components_available": "; ".join(available_components),
				"evidence_components_unavailable": "; ".join(unavailable_components),
				"evidence_summary": evidence_summary,
				"score_coverage": round(score_coverage, 6),
				"upstream_score_contribution": upstream_contribution,
				"discharge_timing_score_contribution": discharge_contribution,
				"operating_event_score_contribution": operation_contribution,
				"sensor_timing_score_contribution": sensor_contribution,
				"score": score,
				"confidence": confidence,
				"reasons": conclusion + " " + " ".join(reasons) + " " + evidence_summary,
			}
		)
	return candidates


def _event_observation_count(value: object) -> int:
	try:
		return max(0, int(value)) if pd.notna(value) else 0
	except (TypeError, ValueError):
		return 0


def _status_text(value: object) -> str:
	return str(value) if pd.notna(value) else "unavailable"


def _matched_event_counts(
	events: pd.DataFrame,
	observations: Optional[pd.DataFrame],
	associate: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
	support_column: str,
	support_value: str,
	starts_with: bool = False,
) -> Dict[str, int]:
	"""Count only valid optional observations associated with each event."""
	counts = {str(event_id): 0 for event_id in events["event_id"]}
	if observations is None:
		return counts
	associated = associate(observations, events)
	if starts_with:
		matched = associated[support_column].astype("string").str.startswith(
			support_value, na=False
		)
	else:
		matched = associated[support_column].astype("string").eq(support_value).fillna(False)
	matched_rows = associated.loc[matched]
	if not matched_rows.empty:
		observed_counts = matched_rows["event_id"].astype(str).value_counts()
		for event_id, count in observed_counts.items():
			counts[event_id] = int(count)
	return counts


def _optional_event_observation_evidence(
	events: pd.DataFrame,
	citizen_observations_path: Optional[Path] = None,
	environmental_observations_path: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
	"""Load optional observations as event context, never as source-specific scores."""
	citizen_observations, citizen_status = load_optional_observations(
		citizen_observations_path
	)
	environmental_observations, environmental_status = (
		load_optional_environmental_observations(environmental_observations_path)
	)
	citizen_counts = _matched_event_counts(
		events,
		citizen_observations,
		associate_observations_with_events,
		"association_status",
		"supporting observation",
		starts_with=True,
	)
	environmental_counts = _matched_event_counts(
		events,
		environmental_observations,
		associate_environmental_observations_with_events,
		"evidence_label",
		"Supporting environmental evidence",
	)
	return {
		str(event["event_id"]): {
			"citizen_observation_match_count": citizen_counts[str(event["event_id"])],
			"citizen_observation_data_status": citizen_status.get("status", "unavailable"),
			"satellite_environmental_match_count": environmental_counts[
				str(event["event_id"])
			],
			"satellite_environmental_data_status": environmental_status.get(
				"status", "unavailable"
			),
		}
		for _, event in events.iterrows()
	}


def run_source_tracing(
	events: Optional[pd.DataFrame] = None,
	data_dir: Optional[Path] = None,
	weights_path: Optional[Path] = None,
	output_path: Optional[Path] = None,
	citizen_observations_path: Optional[Path] = None,
	environmental_observations_path: Optional[Path] = None,
) -> pd.DataFrame:
	"""Rank plausible industrial sources for every pollution event."""
	event_data = load_pollution_events() if events is None else events.copy()
	source_data = load_source_tracing_data(data_dir=data_dir)
	weights = load_weights(weights_path)
	event_observation_evidence = _optional_event_observation_evidence(
		event_data,
		citizen_observations_path=citizen_observations_path,
		environmental_observations_path=environmental_observations_path,
	)
	rows = []
	for _, event in event_data.sort_values("event_id").iterrows():
		event = event.copy()
		for key, value in event_observation_evidence[str(event["event_id"])].items():
			event[key] = value
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
