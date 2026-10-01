from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.source_tracing import _rank_event_candidates, run_source_tracing


EVENT_TIME = pd.Timestamp("2025-01-01T12:00:00Z")
WEIGHTS = {
	"upstream_weight": 0.30,
	"discharge_timing_weight": 0.30,
	"operation_weight": 0.20,
	"sensor_relationship_weight": 0.20,
	"high_confidence_threshold": 0.75,
	"medium_confidence_threshold": 0.45,
}


def make_event():
	return pd.Series(
		{
			"event_id": "EV-1",
			"start_time": EVENT_TIME,
			"end_time": EVENT_TIME,
			"station_id": "S3",
			"peak_value": 18.0,
			"affected_parameter": "turbidity_ntu",
			"severity": "high",
		}
	)


def make_tracing_data():
	return {
		"flow": pd.DataFrame(
			{
				"timestamp": [EVENT_TIME, EVENT_TIME],
				"from_station": ["S1", "S2"],
				"to_station": ["S2", "S3"],
				"direction": ["downstream", "downstream"],
			}
		),
		"units": pd.DataFrame(
			[
				{"unit_id": "U1", "station_id": "S1"},
				{"unit_id": "U2", "station_id": "S2"},
			]
		),
		"schedules": pd.DataFrame(
			[
				{
					"unit_id": unit_id,
					"discharge_start": EVENT_TIME - pd.Timedelta(hours=1),
					"discharge_end": EVENT_TIME + pd.Timedelta(hours=1),
				}
				for unit_id in ("U1", "U2")
			]
		),
		"operations": pd.DataFrame(
			[
				{
					"unit_id": unit_id,
					"timestamp": EVENT_TIME,
					"operation_status": "active",
				}
				for unit_id in ("U1", "U2")
			]
		),
	}


def write_tracing_inputs(directory: Path, data=None):
	data = data or make_tracing_data()
	frames = {
		"flow_direction.csv": data["flow"],
		"industrial_units.csv": data["units"].assign(
			unit_name=lambda frame: frame["unit_id"], industry_type="test"
		),
		"discharge_schedule.csv": data["schedules"].assign(discharge_status="scheduled"),
		"unit_operations.csv": data["operations"],
	}
	for filename, frame in frames.items():
		frame.to_csv(directory / filename, index=False)


def write_citizen_observation(path: Path):
	pd.DataFrame(
		[
			{
				"observation_id": "C-1",
				"observation_time": EVENT_TIME.isoformat(),
				"station_id": "S3",
				"observation_type": "water_colour",
			}
		]
	).to_csv(path, index=False)


def write_satellite_observation(path: Path):
	pd.DataFrame(
		[
			{
				"observation_id": "R-1",
				"observation_time": EVENT_TIME.isoformat(),
				"station_id": "S3",
				"measurement_name": "surface_indicator",
				"measurement_value": 0.25,
			}
		]
	).to_csv(path, index=False)


class SourceFusionTests(unittest.TestCase):
	def test_all_available_evidence_is_reported_and_score_is_transparent(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			write_tracing_inputs(root)
			citizen_path = root / "citizen.csv"
			write_citizen_observation(citizen_path)
			satellite_path = root / "satellite.csv"
			write_satellite_observation(satellite_path)
			ranking = run_source_tracing(
				events=pd.DataFrame([make_event()]),
				data_dir=root,
				output_path=root / "source_ranking.csv",
				citizen_observations_path=citizen_path,
				environmental_observations_path=satellite_path,
			)

		candidate = ranking[ranking["source_id"] == "U1"].iloc[0]
		self.assertEqual(candidate["score"], 1.0)
		self.assertEqual(candidate["score_coverage"], 1.0)
		self.assertEqual(candidate["evidence_available"], "Evidence available")
		self.assertTrue(candidate["sensor_anomaly_evidence_available"])
		self.assertTrue(candidate["citizen_observation_evidence_available"])
		self.assertTrue(candidate["satellite_environmental_evidence_available"])
		self.assertEqual(candidate["citizen_observation_match_count"], 1)
		self.assertEqual(candidate["satellite_environmental_match_count"], 1)
		self.assertEqual(
			candidate["score"],
			candidate[
				[
					"upstream_score_contribution",
					"discharge_timing_score_contribution",
					"operating_event_score_contribution",
					"sensor_timing_score_contribution",
				]
			].sum(),
		)
		self.assertIn("event context only", candidate["evidence_summary"])

	def test_missing_citizen_observations_are_explicit_and_neutral(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			write_tracing_inputs(root)
			satellite_path = root / "satellite.csv"
			write_satellite_observation(satellite_path)
			ranking = run_source_tracing(
				events=pd.DataFrame([make_event()]),
				data_dir=root,
				output_path=root / "ranking.csv",
				citizen_observations_path=root / "missing_citizen.csv",
				environmental_observations_path=satellite_path,
			)

		self.assertFalse(ranking["citizen_observation_evidence_available"].any())
		self.assertTrue(ranking["evidence_summary"].str.contains("citizen observation data").all())
		self.assertTrue(ranking["satellite_environmental_evidence_available"].all())
		self.assertTrue(ranking["score"].eq(1.0).all())

	def test_missing_satellite_observations_are_explicit_and_neutral(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			write_tracing_inputs(root)
			citizen_path = root / "citizen.csv"
			write_citizen_observation(citizen_path)
			ranking = run_source_tracing(
				events=pd.DataFrame([make_event()]),
				data_dir=root,
				output_path=root / "ranking.csv",
				citizen_observations_path=citizen_path,
				environmental_observations_path=root / "missing_satellite.csv",
			)

		self.assertFalse(ranking["satellite_environmental_evidence_available"].any())
		self.assertTrue(ranking["citizen_observation_evidence_available"].all())
		self.assertTrue(
			ranking["evidence_summary"].str.contains("satellite/environmental observation data").all()
		)
		self.assertTrue(ranking["score"].eq(1.0).all())

	def test_missing_flow_contributes_no_upstream_or_timing_score(self):
		data = make_tracing_data()
		data["flow"] = data["flow"].iloc[0:0]
		candidate = pd.DataFrame(_rank_event_candidates(make_event(), data, WEIGHTS)).iloc[0]

		self.assertEqual(candidate["upstream_compatibility"], 0.0)
		self.assertEqual(candidate["sensor_timing_relationship"], 0.0)
		self.assertFalse(candidate["upstream_evidence_available"])
		self.assertEqual(candidate["evidence_available"], "Evidence available")
		self.assertNotEqual(candidate["confidence"], "High")

	def test_missing_discharge_schedule_is_unavailable_not_a_match(self):
		data = make_tracing_data()
		data["schedules"] = data["schedules"].iloc[0:0]
		candidate = pd.DataFrame(_rank_event_candidates(make_event(), data, WEIGHTS)).iloc[0]

		self.assertEqual(candidate["discharge_timing_match"], 0.0)
		self.assertFalse(candidate["discharge_evidence_available"])
		self.assertIn("discharge schedule", candidate["evidence_components_unavailable"])

	def test_missing_operating_event_is_unavailable_and_caps_high_confidence(self):
		data = make_tracing_data()
		data["operations"] = data["operations"].iloc[0:0]
		candidate = pd.DataFrame(_rank_event_candidates(make_event(), data, WEIGHTS)).iloc[0]

		self.assertEqual(candidate["operating_event_match"], 0.0)
		self.assertFalse(candidate["operating_event_evidence_available"])
		self.assertEqual(candidate["score"], 0.8)
		self.assertEqual(candidate["confidence"], "Medium")

	def test_multiple_plausible_sources_remain_ranked_independently(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			write_tracing_inputs(root)
			ranking = run_source_tracing(
				events=pd.DataFrame([make_event()]),
				data_dir=root,
				output_path=root / "ranking.csv",
				citizen_observations_path=root / "missing_citizen.csv",
				environmental_observations_path=root / "missing_satellite.csv",
			)

		plausible = ranking[ranking["confidence"].isin(["High", "Medium"])]
		self.assertEqual(set(plausible["source_id"]), {"U1", "U2"})
		self.assertEqual(plausible["rank"].tolist(), [1, 2])
		self.assertTrue(plausible["reasons"].str.contains("plausible source for inspection").all())

	def test_no_source_support_reports_unavailable_without_positive_score(self):
		data = make_tracing_data()
		data["flow"] = data["flow"].iloc[0:0]
		data["schedules"] = data["schedules"].iloc[0:0]
		data["operations"] = data["operations"].iloc[0:0]
		candidates = pd.DataFrame(_rank_event_candidates(make_event(), data, WEIGHTS))

		self.assertTrue(candidates["evidence_available"].eq("Evidence unavailable").all())
		self.assertTrue(candidates["score"].eq(0.0).all())
		self.assertTrue(candidates["sensor_anomaly_evidence_available"].all())
		self.assertTrue(
			candidates["evidence_summary"].str.contains("Unavailable evidence:").all()
		)


if __name__ == "__main__":
	unittest.main()