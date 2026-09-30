from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.citizen_observations import (
	associate_observations_with_events,
	load_optional_observations,
	validate_observations,
	write_observation_reports,
)


class CitizenObservationTests(unittest.TestCase):
	def test_missing_optional_dataset_is_reported_without_rows(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			observations, status = load_optional_observations(
				Path(temporary_directory) / "not_present.csv"
			)

		self.assertIsNone(observations)
		self.assertEqual(status["status"], "unavailable")
		self.assertFalse(status["observations_available"])
		self.assertEqual(status["input_rows"], 0)

	def test_empty_schema_validates_without_fabricating_observations(self):
		frame = pd.DataFrame(
			columns=[
				"observation_id",
				"observation_time",
				"station_id",
				"observation_type",
			]
		)

		validated, status = validate_observations(frame)

		self.assertTrue(validated.empty)
		self.assertEqual(status["input_rows"], 0)
		self.assertEqual(status["valid_rows"], 0)
		self.assertEqual(status["invalid_rows"], 0)
		self.assertIn("observation_time_utc", validated.columns)

	def test_required_schema_is_enforced(self):
		frame = pd.DataFrame(columns=["observation_time", "station_id"])

		with self.assertRaisesRegex(ValueError, "missing required columns"):
			validate_observations(frame)

	def test_empty_observations_produce_status_not_an_observation_report(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			events = pd.DataFrame(
				columns=["event_id", "station_id", "start_time", "end_time"]
			)
			outputs = write_observation_reports(
				events,
				reports_dir=root / "reports",
				observations_path=root / "missing_observations.csv",
			)

			status_report = pd.read_csv(outputs["status_path"])
			self.assertEqual(status_report.loc[0, "status"], "unavailable")
			self.assertFalse((root / "reports" / "citizen_observations.csv").exists())

	def test_empty_validated_observations_have_no_event_matches(self):
		observations = pd.DataFrame(
			columns=[
				"observation_id",
				"observation_time",
				"station_id",
				"observation_type",
			]
		)
		events = pd.DataFrame(
			columns=["event_id", "station_id", "start_time", "end_time"]
		)

		associated = associate_observations_with_events(observations, events)

		self.assertTrue(associated.empty)
		self.assertIn("association_status", associated.columns)


if __name__ == "__main__":
	unittest.main()