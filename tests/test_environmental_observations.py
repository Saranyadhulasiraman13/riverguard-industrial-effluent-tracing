from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.environmental_observations import (
	associate_environmental_observations_with_events,
	load_optional_environmental_observations,
	validate_environmental_observations,
	write_environmental_observation_reports,
)


def observation(**overrides):
	values = {
		"observation_id": "OBS-1",
		"observation_time": "2025-01-01T06:00:00Z",
		"station_id": "S1",
		"measurement_name": "surface_reflectance",
		"measurement_value": 0.25,
		"measurement_unit": "unitless",
	}
	values.update(overrides)
	return values


class EnvironmentalObservationTests(unittest.TestCase):
	def test_valid_observation_is_parsed_and_kept(self):
		validated, status = validate_environmental_observations(
			pd.DataFrame([observation()])
		)

		self.assertTrue(validated.loc[0, "is_valid"])
		self.assertEqual(validated.loc[0, "measurement_value_numeric"], 0.25)
		self.assertEqual(status["valid_rows"], 1)

	def test_missing_timestamp_and_location_are_flagged(self):
		validated, status = validate_environmental_observations(
			pd.DataFrame(
				[
					observation(observation_time="", station_id="", observation_location=""),
				]
			)
		)

		self.assertFalse(validated.loc[0, "is_valid"])
		self.assertIn("invalid_observation_time", validated.loc[0, "validation_issues"])
		self.assertIn("missing_location", validated.loc[0, "validation_issues"])
		self.assertEqual(status["invalid_rows"], 1)

	def test_invalid_measurements_and_coordinates_are_flagged_without_row_removal(self):
		validated, status = validate_environmental_observations(
			pd.DataFrame(
				[
					observation(
						station_id="",
						latitude=95,
						longitude=181,
						measurement_value="not numeric",
					),
				]
			)
		)

		self.assertEqual(len(validated), 1)
		self.assertIn("invalid_coordinates", validated.loc[0, "validation_issues"])
		self.assertIn("invalid_measurement_value", validated.loc[0, "validation_issues"])
		self.assertEqual(status["invalid_coordinate_rows"], 1)
		self.assertEqual(status["invalid_measurement_values"], 1)

	def test_duplicate_observations_are_flagged(self):
		validated, status = validate_environmental_observations(
			pd.DataFrame([observation(), observation()])
		)

		self.assertTrue(validated["validation_issues"].str.contains("duplicate_observation").all())
		self.assertEqual(status["duplicate_observation_rows"], 2)

	def test_event_association_requires_exact_station_and_inclusive_window(self):
		observations = pd.DataFrame(
			[
				observation(),
				observation(
					observation_id="OBS-2",
					observation_time="2025-01-01T12:00:01Z",
				),
			]
		)
		events = pd.DataFrame(
			[
				{
					"event_id": "EV-1",
					"station_id": "S1",
					"start_time": "2025-01-01T06:00:00Z",
					"end_time": "2025-01-01T12:00:00Z",
				}
			]
		)

		associated = associate_environmental_observations_with_events(
			observations, events
		).set_index("observation_id")

		self.assertEqual(associated.loc["OBS-1", "event_id"], "EV-1")
		self.assertEqual(
			associated.loc["OBS-1", "evidence_label"],
			"Supporting environmental evidence",
		)
		self.assertTrue(pd.isna(associated.loc["OBS-2", "event_id"]))

	def test_missing_optional_file_writes_status_without_observation_rows(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			observations, status = load_optional_environmental_observations(
				root / "not_present.csv"
			)
			outputs = write_environmental_observation_reports(
				pd.DataFrame(
					columns=["event_id", "station_id", "start_time", "end_time"]
				),
				reports_dir=root / "reports",
				observations_path=root / "not_present.csv",
			)

			self.assertIsNone(observations)
			self.assertEqual(status["status"], "unavailable")
			self.assertTrue(outputs["status_path"].is_file())
			self.assertFalse((root / "reports" / "satellite_environmental_observations.csv").exists())


if __name__ == "__main__":
	unittest.main()