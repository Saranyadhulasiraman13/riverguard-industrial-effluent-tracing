from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.anomaly_detection import load_sensor_readings, run_anomaly_pipeline, run_baseline_detector
from src.data_cleaning import clean_dataset, run_cleaning_pipeline
from src.evaluation import build_experiment_results, save_experiment_results
from src.source_tracing import _rank_event_candidates, run_source_tracing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "upstream_weight": 0.30,
    "discharge_timing_weight": 0.30,
    "operation_weight": 0.20,
    "sensor_relationship_weight": 0.20,
    "high_confidence_threshold": 0.75,
    "medium_confidence_threshold": 0.45,
}


class PipelineTests(unittest.TestCase):
    def test_missing_sensor_value_is_imputed_and_reported(self):
        readings = pd.DataFrame(
            {
                "timestamp": [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T06:00:00Z",
                    "2025-01-01T12:00:00Z",
                    "2025-01-01T18:00:00Z",
                ],
                "station_id": ["S1"] * 4,
                "parameter": ["turbidity_ntu"] * 4,
                "value": [1.0, pd.NA, 2.0, 50.0],
            }
        )

        cleaned, summary = clean_dataset("sensor_readings", readings)
        self.assertEqual(summary["numerical_values_imputed"], 1)
        self.assertEqual(summary["missing_after"], 0)
        self.assertEqual(cleaned.loc[1, "value"], 2.0)

        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "anomaly_report.csv"
            result = run_anomaly_pipeline(cleaned, report_path=report_path)

            self.assertTrue(report_path.is_file())
            self.assertTrue(result["events_path"].is_file())
            self.assertTrue(result["baseline_path"].is_file())
            self.assertEqual(len(cleaned), 4)

    def test_no_matching_discharge_uses_only_known_candidates(self):
        event_time = pd.Timestamp("2099-01-01T00:00:00Z")
        events = pd.DataFrame(
            [
                {
                    "event_id": "TEMP_NO_DISCHARGE",
                    "start_time": event_time,
                    "end_time": event_time,
                    "station_id": "S4",
                    "peak_value": 20.0,
                    "affected_parameter": "turbidity_ntu",
                }
            ]
        )
        data_dir = PROJECT_ROOT / "data" / "processed"
        known_units = set(pd.read_csv(data_dir / "industrial_units.csv")["unit_id"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            ranking = run_source_tracing(
                events=events,
                data_dir=data_dir,
                output_path=Path(temporary_directory) / "source_ranking.csv",
            )

        self.assertFalse(ranking.empty)
        self.assertTrue(set(ranking["source_id"]).issubset(known_units))
        self.assertTrue(ranking["discharge_timing_match"].eq(0.0).all())
        self.assertTrue(ranking["reasons"].str.contains("No matching scheduled discharge").all())

    def test_missing_flow_does_not_establish_upstream_relationship(self):
        event_time = pd.Timestamp("2025-01-01T00:00:00Z")
        event = pd.Series(
            {
                "event_id": "TEMP_MISSING_FLOW",
                "start_time": event_time,
                "end_time": event_time,
                "station_id": "S2",
            }
        )
        data = {
            "flow": pd.DataFrame(
                {
                    "timestamp": [event_time],
                    "from_station": ["S1"],
                    "to_station": ["S2"],
                    "direction": [pd.NA],
                }
            ),
            "units": pd.DataFrame(
                [{"unit_id": "U1", "station_id": "S1"}]
            ),
            "schedules": pd.DataFrame(
                columns=["unit_id", "discharge_start", "discharge_end"]
            ),
            "operations": pd.DataFrame(
                columns=["unit_id", "timestamp", "operation_status"]
            ),
        }

        candidates = pd.DataFrame(_rank_event_candidates(event, data, WEIGHTS))

        self.assertEqual(candidates.loc[0, "upstream_compatibility"], 0.25)
        self.assertEqual(candidates.loc[0, "sensor_timing_relationship"], 0.0)
        self.assertNotEqual(candidates.loc[0, "confidence"], "High")

    def test_real_pipeline_regression_and_report_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed_dir = root / "processed"
            reports_dir = root / "reports"
            summary = run_cleaning_pipeline(
                data_dir=PROJECT_ROOT / "data" / "raw",
                output_dir=processed_dir,
                summary_path=reports_dir / "cleaning_summary.csv",
            )
            readings = load_sensor_readings(processed_dir / "sensor_readings.csv")
            baseline, parameter, threshold = run_baseline_detector(
                readings,
                output_path=reports_dir / "baseline_anomalies.csv",
            )
            detected = run_anomaly_pipeline(
                readings,
                report_path=reports_dir / "anomaly_report.csv",
            )
            ranking = run_source_tracing(
                events=detected["pollution_events"],
                data_dir=processed_dir,
                output_path=reports_dir / "source_ranking.csv",
            )
            results = build_experiment_results(
                baseline,
                detected["improved_anomalies"],
                detected["pollution_events"],
                ranking,
            )
            results_path = save_experiment_results(
                results,
                path=reports_dir / "experiment_results.csv",
            )
            result_metrics = results.set_index(["phase", "metric"])["value"]

            expected_reports = [
                "cleaning_summary.csv",
                "baseline_anomalies.csv",
                "anomaly_report.csv",
                "pollution_events.csv",
                "source_ranking.csv",
                "experiment_results.csv",
            ]
            self.assertTrue(all((reports_dir / name).is_file() for name in expected_reports))
            self.assertEqual(len(summary), 7)
            self.assertEqual(int(summary["input_rows"].sum()), 106)
            self.assertEqual(len(readings), 48)
            self.assertEqual((parameter, threshold), ("turbidity_ntu", 10.0))
            self.assertEqual(int(baseline["anomaly"].sum()), 11)
            self.assertEqual(len(detected["improved_anomalies"]), 7)
            self.assertEqual(len(detected["pollution_events"]), 7)
            self.assertEqual(len(ranking), 21)
            self.assertEqual(result_metrics.loc[("comparison", "anomaly_reading_count_change")], -4)
            self.assertAlmostEqual(
                result_metrics.loc[("comparison", "anomaly_rate_change")],
                -4 / 48,
            )
            self.assertAlmostEqual(
                result_metrics.loc[("comparison", "relative_anomaly_count_change")],
                -4 / 11,
            )
            self.assertIn("Sensor readings only", result_metrics.loc[("baseline", "feature_scope")])
            self.assertIn("sensor readings", result_metrics.loc[("improved", "detector_scope")])
            self.assertIn("Flow direction", result_metrics.loc[("prototype", "source_tracing_inputs")])
            self.assertFalse(result_metrics.loc[("evaluation", "ground_truth_available")])
            self.assertEqual(results_path, reports_dir / "experiment_results.csv")


if __name__ == "__main__":
    unittest.main()