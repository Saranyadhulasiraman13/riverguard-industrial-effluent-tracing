"""Run the RiverGuard before-vs-after experiment and failure tests."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import run_anomaly_pipeline, run_baseline_detector
from src.anomaly_detection import load_sensor_readings
from src.evaluation import build_experiment_results, save_experiment_results
from src.source_tracing import run_source_tracing


def run_experiment() -> object:
	"""Evaluate sensor-only detection against the full prototype pipeline."""
	readings = load_sensor_readings()
	baseline, _, _ = run_baseline_detector(
		readings,
		output_path=Path("reports") / "baseline_anomalies.csv",
	)
	after = run_anomaly_pipeline(readings)
	ranking = run_source_tracing(events=after["pollution_events"])
	results = build_experiment_results(
		baseline,
		after["improved_anomalies"],
		after["pollution_events"],
		ranking,
	)
	path = save_experiment_results(results)
	processed_dir = PROJECT_ROOT / "data" / "processed"
	data_summary = {
		file.stem: len(pd.read_csv(file)) for file in sorted(processed_dir.glob("*.csv"))
	}
	cleaning = pd.read_csv(PROJECT_ROOT / "reports" / "cleaning_summary.csv")
	events = after["pollution_events"]
	plausible = ranking[
		(ranking["upstream_compatibility"] == 1.0)
		& ranking["confidence"].isin(["High", "Medium"])
	]
	edge_results = results[results["phase"] == "edge_case"]
	print("DATA SUMMARY")
	print(data_summary)
	print("CLEANING SUMMARY")
	print(
		f"datasets={len(cleaning)}; input_rows={int(cleaning['input_rows'].sum())}; "
		f"output_rows={int(cleaning['output_rows'].sum())}; "
		f"missing_after={int(cleaning['missing_after'].sum())}"
	)
	print("POLLUTION ANOMALY RESULTS")
	print(f"baseline_anomalies={int(baseline['anomaly'].sum())}; improved_anomalies={len(after['improved_anomalies'])}")
	print("POLLUTION EVENT RESULTS")
	print(f"events={len(events)}; severities={events['severity'].value_counts().to_dict()}")
	print("SOURCE-TRACING RESULTS")
	print(f"ranked_candidates={len(ranking)}; plausible_events={plausible['event_id'].nunique()}")
	print("TOP PLAUSIBLE SOURCES")
	print(ranking[(ranking["rank"] == 1) & (ranking["event_id"].isin(plausible["event_id"]))][["event_id", "source_id", "score", "confidence"]].to_string(index=False))
	print(f"experiment_report={path}")
	print(results[results["status"].isin(["measured", "proxy"])].to_string(index=False))
	print(results[results["phase"] == "edge_case"].to_string(index=False))
	print("ERROR ANALYSIS")
	print("No validated anomaly or source ground truth is present; precision, recall, F1, Top-1, Top-3, and false-attribution rate are unavailable.")
	print(f"edge_cases_passed={int((edge_results['status'] == 'passed').sum())}; untraceable_events={int(results.loc[(results['metric'] == 'untraceable_events'), 'value'].iloc[0])}")
	return results


if __name__ == "__main__":
	run_experiment()
