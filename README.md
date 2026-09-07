# RiverGuard

RiverGuard is a small river-restoration prototype that detects pollution spikes and ranks plausible industrial sources for inspection. The included CSV data is synthetic and simulated for prototype testing; it does not represent real environmental measurements or real facilities.

## Problem

Pollution spikes need rapid, explainable investigation. Sensor anomalies alone do not establish which industrial unit should be inspected, especially when flow direction, discharge timing, and operating status must be considered together.

## Solution

RiverGuard combines cleaned sensor readings, historical anomaly detection, river flow direction, industrial discharge schedules, and unit operating events. It produces evidence-based rankings for inspection and never treats a ranked unit as proven responsible.

## Architecture

- `data/raw/`: synthetic input CSV files.
- `data/processed/`: cleaned CSV copies.
- `src/data_loader.py`: reusable CSV loading functions.
- `src/data_cleaning.py`: timestamp, duplicate, category, numeric, and range cleaning.
- `src/anomaly_detection.py`: baseline threshold detection, rolling-IQR detection, and event grouping.
- `src/source_tracing.py`: weighted flow, discharge, operation, and timing evidence ranking.
- `src/evaluation.py`: proxy metrics and failure-case checks.
- `experiments/run_experiment.py`: complete backend experiment runner.
- `reports/`: generated CSV outputs.

## Baseline

The baseline selects the available pollution-related numeric parameter. With the current data it uses `turbidity_ntu` and a transparent fixed threshold of 10.0 NTU. It uses sensor readings only.

## Improved approach

The improved detector uses a shifted rolling IQR per station and parameter. The current reading is excluded from its own baseline, and nearby anomalies are grouped into pollution events.

## Source ranking

Each industrial unit receives separate evidence scores for upstream compatibility, discharge timing, operating status, and sensor/timing relationship. Weights are configured in `config/weights.yaml`.

## Explainability

Reports include evidence components, scores, confidence, ranks, and human-readable reasons. Missing discharge evidence reduces a score but does not automatically eliminate a unit. Uncertain flow reduces confidence and flags manual inspection.

RiverGuard provides decision support for inspection. A high-ranked source is a plausible source, not proof of legal or causal responsibility.

## Edge cases

The experiment tests missing sensor data, anomalies without matching scheduled discharge, and missing or conflicting flow direction. The system returns explicit messages and does not fabricate anomalies or crash.

## Experiment

The before-vs-after experiment compares the sensor-only baseline with the historical detector and the full source-ranking workflow. It also writes edge-case and evaluation results.

## Metrics

The current synthetic datasets contain no validated anomaly or source ground-truth labels. Therefore precision, recall, F1, Top-1 accuracy, Top-3 coverage, and false-attribution rate are reported as unavailable. Proxy metrics such as anomaly counts, event counts, ranking success rate, and untraceable event rate are clearly labelled.

## Limitations

- Input data is synthetic and simulated.
- No legal or causal attribution is performed.
- No validated ground truth is available for supervised evaluation.
- Flow and industrial relationships are represented by the prototype CSV schemas.
- This is a backend prototype, not a complete production application.

## Installation

Use Python 3.11 or newer, then install the permitted dependencies:

```powershell
python -m pip install pandas numpy scikit-learn plotly streamlit folium pyyaml
```

## Running the system

From the project root, run:

```powershell
python experiments/run_experiment.py
```

The command runs loading, cleaning, anomaly detection, event grouping, source ranking, evaluation, and edge-case checks. Generated reports include:

- `reports/anomaly_report.csv`
- `reports/source_ranking.csv`
- `reports/experiment_results.csv`
