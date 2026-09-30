from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from src.citizen_observations import (
    associate_observations_with_events,
    load_optional_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def read_csv(path: Path) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    if not path.is_file():
        return None, f"{path.name}: Report not available yet."
    try:
        return pd.read_csv(path), None
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        return None, f"{path.name}: Report not available yet. ({error})"


def read_project_data(filename: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    for directory in ("processed", "raw"):
        path = PROJECT_ROOT / "data" / directory / filename
        if path.is_file():
            return read_csv(path)
    return None, f"{filename}: Data file not available yet."


def has_columns(
    frame: Optional[pd.DataFrame], required: Sequence[str], label: str
) -> bool:
    if frame is None:
        return False
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        st.info(f"{label}: Report not available yet. Missing fields: {', '.join(missing)}.")
        return False
    return True


def unavailable(message: Optional[str]) -> None:
    st.info(message or "Report not available yet.")


def count_value(frame: Optional[pd.DataFrame]) -> str:
    return f"{len(frame):,}" if frame is not None else "--"


def total_column(frame: Optional[pd.DataFrame], column: str) -> Optional[int]:
    if frame is None or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int(values.sum())


def display_number(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "--"


def prepare_timestamps(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in columns:
        if column in prepared:
            prepared[column] = pd.to_datetime(
                prepared[column], utc=True, errors="coerce"
            )
    return prepared


def draw_station_map(
    stations: pd.DataFrame,
    flow: Optional[pd.DataFrame],
    selected_stations: set[str],
) -> None:
    required_coordinates = {"station_id", "latitude", "longitude"}
    if not required_coordinates.issubset(stations.columns):
        st.info(
            "Geographic mapping is unavailable because valid station coordinates "
            "are not present in the available station metadata."
        )
        return

    plotted_stations = stations.copy()
    plotted_stations["latitude"] = pd.to_numeric(
        plotted_stations["latitude"], errors="coerce"
    )
    plotted_stations["longitude"] = pd.to_numeric(
        plotted_stations["longitude"], errors="coerce"
    )
    plotted_stations = plotted_stations.dropna(subset=["latitude", "longitude"])
    plotted_stations = plotted_stations[
        plotted_stations["latitude"].between(-90, 90)
        & plotted_stations["longitude"].between(-180, 180)
    ]
    if plotted_stations.empty:
        st.info(
            "Geographic mapping is unavailable because valid station coordinates "
            "are not present in the available station metadata."
        )
        return

    station_lookup = plotted_stations.set_index("station_id").to_dict("index")
    links = pd.DataFrame(columns=["from_station", "to_station", "direction"])
    if flow is not None and {"from_station", "to_station", "direction"}.issubset(flow.columns):
        links = flow.copy()
        links["direction"] = links["direction"].astype(str).str.lower()
        links = links[links["direction"] == "downstream"]
        links = links.drop_duplicates(["from_station", "to_station", "direction"])
        links = links[
            links["from_station"].isin(station_lookup)
            & links["to_station"].isin(station_lookup)
        ]
        if selected_stations:
            links = links[
                links["from_station"].isin(selected_stations)
                | links["to_station"].isin(selected_stations)
            ]

    visible_ids = set(selected_stations)
    if not visible_ids:
        visible_ids = set(station_lookup)
    if not links.empty:
        visible_ids.update(links["from_station"])
        visible_ids.update(links["to_station"])
    visible_stations = plotted_stations[
        plotted_stations["station_id"].isin(visible_ids)
    ]
    if visible_stations.empty:
        visible_stations = plotted_stations

    river_map = folium.Map(
        location=[
            float(visible_stations["latitude"].mean()),
            float(visible_stations["longitude"].mean()),
        ],
        zoom_start=12,
        control_scale=True,
    )
    for station in visible_stations.itertuples(index=False):
        station_name = getattr(station, "station_name", station.station_id)
        folium.Marker(
            location=[station.latitude, station.longitude],
            tooltip=f"{station.station_id} - {station_name}",
            popup=f"{station.station_id}: {station_name}",
        ).add_to(river_map)

    for link in links.itertuples(index=False):
        source = station_lookup[link.from_station]
        destination = station_lookup[link.to_station]
        folium.PolyLine(
            locations=[
                [source["latitude"], source["longitude"]],
                [destination["latitude"], destination["longitude"]],
            ],
            color="#167d8d",
            weight=3,
            tooltip=f"Observed downstream link: {link.from_station} to {link.to_station}",
        ).add_to(river_map)

    st_folium(river_map, height=430, key="riverguard_station_map")


st.set_page_config(page_title="RiverGuard", page_icon="RG", layout="wide")
st.title("RiverGuard")
st.caption(
    "Synthetic river-monitoring prototype for anomaly review and plausible-source "
    "inspection. Ranked sources are not confirmed causes."
)

events, events_error = read_csv(REPORTS_DIR / "pollution_events.csv")
anomalies, anomalies_error = read_csv(REPORTS_DIR / "anomaly_report.csv")
ranking, ranking_error = read_csv(REPORTS_DIR / "source_ranking.csv")
cleaning, cleaning_error = read_csv(REPORTS_DIR / "cleaning_summary.csv")
evaluation, evaluation_error = read_csv(REPORTS_DIR / "experiment_results.csv")
stations, stations_error = read_project_data("sensors.csv")
readings, readings_error = read_project_data("sensor_readings.csv")
flow, flow_error = read_project_data("flow_direction.csv")
units, units_error = read_project_data("industrial_units.csv")
citizen_observations, citizen_observation_status = load_optional_observations()

if events is not None:
    events = prepare_timestamps(events, ["start_time", "end_time"])
if anomalies is not None:
    anomalies = prepare_timestamps(anomalies, ["timestamp"])

station_ids: set[str] = set()
for frame, column in (
    (stations, "station_id"),
    (events, "station_id"),
    (ranking, "affected_station"),
):
    if frame is not None and column in frame:
        station_ids.update(frame[column].dropna().astype(str))

severity_values = (
    sorted(events["severity"].dropna().astype(str).unique())
    if events is not None and "severity" in events
    else []
)
source_ids = (
    sorted(ranking["source_id"].dropna().astype(str).unique())
    if ranking is not None and "source_id" in ranking
    else []
)

st.sidebar.header("Filters")
selected_stations = set(
    st.sidebar.multiselect("Station", sorted(station_ids), default=sorted(station_ids))
)
selected_severities = set(
    st.sidebar.multiselect("Severity", severity_values, default=severity_values)
)
unit_names: dict[str, str] = {}
if units is not None and {"unit_id", "unit_name"}.issubset(units.columns):
    unit_names = dict(zip(units["unit_id"].astype(str), units["unit_name"].astype(str)))
selected_sources = set(
    st.sidebar.multiselect(
        "Source / industrial unit",
        source_ids,
        default=source_ids,
        format_func=lambda source_id: (
            f"{source_id} - {unit_names[source_id]}"
            if source_id in unit_names
            else source_id
        ),
    )
)

date_bounds: Optional[tuple[date, date]] = None
if events is not None and "start_time" in events:
    event_dates = events["start_time"].dropna().dt.date
    if not event_dates.empty:
        earliest, latest = min(event_dates), max(event_dates)
        selected_date_range = st.sidebar.date_input(
            "Event date range",
            value=(earliest, latest),
            min_value=earliest,
            max_value=latest,
        )
        if isinstance(selected_date_range, tuple):
            if len(selected_date_range) == 2:
                date_bounds = (selected_date_range[0], selected_date_range[1])
            elif len(selected_date_range) == 1:
                date_bounds = (selected_date_range[0], selected_date_range[0])
        else:
            date_bounds = (selected_date_range, selected_date_range)
else:
    st.sidebar.caption("Date filtering is unavailable until event timestamps are present.")

filtered_events = events.copy() if events is not None else None
if filtered_events is not None:
    if "station_id" in filtered_events:
        filtered_events = filtered_events[
            filtered_events["station_id"].astype(str).isin(selected_stations)
        ]
    if "severity" in filtered_events:
        filtered_events = filtered_events[
            filtered_events["severity"].astype(str).isin(selected_severities)
        ]
    if date_bounds is not None and "start_time" in filtered_events:
        event_dates = filtered_events["start_time"].dt.date
        filtered_events = filtered_events[
            event_dates.between(date_bounds[0], date_bounds[1])
        ]
    if selected_sources != set(source_ids) and ranking is not None:
        if {"event_id", "source_id"}.issubset(ranking.columns):
            source_event_ids = set(
                ranking.loc[
                    ranking["source_id"].astype(str).isin(selected_sources), "event_id"
                ].astype(str)
            )
            filtered_events = filtered_events[
                filtered_events["event_id"].astype(str).isin(source_event_ids)
            ]

filtered_event_ids = (
    set(filtered_events["event_id"].astype(str))
    if filtered_events is not None and "event_id" in filtered_events
    else set()
)
filtered_ranking = ranking.copy() if ranking is not None else None
if filtered_ranking is not None:
    if "event_id" in filtered_ranking:
        filtered_ranking = filtered_ranking[
            filtered_ranking["event_id"].astype(str).isin(filtered_event_ids)
        ]
    if "source_id" in filtered_ranking:
        filtered_ranking = filtered_ranking[
            filtered_ranking["source_id"].astype(str).isin(selected_sources)
        ]

filtered_anomalies = anomalies.copy() if anomalies is not None else None
if filtered_anomalies is not None:
    if "station_id" in filtered_anomalies:
        filtered_anomalies = filtered_anomalies[
            filtered_anomalies["station_id"].astype(str).isin(selected_stations)
        ]
    if "event_id" in filtered_anomalies:
        filtered_anomalies = filtered_anomalies[
            filtered_anomalies["event_id"].astype(str).isin(filtered_event_ids)
        ]
    if date_bounds is not None and "timestamp" in filtered_anomalies:
        anomaly_dates = filtered_anomalies["timestamp"].dt.date
        filtered_anomalies = filtered_anomalies[
            anomaly_dates.between(date_bounds[0], date_bounds[1])
        ]

st.header("Project / Data Summary")
if has_columns(stations, ["station_id"], "Station metadata"):
    station_count: Optional[int] = int(stations["station_id"].nunique())
else:
    station_count = None
if has_columns(readings, ["timestamp", "station_id", "value"], "Sensor readings"):
    reading_count: Optional[int] = len(readings)
else:
    reading_count = None
if has_columns(anomalies, ["event_id", "timestamp"], "Anomaly report"):
    anomaly_count: Optional[int] = len(anomalies)
else:
    anomaly_count = None
if has_columns(events, ["event_id", "station_id", "start_time"], "Pollution events"):
    event_count: Optional[int] = len(events)
else:
    event_count = None

plausible_count: Optional[int] = None
if ranking is not None and {"upstream_compatibility", "confidence"}.issubset(ranking.columns):
    plausible_mask = (ranking["upstream_compatibility"] == 1.0) & ranking[
        "confidence"
    ].isin(["High", "Medium"])
    plausible_count = int(plausible_mask.sum())
elif ranking is None:
    unavailable(ranking_error)

summary_columns = st.columns(5)
summary_values = [
    ("Monitoring stations", station_count),
    ("Sensor readings", reading_count),
    ("Detected anomalies", anomaly_count),
    ("Pollution events", event_count),
    ("Plausible source records", plausible_count),
]
for column, (label, value) in zip(summary_columns, summary_values):
    column.metric(label, display_number(value))
st.caption(
    "Plausible source records are flow-supported medium/high-confidence candidates "
    "for inspection, not confirmed causes."
)

st.header("Pollution Events")
if has_columns(
    filtered_events,
    ["event_id", "station_id", "start_time", "end_time"],
    "Pollution event report",
):
    event_columns = [
        "event_id",
        "station_id",
        "start_time",
        "end_time",
        "affected_parameter",
        "peak_value",
        "duration",
        "severity",
    ]
    available_event_columns = [
        column for column in event_columns if column in filtered_events.columns
    ]
    event_table = filtered_events[available_event_columns].rename(
        columns={"affected_parameter": "parameter"}
    )
    st.dataframe(event_table, width="stretch", hide_index=True)
else:
    unavailable(events_error)

st.header("Anomaly Trend")
if has_columns(filtered_anomalies, ["timestamp", "event_id"], "Anomaly report"):
    if filtered_anomalies.empty:
        st.info("No anomalies match the selected filters.")
    else:
        trend = (
            filtered_anomalies.assign(day=filtered_anomalies["timestamp"].dt.floor("D"))
            .groupby("day", as_index=False)
            .size()
            .rename(columns={"size": "anomaly_count"})
        )
        chart = px.line(
            trend,
            x="day",
            y="anomaly_count",
            markers=True,
            labels={"day": "Date", "anomaly_count": "Detected anomaly readings"},
            title="Detected anomaly readings over time",
        )
        chart.update_layout(margin={"l": 10, "r": 10, "t": 50, "b": 10})
        st.plotly_chart(chart, width="stretch")
else:
    unavailable(anomalies_error)

st.header("Stations / River Flow")
station_tab, flow_tab = st.tabs(["Stations and map", "Observed flow links"])
with station_tab:
    if stations is None:
        unavailable(stations_error)
    else:
        visible_station_table = stations.copy()
        if "station_id" in visible_station_table:
            visible_station_table = visible_station_table[
                visible_station_table["station_id"].astype(str).isin(selected_stations)
            ]
        st.dataframe(visible_station_table, width="stretch", hide_index=True)
        draw_station_map(stations, flow, selected_stations)

with flow_tab:
    if has_columns(flow, ["from_station", "to_station", "direction"], "Flow data"):
        visible_flow = flow.copy()
        visible_flow["direction"] = visible_flow["direction"].astype(str).str.lower()
        if selected_stations:
            visible_flow = visible_flow[
                visible_flow["from_station"].astype(str).isin(selected_stations)
                | visible_flow["to_station"].astype(str).isin(selected_stations)
            ]
        flow_columns = ["from_station", "to_station", "direction"]
        if "timestamp" in visible_flow:
            flow_columns.append("timestamp")
        st.dataframe(
            visible_flow[flow_columns].drop_duplicates().sort_values(flow_columns),
            width="stretch",
            hide_index=True,
        )
    else:
        unavailable(flow_error)

st.header("Plausible Sources for Inspection")
st.caption(
    "Rankings are inspection candidates based on the recorded evidence. They do not "
    "establish that a source caused an event."
)
if has_columns(
    filtered_ranking,
    ["event_id", "source_id", "affected_station", "score", "rank", "reasons"],
    "Source ranking report",
):
    source_table = filtered_ranking.copy()
    if units is not None and {"unit_id", "unit_name"}.issubset(units.columns):
        source_table = source_table.merge(
            units[["unit_id", "unit_name"]].rename(columns={"unit_id": "source_id"}),
            on="source_id",
            how="left",
        )
    source_columns = [
        "event_id",
        "source_id",
        "unit_name",
        "affected_station",
        "score",
        "confidence",
        "rank",
        "upstream_compatibility",
        "discharge_timing_match",
        "operating_event_match",
        "sensor_timing_relationship",
        "reasons",
    ]
    source_columns = [column for column in source_columns if column in source_table]
    st.dataframe(
        source_table[source_columns].sort_values(
            [column for column in ("event_id", "rank") if column in source_table]
        ),
        width="stretch",
        hide_index=True,
    )
else:
    unavailable(ranking_error)

st.subheader("Evidence for a selected event and source")
if filtered_events is not None and not filtered_events.empty and filtered_ranking is not None:
    selected_event_id = st.selectbox(
        "Pollution event",
        filtered_events["event_id"].astype(str).tolist(),
        key="evidence_event",
    )
    event_candidates = filtered_ranking[
        filtered_ranking["event_id"].astype(str) == selected_event_id
    ]
    if event_candidates.empty:
        st.info("No source-ranking records are available for this event.")
    else:
        selected_source_id = st.selectbox(
            "Ranked source candidate",
            event_candidates["source_id"].astype(str).tolist(),
            format_func=lambda source_id: (
                f"{source_id} - {unit_names[source_id]}"
                if source_id in unit_names
                else source_id
            ),
            key="evidence_source",
        )
        selected_evidence = event_candidates[
            event_candidates["source_id"].astype(str) == selected_source_id
        ].iloc[0]
        evidence_fields = [
            ("Upstream compatibility", "upstream_compatibility"),
            ("Discharge timing match", "discharge_timing_match"),
            ("Operating event match", "operating_event_match"),
            ("Sensor timing relationship", "sensor_timing_relationship"),
        ]
        evidence_columns = st.columns(len(evidence_fields))
        for column, (label, field) in zip(evidence_columns, evidence_fields):
            value = pd.to_numeric(pd.Series([selected_evidence.get(field)]), errors="coerce").iloc[0]
            column.metric(label, f"{value:.2f}" if pd.notna(value) else "--")
        st.write(str(selected_evidence["reasons"]))
else:
    st.info("Select filters that include an event with source-ranking records.")

st.header("Citizen Observations")
if citizen_observations is None:
    st.info("Citizen-science observations are not currently available.")
    if citizen_observation_status.get("status") == "invalid":
        st.caption(str(citizen_observation_status.get("message", "")))
else:
    st.caption(
        "Observations associated by station and event time are supporting observations, "
        "not proof of a pollution source."
    )
    if citizen_observations.empty:
        st.info("The optional observation file is present but contains no observation rows.")
    else:
        citizen_view = citizen_observations.copy()
        citizen_filters = st.columns(3)
        if "station_id" in citizen_view:
            citizen_station_ids = sorted(
                citizen_view["station_id"].dropna().astype(str).unique()
            )
            selected_citizen_stations = citizen_filters[0].multiselect(
                "Observation station",
                citizen_station_ids,
                default=citizen_station_ids,
                key="citizen_station_filter",
            )
            if citizen_station_ids:
                citizen_view = citizen_view[
                    citizen_view["station_id"].astype(str).isin(selected_citizen_stations)
                ]
        if "observation_type" in citizen_view:
            observation_types = sorted(
                citizen_view["observation_type"].dropna().astype(str).unique()
            )
            selected_types = citizen_filters[1].multiselect(
                "Observation type",
                observation_types,
                default=observation_types,
                key="citizen_type_filter",
            )
            if observation_types:
                citizen_view = citizen_view[
                    citizen_view["observation_type"].astype(str).isin(selected_types)
                ]
        if "observation_time_utc" in citizen_view:
            citizen_dates = citizen_view["observation_time_utc"].dropna().dt.date
            if not citizen_dates.empty:
                earliest_citizen_date = min(citizen_dates)
                latest_citizen_date = max(citizen_dates)
                citizen_date_selection = citizen_filters[2].date_input(
                    "Observation date range",
                    value=(earliest_citizen_date, latest_citizen_date),
                    min_value=earliest_citizen_date,
                    max_value=latest_citizen_date,
                    key="citizen_date_filter",
                )
                if isinstance(citizen_date_selection, tuple) and len(citizen_date_selection) == 2:
                    citizen_times = citizen_view["observation_time_utc"].dt.date
                    citizen_view = citizen_view[
                        citizen_times.between(
                            citizen_date_selection[0], citizen_date_selection[1]
                        )
                    ]

        if events is not None and {"event_id", "station_id", "start_time", "end_time"}.issubset(events.columns):
            citizen_view = associate_observations_with_events(citizen_view, events)
            associated = citizen_view[
                citizen_view["association_status"].str.startswith("supporting observation")
            ]
            st.metric("Observations associated with an event", len(associated))
        else:
            citizen_view["event_id"] = pd.NA
            citizen_view["association_status"] = "event report unavailable"
            st.info("Event association is unavailable until pollution event data is present.")

        citizen_display_columns = [
            "observation_id",
            "observation_time_utc",
            "station_id",
            "observation_location",
            "observation_type",
            "observation_value",
            "observation_description",
            "event_id",
            "association_status",
            "is_valid",
            "validation_issues",
        ]
        citizen_display_columns = [
            column for column in citizen_display_columns if column in citizen_view.columns
        ]
        st.dataframe(
            citizen_view[citizen_display_columns],
            width="stretch",
            hide_index=True,
        )
        if citizen_observation_status.get("invalid_rows", 0):
            st.warning(
                f"{citizen_observation_status['invalid_rows']} observation row(s) need validation. "
                "Rows are retained and shown with validation issues."
            )

st.header("Data Quality & Limitations")
if cleaning is not None:
    quality_metrics = [
        ("Datasets cleaned", "dataset"),
        ("Input rows", "input_rows"),
        ("Output rows", "output_rows"),
        ("Missing before cleaning", "missing_before"),
        ("Missing after cleaning", "missing_after"),
        ("Numeric values imputed", "numerical_values_imputed"),
        ("Invalid values detected", "invalid_values_detected"),
        ("Timestamp rows removed", "timestamps_removed"),
    ]
    quality_values = []
    for label, field in quality_metrics:
        if field == "dataset":
            value = len(cleaning)
        else:
            value = total_column(cleaning, field)
        if value is not None:
            quality_values.append((label, value))
    for start in range(0, len(quality_values), 4):
        quality_columns = st.columns(min(4, len(quality_values) - start))
        for column, (label, value) in zip(
            quality_columns, quality_values[start : start + 4]
        ):
            column.metric(label, f"{value:,}")
else:
    unavailable(cleaning_error)

if evaluation is not None and {"phase", "metric", "value", "explanation"}.issubset(
    evaluation.columns
):
    ground_truth_rows = evaluation[
        (evaluation["phase"] == "evaluation")
        & (evaluation["metric"] == "ground_truth_available")
    ]
    if not ground_truth_rows.empty and str(ground_truth_rows.iloc[0]["value"]).lower() == "false":
        st.warning(
            "No validated anomaly or responsible-source labels are available. "
            "Accuracy, recall, F1, Top-1, Top-3, and false-attribution metrics cannot be measured."
        )
    interpretation = evaluation[
        (evaluation["phase"] == "evaluation")
        & (evaluation["metric"] == "interpretation")
    ]
    if not interpretation.empty:
        st.caption(str(interpretation.iloc[0]["value"]))
    edge_cases = evaluation[evaluation["phase"] == "edge_case"]
    if not edge_cases.empty:
        st.dataframe(
            edge_cases[[column for column in ("metric", "status", "explanation") if column in edge_cases]],
            width="stretch",
            hide_index=True,
        )
else:
    unavailable(evaluation_error)

st.caption(
    "All displayed measurements and explanations come from the generated project "
    "reports and available CSV datasets."
)