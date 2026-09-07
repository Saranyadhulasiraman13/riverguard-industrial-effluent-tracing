# RiverGuard prototype data

All files in this directory are **synthetic and simulated**. They were created only for RiverGuard prototype testing and do not represent real environmental measurements, real facilities, or real citizen reports.

## Data dictionary

- `sensors.csv`: Synthetic monitoring-station metadata for four stations along the Willow River.
- `sensor_readings.csv`: Six-hourly synthetic turbidity readings. The higher values represent simulated pollution-spike scenarios.
- `flow_direction.csv`: Synthetic downstream links observed at each reading timestamp. The network is S1 -> S2 -> S3 -> S4.
- `industrial_units.csv`: Synthetic industrial units associated with monitoring stations.
- `discharge_schedule.csv`: Synthetic scheduled discharge windows for each unit.
- `unit_operations.csv`: Synthetic unit status observations, including active, maintenance, and idle states.
- `citizen_observations.csv`: Synthetic reports aligned with selected simulated events.

## Scenarios included

- A downstream spike after the U1 scheduled discharge and active operation window, providing a plausible source for inspection.
- A downstream spike during the U2 discharge window while U2 is in maintenance, providing a timing match but weaker operational evidence.
- An upper-reach spike with no matching discharge schedule, providing a no-confident-match case.

These scenarios are deliberately designed for explainable source-ranking tests. They must not be interpreted as proof that any industry caused pollution.
