# SEM Twin

A lightweight longitudinal simulation project for an electric vehicle energy and performance twin. The code is organized around a config-driven vehicle model, a battery and powertrain, track data, and simple strategy or optimization workflows.

## What this project includes

- Vehicle and powertrain configuration from a YAML file
- Track definitions, including GPX-based road loading
- Longitudinal dynamics for a battery-electric vehicle
- Battery models for simple and ECM-style behavior
- Lap simulation using driver strategies
- Basic optimization routines for strategy parameter tuning

## Project layout

- `config/vehicle_params.yaml` — baseline vehicle, drivetrain, motor, and battery settings
- `example_run.py` — example entry point showing how the model is wired together
- `sem_twin/` — core simulation modules
  - `battery_model.py`
  - `config.py`
  - `integrators.py`
  - `ocp_dynamics.py`
  - `optimizer.py`
  - `powertrain_model.py`
  - `sim_runner.py`
  - `strategies.py`
  - `track_model.py`
  - `vehicle_long.py`

## Quick start

From the project root, run:

```bash
python example_run.py
```

This loads the configuration, builds a vehicle, simulates a lap, and demonstrates the strategy/optimization flow.

## Configuration

The main parameter file is:

```text
config/vehicle_params.yaml
```

Adjust values such as:

- vehicle mass
- aerodynamic drag and rolling resistance
- battery capacity and voltage
- motor efficiency map and torque envelope
- wheel radius and gear ratio
- brake force limit

## Track data

The project supports synthetic tracks and GPX imports. A GPX example is included in the repository root:

```text
Silverstone Circuit Loop.gpx
```

## Notes on the model

This project is meant to be an iterative engineering model rather than a finished turnkey app. The typical workflow is:

1. Define parameters in the YAML file.
2. Build a vehicle and track.
3. Run a strategy over one lap.
4. Compare lap time and SoC outcomes.
5. Use the optimizer to sweep strategy parameters.

## Example workflow

```python
from sem_twin.config import load_config
from sem_twin.track_model import Track
from sem_twin.vehicle_long import LongitudinalVehicle
from sem_twin.sim_runner import run_lap
from sem_twin.strategies import PulseAndGlideStrategy

cfg = load_config("config/vehicle_params.yaml", battery_type="simple")
track = Track.from_gpx("Silverstone Circuit Loop.gpx")
vehicle = LongitudinalVehicle(
    params=cfg.vehicle_params,
    powertrain=cfg.powertrain,
    battery=cfg.build_battery(),
    track=track,
    brake_max_force_N=cfg.brake_max_force_N,
)

log = run_lap(vehicle, PulseAndGlideStrategy(v_high=9.0, v_low=6.0), track)
print(log.t[-1], log.soc[-1])
```

## Requirements

This project is built with Python and commonly relies on:

- NumPy
- PyYAML
- GPX parsing support when loading track files

## Status

The repository is structured as a modeling and simulation sandbox for vehicle performance and energy estimation. It is suitable for experimentation, tuning, and extending the model as the project evolves.
