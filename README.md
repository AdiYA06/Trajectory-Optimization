# SEM Digital Twin — Skeleton

Every function/method that says `raise NotImplementedError` is where you write
the actual physics/logic. The docstrings tell you *what* the function needs
to do and roughly *how*, matching the equations in the project roadmap doc.
Nothing here does any real math yet — it's the scaffolding, not the twin.

`example_run.py` at the project root shows the full wiring end-to-end (config
-> track -> vehicle -> strategy -> sim -> optimizer). It won't run until the
stubs below are filled in, but it's the reference for how the pieces connect.
Needs `pyyaml` (`pip install pyyaml --break-system-packages`).

## Build order (fill things in, in this order)

1. `config.py::load_config` — already fully implemented (it's I/O, not
   physics). Skim it so you know what `config/vehicle_params.yaml` maps to
   in each model class; add your own fields here as you need them.
2. `track_model.py::Track.make_synthetic_track` — get a track you can plot.
3. `vehicle_long.py::LongitudinalVehicle.road_load_forces` and `.derivatives`
   — get the core equation of motion working with a *dummy* constant-efficiency
   motor and `SimpleBattery` first. Don't touch the real motor map yet.
   Note: `derivatives()` is a pure rates function (no battery side effects);
   `commit_step()` is the separate method that actually advances SoC once
   per real timestep. Keep that split — it matters once you use RK4.
4. `integrators.py::euler_step` — implement this before `rk4_step`. It's the
   simplest possible way to advance `state` using `derivatives()`.
5. `sim_runner.py::run_lap` — wire it up with `euler_step`, run
   `ConstantSpeedStrategy`, plot `v(t)` and `s(t)`. This is your first real
   milestone. Swap in `rk4_step` later once the loop itself is correct.
6. `powertrain_model.py::MotorMap` + `Powertrain` — swap the dummy motor for
   the real map interpolation.
7. `battery_model.py::SimpleBattery` fully, then validate energy balance
   against `road_load_forces` output (Section 6 of the roadmap).
8. `strategies.py` — implement all three parametrized strategies.
9. `optimizer.py::optimize_strategy_params` — compare strategies properly.
10. `battery_model.py::ECMBattery` — swap in without touching anything else.
11. `vehicle_lat.py` (not yet created — add when you get here) — bicycle model,
   feeds real `v_limit(s)` back into `track_model.py`.
12. `optimizer.py::build_ocp` — the CasADi optimal control formulation.

## Sanity checks to run as you go

- Plot `track.gradient`, `track.curvature`, `track.v_limit` vs `s` before
  doing anything else — a bug here silently breaks everything downstream.
- After step 3: does the car's terminal speed on a flat straight roughly match
  `F_motor(v) = F_aero(v) + F_roll` solved for v? (steady-state check)
- After step 5: does `energy_in == KE_change + PE_change + losses`, within a
  small numerical tolerance? If not, you have a sign error or a missing term
  somewhere in `derivatives()`.

## Not included yet (intentionally)

- `vehicle_lat.py` (bicycle model) — add once the longitudinal side is
  validated; the roadmap doc has the equations.
- Real motor efficiency data, real OCV-SoC curve, real Cd·A/Crr from a
  coast-down test — the config file has placeholders only.
