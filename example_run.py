from sem_twin.config import load_config
from sem_twin.track_model import Track
from sem_twin.vehicle_long import LongitudinalVehicle
from sem_twin.sim_runner import run_lap
from sem_twin.strategies import ConstantSpeedStrategy, PulseAndGlideStrategy, FullAccelerateThenCoastStrategy, OCPReplay
from sem_twin.optimizer import optimize_strategy_params, build_ocp


def main():
    # 1. Everything car-related comes from ONE file. Change values in
    #    config/vehicle_params.yaml, not here.
    cfg = load_config(r"sem_twin\config\vehicle_params.yaml", battery_type="simple")

    # 2. Track is independent of the vehicle — build it once, reuse across
    #    every strategy/optimizer run below.
    # track = Track.make_synthetic_track()
    track = Track.from_gpx(r"sem_twin\Silverstone Circuit Loop.gpx")
    print(f"Track length: {track.length_m} m")

    # 3. Battery is stateful (SoC changes as a lap runs), so build a *fresh*
    #    one per run rather than reusing an instance across multiple laps.
    battery = cfg.build_battery()

    vehicle = LongitudinalVehicle(
        params=cfg.vehicle_params,
        powertrain=cfg.powertrain,
        battery=battery,
        track=track,
        brake_max_force_N=cfg.brake_max_force_N,
    )

    # 4. Run one strategy manually first — cheapest way to sanity-check the
    #    model before handing it to an optimizer.
    # strategy = ConstantSpeedStrategy(target_v=8.0)
    # log = run_lap(vehicle, strategy, track)
    # print("Constant-speed lap: final SoC =", log.soc[-1], " lap time =", log.t[-1])

    # 5. Same config, different strategy — no reloading, no duplication.
    #    Note: battery must be rebuilt fresh for each independent run.
    vehicle.battery = cfg.build_battery()
    pulse_glide = PulseAndGlideStrategy(v_high=9.0, v_low=6.0)
    log2 = run_lap(vehicle, pulse_glide, track)
    print("Pulse-and-glide lap: final SoC =", log2.soc[-1], " lap time =", log2.t[-1])

    # 6. Optimizer reuses the same `vehicle` and `track` — it's just searching
    #    over a strategy's free parameters, rebuilding the battery internally
    #    for each candidate it evaluates (see optimizer.py's TODO for where
    #    that reset needs to happen inside the cost function).
    # pulse_params, pulse_log = optimize_strategy_params(strategy_factory=lambda p: PulseAndGlideStrategy(v_high=p[0], v_low=p[1]),
    #     param_bounds=[(7.0, 12.0), (4.0, 8.0)], # bounds for v_high and v_low
    #     vehicle=vehicle,
    #     track=track,
    #     battery_factory=cfg.build_battery,   # fresh battery per candidate evaluated
    #     lap_time_limit=1000
    # )

    # print("Optimized pulse-and-glide params:", pulse_params)
    # print("Optimized pulse-and-glide lap: final SoC =", pulse_log.soc[-1], " lap time =", pulse_log.t[-1])
    # strategies = {
    #     "constant_speed": ConstantSpeedStrategy(target_v=8.0),
    #     "pulse_glide": PulseAndGlideStrategy(v_high=9.0, v_low=6.0),
    #     "full_accel_then_coast": FullAccelerateThenCoastStrategy(release_v=10.0)
    # }

    # results = {}

    # for name, strategy in strategies.items():
    #     vehicle.battery = cfg.build_battery()
    #     log = run_lap(vehicle, strategy, track)
    #     results[name] = {
    #         "final_soc": log.soc[-1],
    #         "lap_time": log.t[-1]
    #     }
    
    # best_name = max(results, key = lambda n: results[n]["final_soc"])
    # print(f"Best strategy: {best_name}")
    # for name, r in results.items():
    #     print(f"  {name}: final_soc={r['final_soc']}, lap_time={r['lap_time']}s")

    result = build_ocp(vehicle)


if __name__ == "__main__":
    main()
