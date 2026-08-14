import numpy as np
from sem_twin.config import load_config
from sem_twin.track_model import Track
from sem_twin.vehicle_long import LongitudinalVehicle
from sem_twin.sim_runner import run_lap
from sem_twin.strategies import ConstantSpeedStrategy, PulseAndGlideStrategy, FullAccelerateThenCoastStrategy, OCPReplay
from sem_twin.optimizer import optimize_strategy_params, build_ocp



# 1. Everything car-related comes from ONE file. Change values in
#    config/vehicle_params.yaml, not here.
cfg = load_config(r"config\vehicle_params.yaml", battery_type="ecm")

# 2. Track is independent of the vehicle — build it once, reuse across
#    every strategy/optimizer run below.
# track = Track.make_synthetic_track()
# flat_track = Track(
#     s=np.linspace(0, 645, 100),
#     gradient=np.zeros(100),
#     curvature=np.zeros(100),
#     v_limit=np.full(100, 20.0),
# )
# sol, vars = build_ocp(vehicle_params, motor_map, battery_kwargs, flat_track, ..., N=10)
track = Track.from_gpx(r"Silverstone Circuit Loop.gpx")
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
m_eff = cfg.vehicle_params.mass_kg + cfg.vehicle_params.rotational_inertia_kg

# 4. Run one strategy manually first — cheapest way to sanity-check the
#    model before handing it to an optimizer.
# strategy = ConstantSpeedStrategy(target_v=8.0)
# log = run_lap(vehicle, strategy, track)
# print("Constant-speed lap: final SoC =", log.soc[-1], " lap time =", log.t[-1])

# 5. Same config, different strategy — no reloading, no duplication.
#    Note: battery must be rebuilt fresh for each independent run.
vehicle.battery = cfg.build_battery()

sol, variables = build_ocp(vehicle, cfg.vehicle_params, cfg.motor_map, cfg.battery_kwargs, track,
                            wheel_radius_m=cfg.powertrain.wheel_radius_m,
                            gear_ratio=cfg.powertrain.gear_ratio,
                            brake_max_force=cfg.brake_max_force_N,
                            N = 5000,
                    )

Ek_opt = sol.value(variables["Ek"])
v_opt = np.sqrt(2 * Ek_opt / m_eff)
t_opt = sol.value(variables["t"])
soc_opt = sol.value(variables["soc"])
throttle_opt = sol.value(variables["throttle"])
brake_opt = sol.value(variables["brake"])

# s_dense = np.linspace(0, 30, 15)          # fine spacing near the start
# s_rest = np.linspace(30, track.length_m, 100-14+1)
# s_nodes = np.concatenate([s_dense, s_rest[1:]])
s_nodes = np.linspace(0, track.length_m, 5000+1)
# print(s_nodes[0:40])

vehicle.battery = cfg.build_battery()
replay_strategy = OCPReplay(s_nodes, throttle_opt, brake_opt)
replay_log = run_lap(vehicle, replay_strategy, track)
total_steps = len(replay_log.t)


print("OCP result:")
print("  final time:", t_opt[-1])
print("  final soc:", soc_opt[-1])
print("  v profile:", v_opt)
print("  throttle profile:", throttle_opt)
print("  brake profile:", brake_opt)
print("replay final soc:", replay_log.soc[-1], "vs OCP claimed:", soc_opt[-1])
print("replay lap time:", replay_log.t[-1], "vs OCP claimed:", t_opt[-1])
print(f"stall recovery triggered on {replay_strategy.stall_recovery_count} / {total_steps} steps")
