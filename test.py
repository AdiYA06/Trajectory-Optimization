from sem_twin.config import load_config
from sem_twin.track_model import Track
from sem_twin.ocp_dynamics import ocp_dynamics_function

cfg = load_config("config/vehicle_params.yaml", battery_type="ecm")   # ECM needed for ocv_soc_table
track = Track.make_synthetic_track()

f = ocp_dynamics_function(
    cfg.vehicle_params,
    cfg.motor_map,
    track,
    cfg.battery_kwargs,                       # plain dict, NOT cfg.build_battery()
    wheel_radius_m=cfg.powertrain.wheel_radius_m,
    gear_ratio=cfg.powertrain.gear_ratio,
    brake_max_force=cfg.brake_max_force_N,
)

# Evaluate at one concrete point — plain numbers in, plain numbers out
rates = f([5.0, 10.0, 0.95], 100.0, 1.0, 0.0)   # state=[v,t,soc], s=100, throttle=1, brake=0
print("dv_ds, dt_ds, dsoc_ds =", rates)