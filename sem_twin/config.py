import numpy as np
import yaml

from sem_twin.vehicle_long import VehicleParams
from sem_twin.powertrain_model import MotorMap, Powertrain
from sem_twin.battery_model import SimpleBattery, ECMBattery


class SimConfig:
    """Bundles everything built from the config file. Pass this single
    object around instead of five separate arguments — e.g.
    `optimizer.optimize_strategy_params(..., config.vehicle_params, ...)`
    or unpack whichever pieces a given function actually needs.
    """

    def __init__(self, vehicle_params: VehicleParams, motor_map: MotorMap,
                 powertrain: Powertrain, battery_type: str, battery_kwargs: dict,
                 brake_max_force_N: float):
        self.vehicle_params = vehicle_params
        self.motor_map = motor_map
        self.powertrain = powertrain
        self.battery_type = battery_type      # "simple" or "ecm"
        self.battery_kwargs = battery_kwargs  # passed to the battery constructor
        self.brake_max_force_N = brake_max_force_N

    def build_battery(self):
        """Construct a fresh battery instance from config. Call this once per
        simulation run (not once for the whole program) since batteries hold
        mutable state (SoC) that a run mutates — you don't want two lap runs
        accidentally sharing one battery's state.
        """
        if self.battery_type == "simple":
            return SimpleBattery(**self.battery_kwargs)
        elif self.battery_type == "ecm":
            return ECMBattery(**self.battery_kwargs)
        else:
            raise ValueError(f"Unknown battery_type: {self.battery_type!r}")


def load_config(path: str, battery_type: str = "simple") -> SimConfig:
    """Parse the YAML file into a SimConfig.

    Args:
        path: path to vehicle_params.yaml (or your own variant — see the
            "parameter studies" note at the bottom of this file).
        battery_type: "simple" to use SimpleBattery (recommended while you're
            still on roadmap steps 1-5), "ecm" once ECMBattery is implemented
            (step 8/9). Switching later is a one-line change here, nowhere else.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    vehicle_params = VehicleParams(
        mass_kg=raw["mass_kg"],
        rotational_inertia_kg=raw["rotational_inertia_kg"],
        Cd=raw["aero"]["Cd"],
        frontal_area_m2=raw["aero"]["frontal_area_m2"],
        air_density_kg_m3=raw["aero"]["air_density_kg_m3"],
        Crr=raw["rolling_resistance"]["Crr"],
    )

    motor_map = MotorMap(
        rpm_points=np.array(raw["motor"]["rpm_points"], dtype=float),
        torque_points=np.array(raw["motor"]["torque_points"], dtype=float),
        efficiency=np.array(raw["motor"]["efficiency_table"], dtype=float),
        stall_torque_Nm=raw["motor"]["stall_torque_Nm"],
        no_load_rpm=raw["motor"]["no_load_rpm"],
    )

    powertrain = Powertrain(
        motor_map=motor_map,
        wheel_radius_m=raw["drivetrain"]["wheel_radius_m"],
        gear_ratio=raw["drivetrain"]["gear_ratio"],
    )

    if battery_type == "simple":
        battery_kwargs = dict(
            capacity_Ah=raw["battery"]["capacity_Ah"],
            nominal_voltage_V=raw["battery"]["nominal_voltage_V"],
        )
    elif battery_type == "ecm":
        battery_kwargs = dict(
            capacity_Ah=raw["battery"]["capacity_Ah"],
            r_int_ohm=raw["battery"]["internal_resistance_ohm"],
            ocv_soc_table=raw["battery"]["ocv_soc_table"],
        )
    else:
        raise ValueError(f"Unknown battery_type: {battery_type!r}")

    return SimConfig(
        vehicle_params=vehicle_params,
        motor_map=motor_map,
        powertrain=powertrain,
        battery_type=battery_type,
        battery_kwargs=battery_kwargs,
        brake_max_force_N=raw["brakes"]["max_force_N"],
    )


# --- Parameter studies -------------------------------------------------
# For "what if the car were 5 kg heavier" type questions (useful when
# reporting on optimizer sensitivity), don't hand-edit the YAML back and
# forth. Either:
#   (a) keep separate files (vehicle_params.yaml, vehicle_params_heavy.yaml)
#       and load_config() whichever one you need, or
#   (b) load once, then build a modified copy in code:
#       cfg = load_config("config/vehicle_params.yaml")
#       cfg.vehicle_params.mass_kg += 5.0   # mutate the already-built object
# (b) is quicker for a one-off sweep; (a) is better if "heavier car" is a
# scenario you'll want to reproduce and reference later.
