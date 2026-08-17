import numpy as np

class VehicleParams:
    def __init__(self, mass_kg: float, rotational_inertia_kg: float, 
                 Cd: float, frontal_area_m2: float, air_density_kg_m3: float, Crr: float):
        self.mass_kg = mass_kg
        self.rotational_inertia_kg = rotational_inertia_kg
        self.Cd = Cd
        self.frontal_area_m2 = frontal_area_m2
        self.air_density_kg_m3 = air_density_kg_m3
        self.Crr = Crr

    @property
    def m_eff(self) -> float:
        return self.mass_kg + self.rotational_inertia_kg


class LongState:
    def __init__(self, v: float, s: float, t: float, soc: float):
        self.v = v      # speed, m/s
        self.s = s      # distance, m
        self.t = t      # time, s
        self.soc = soc  # battery state of charge, 0-1

class LongitudinalVehicle:
    def __init__(self, params: VehicleParams, powertrain, battery, track, brake_max_force_N: float):
        self.params = params
        self.powertrain = powertrain
        self.battery = battery
        self.track = track
        self.brake_max_force_N = brake_max_force_N  # from config.py's SimConfig

    def road_load_forces(self, v: float, s: float) -> dict:
        gradient = self.track.gradient_at(s)
        F_aero  = 0.5 * self.params.air_density_kg_m3 * self.params.Cd * self.params.frontal_area_m2 * v**2
        F_roll  = self.params.Crr * self.params.mass_kg * 9.81 * np.cos(gradient) # * direction
        F_grad = self.params.mass_kg * 9.81 * np.sin(gradient)
        return F_aero, F_roll, F_grad
    def derivatives(self, state: LongState, throttle: float, brake: float) -> LongState:
        # Given current state and driver inputs, return d(state)/dt — a
        # LongState-shaped object holding rates, not an actual new state.
        F_aero, F_roll, F_grad = self.road_load_forces(state.v, state.s)
        F_traction = self.powertrain.throttle_to_traction_force(throttle, state.v)
        # if F_grad > F_traction:
        #     return f"traction force {F_traction} N insufficient to overcome gradient force {F_grad} N at s={state.s} m, v={state.v} m/s"
        F_brake = brake * self.brake_max_force_N
        F_net = F_traction - F_brake - F_aero - F_roll - F_grad
        dv_dt = F_net / self.params.m_eff
        ds_dt = state.v
        dt_dt = 1.0
        dsoc_dt = 0.0
        # Reusing the same LongState class to hold rates-of-change
        return LongState(v = dv_dt, s = ds_dt, t = dt_dt, soc = dsoc_dt)

    def commit_step(self, state: LongState, throttle: float, 
                    next_v: float, next_s: float, dt: float) -> LongState:
        F_traction = self.powertrain.throttle_to_traction_force(throttle, state.v)
        P_elec = self.powertrain.traction_force_to_electrical_power(F_traction, state.v)["P_elec"]
        battery_state = self.battery.step(P_elec, dt)
        return LongState(v=next_v, s=next_s, t=state.t + dt, soc=battery_state.soc)
