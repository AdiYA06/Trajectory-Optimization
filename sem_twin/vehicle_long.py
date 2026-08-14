"""
Longitudinal vehicle model. This is the heart of the twin — get this right
and validated (Section 6 of the roadmap: coast-down + energy balance) before
building anything on top of it.

Two integration approaches, pick one to start:
    (a) time-domain: state = [v, s], integrate with scipy.integrate.solve_ivp
    (b) space-domain: state = [v, t], independent variable = s, dv/ds = F_net/(m*v)
Space-domain is what you'll want eventually for the optimizer (Section 4),
so consider building it that way from the start rather than converting later.
"""

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
        # """Compute the resistive forces at this state (everything except
        # the motor's traction force).
        gradient = self.track.gradient_at(s)
        F_aero  = 0.5 * self.params.air_density_kg_m3 * self.params.Cd * self.params.frontal_area_m2 * v**2
        # direction = np.sign(v) if abs(v) > 1e-3 else 0.0
        F_roll  = self.params.Crr * self.params.mass_kg * 9.81 * np.cos(gradient) # * direction
        F_grad = self.params.mass_kg * 9.81 * np.sin(gradient)
        return F_aero, F_roll, F_grad
        # Keep them separate (don't just sum) — you'll want the breakdown for
        # the energy-balance validation check later.

    def derivatives(self, state: LongState, throttle: float, brake: float) -> LongState:
        """PURE function: no side effects, safe to call multiple times per
        step (RK4 calls this 4x per step for its k1..k4 estimates). Must NOT
        advance the battery — that happens once per real step, in commit_step
        below, using the electrical power at the *final* accepted state."""

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
            # input for integrator: d(state)/dt, not an actual new state. soc is a placeholder

        """Also decide here: what happens if requested torque/rpm is outside
        the motor's physical limits, or F_net can't be delivered? A
        `max_traction_force_at(v)` check belongs either here or in
        powertrain_model.py — worth ad ding before the optimizer starts
        exploiting a gap in the model.
        """

    def commit_step(self, state: LongState, throttle: float, 
                    next_v: float, next_s: float, dt: float) -> LongState:
        """Called once per real timestep, AFTER the integrator (Euler/RK4)
        has already computed next_v and next_s from derivatives(). This is
        where the battery actually advances — exactly once, using the real
        electrical power draw and the real dt, independent of how many times
        derivatives() was probed internally to get there."""
        F_traction = self.powertrain.throttle_to_traction_force(throttle, state.v)
        P_elec = self.powertrain.traction_force_to_electrical_power(F_traction, state.v)["P_elec"]
        battery_state = self.battery.step(P_elec, dt)
        return LongState(v=next_v, s=next_s, t=state.t + dt, soc=battery_state.soc)

        """This is also a natural place to log motor_power_W / battery_power_W
        for SimLog, since you have both the mechanical and electrical power
        numbers in hand right here.
        """
