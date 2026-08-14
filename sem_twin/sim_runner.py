import pandas as pd
from sem_twin.vehicle_long import LongitudinalVehicle, LongState
from sem_twin.integrators import euler_step, rk4_step


class SimLog:
    def __init__(self):
        self.t = []
        self.s = []
        self.v = []
        self.a = []
        self.throttle = []
        self.brake = []
        self.motor_power_W = []
        self.battery_power_W = []
        self.soc = []

    def to_dataframe(self):
        return pd.DataFrame(self.__dict__)


def run_lap(vehicle: LongitudinalVehicle, strategy, track, dt: float = 0.05, 
            max_time: float = 1000) -> SimLog:
    """Run one lap (until state.s >= track.length_m) with a fixed-step
    integrator, logging everything along the way."""

    state = LongState(v=0, s=0, t=0, soc=1.0)
    log = SimLog()
    while state.s < track.length_m and state.t < max_time:
        throttle, brake = strategy(state, track)
        derivatives = lambda state: vehicle.derivatives(state, throttle, brake)
        next_state = rk4_step(state, derivatives, dt)
        if next_state.v < 0:
            next_state.v = 0
            next_state.s = state.s
        state = vehicle.commit_step(state, throttle, next_v=next_state.v, next_s=next_state.s, dt=dt)
        f_traction = vehicle.powertrain.throttle_to_traction_force(throttle, state.v)
        log.t.append(state.t)
        log.s.append(state.s)
        log.v.append(state.v)
        log.a.append((next_state.v - state.v)/dt)
        log.throttle.append(throttle)
        log.brake.append(brake)
        log.motor_power_W.append(vehicle.powertrain.traction_force_to_electrical_power(f_traction, state.v)["P_motor"])
        log.battery_power_W.append(vehicle.powertrain.traction_force_to_electrical_power(f_traction, state.v)["P_elec"])
        log.soc.append(state.soc)
        if int(state.t * 10) % 100 == 0:   # print roughly every 10 seconds
            print(f"t={state.t:.1f} v={state.v:.3f} s={state.s:.2f}")
    return log
    """Start with euler_step to get the loop itself working end-to-end: it's
    one line simpler per step and easier to debug when something's wrong
    with the *loop* rather than the *integration accuracy*. Swap in rk4_step
    once the loop is correct and you're ready to care about accuracy.

    Things to watch for:
        - What happens if SoC hits 0 mid-lap? Decide: cap electrical power,
          or let the vehicle coast, or raise an error — this matters once
          you're comparing energy-hungry strategies.
        - What happens right at the finish line (s slightly overshoots
          track.length_m in the last step)? Fine to ignore for now, but note
          it if you need exact lap-time precision later.
    """