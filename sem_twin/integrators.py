"""
Fixed-step integrators for the longitudinal model.

Kept separate from sim_runner.py so you can swap Euler <-> RK4 without
touching the simulation loop, and so you can unit-test a single step in
isolation (e.g. against a known analytic solution) before trusting the
full lap simulation.

Both functions have the same shape: given a state, a function that computes
derivatives, and a step size, return the next state. `vehicle_long.LongState`
supports +, *, etc. is NOT assumed here — see the note in each function about
why state arithmetic needs to be explicit for a plain class.
"""

from sem_twin.vehicle_long import LongState


def euler_step(state: LongState, derivatives_fn: LongState, dt: float) -> LongState:
    """Simplest possible integrator: one derivative evaluation per step."""
    d = derivatives_fn(state)   # d is presumably also a LongState-shaped
                                # object holding rates: dv/dt, ds/dt, etc.
    return LongState(v=state.v + d.v * dt, s=state.s + d.s * dt, t=state.t + dt, soc=state.soc + d.soc * dt)

def state_shifted_by(state: LongState, rate: LongState, scale: float) -> LongState:
    """Helper: state + rate * scale, field by field. Used internally by
    rk4_step's midpoint estimates."""
    v = state.v + rate.v * scale
    s = state.s + rate.s * scale
    t = state.t + rate.t * scale
    soc = state.soc + rate.soc * scale
    return LongState(v=v, s=s, t=t, soc=soc)

def rk4_step(state: LongState, derivatives_fn, dt: float) -> LongState:
    """Classic 4th-order Runge-Kutta. ~4x the cost of euler_step per step,
    but you can usually take a much larger dt for the same accuracy, so it
    often ends up cheaper overall."""
    k1 = derivatives_fn(state)
    k2 = derivatives_fn(state_shifted_by(state, k1, dt/2))
    k3 = derivatives_fn(state_shifted_by(state, k2, dt/2))
    k4 = derivatives_fn(state_shifted_by(state, k3, dt))
    next_v = state.v + (dt/6) * (k1.v + 2*k2.v + 2*k3.v + k4.v)
    next_s = state.s + (dt/6) * (k1.s + 2*k2.s + 2*k3.s + k4.s)
    next_t = state.t + (dt/6) * (k1.t + 2*k2.t + 2*k3.t + k4.t)
    next_soc = state.soc + (dt/6) * (k1.soc + 2*k2.soc + 2*k3.soc + k4.soc)
    return LongState(v=next_v, s=next_s, t=next_t, soc=next_soc)

