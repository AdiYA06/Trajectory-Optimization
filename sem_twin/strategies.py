"""
Driving strategies: a strategy is just a function/callable that takes the
current state and returns (throttle, brake) in [0,1]. Keeping this as a
uniform interface means sim_runner.py doesn't care which strategy it's
running, and the optimizer (Section 4) is really just "search over the
parameters of one of these functions".
"""
import numpy as np
from typing import Protocol
from sem_twin.vehicle_long import LongState


class Strategy(Protocol):
    def __call__(self, state: LongState, track) -> tuple[float, float]:
        """Return (throttle, brake), each in [0, 1]."""
        ...


class ConstantSpeedStrategy:
    """Baseline: simple proportional controller trying to hold target_v."""

    def __init__(self, target_v: float, kp: float = 1.0):
        self.target_v = target_v
        self.kp = kp

    def __call__(self, state: LongState, track) -> tuple[float, float]:
        diff_v = self.target_v - state.v
        if diff_v > 0:
            throttle = np.clip(self.kp * diff_v, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-self.kp * diff_v, 0, 1)
        return [throttle, brake]
        """A proper controller might use PI rather than P to remove steady-state
        error on gradients — worth revisiting once you see it misbehave uphill."""

class PulseAndGlideStrategy:
    """Accelerate to v_high, then coast (throttle=0) until v_low, repeat."""

    def __init__(self, v_high: float, v_low: float):
        self.v_high = v_high
        self.v_low = v_low
        self._accelerating = True   # internal mode flag

    def __call__(self, state: LongState, track) -> tuple[float, float]:
        if self._accelerating:
            if state.v >= self.v_high: 
                self._accelerating = False
            throttle = 1.0 if self._accelerating else 0.0
            return [throttle, 0.0]
        else:
            if state.v <= self.v_low:
                self._accelerating = True
            throttle = 1.0 if self._accelerating else 0.0
            return [throttle, 0.0]
        # return is throttle, brake. brale always 0.0 in glide
        """Note the strategy is stateful (mode flag) — reset it between runs,
        e.g. give it a `.reset()` method, or construct a fresh instance per run.
        """


class FullAccelerateThenCoastStrategy:
    """Accelerate fully once, then coast for the rest of the lap. Useful as
    the crudest possible baseline alongside ConstantSpeedStrategy.
    """
    def __init__(self, release_v: float):
        self.release_v = release_v

    def __call__(self, state: LongState, track) -> tuple[float, float]:
        if state.v < self.release_v:
            return [1.0, 0.0]
        else:
            return [0.0, 0.0]

class OCPReplay:
    def __init__(self, s_nodes, throttle_opt, brake_opt, stall_v_threshold=0.3):
        self.s_nodes = s_nodes[:-1]
        self.throttle_opt = throttle_opt
        self.brake_opt = brake_opt
        self.stall_v_threshold = stall_v_threshold
        self.stall_recovery_count = 0

    def __call__(self, state, track):
        throttle = np.interp(state.s, self.s_nodes, self.throttle_opt)
        brake = np.interp(state.s, self.s_nodes, self.brake_opt)
        if state.v < self.stall_v_threshold:
            self.stall_recovery_count += 1
            throttle = max(throttle, 0.5)   # enough to guarantee F_motor > F_roll even near-stall
            brake = 0.0
        return throttle, brake
