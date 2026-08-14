"""
Motor / powertrain model.

Owns the torque-speed efficiency map and converts between mechanical
(traction) power and electrical power drawn from the battery.
"""

import numpy as np
import scipy.interpolate as interp


class MotorMap:
    def __init__(self, rpm_points: np.ndarray, torque_points: np.ndarray,
                 efficiency: np.ndarray, stall_torque_Nm: float = None,
                 no_load_rpm: float = None):
        self.rpm_points = rpm_points        # 1D grid, rpm
        self.torque_points = torque_points   # 1D grid, Nm
        self.efficiency = efficiency        # 2D array, shape (len(torque_points), len(rpm_points))
        # Torque-speed envelope for max_torque_at() below. Simplest possible
        # model: a straight line from (0 rpm, stall_torque_Nm) to
        # (no_load_rpm, 0 Nm). Replace with a real datasheet curve/table
        # when you have one — these two fields are just what's needed for
        # the linear placeholder.
        self.stall_torque_Nm = stall_torque_Nm
        self.no_load_rpm = no_load_rpm

    def efficiency_at(self, torque: float, rpm: float) -> float:
        """Interpolate efficiency at an arbitrary (torque, rpm) point."""
        torque = np.clip(torque, self.torque_points.min(), self.torque_points.max())
        rpm = np.clip(rpm, self.rpm_points.min(), self.rpm_points.max())
        interpolater = interp.RegularGridInterpolator((self.torque_points, self.rpm_points), self.efficiency, bounds_error=False, fill_value=None)
        return float(interpolater([torque, rpm]))

    def max_torque_at(self, rpm: float) -> float:
        """The motor's torque-speed limit envelope: the maximum torque it
        can physically deliver at a given rpm. This is what throttle=1.0
        should mean (see vehicle_long.py::derivatives) — NOT the maximum
        value in your efficiency table's torque_points grid, which is just
        wherever your data happens to stop.

        Typical shape for small BLDC/PMDC motors used in SEM cars: roughly
        constant torque up to a "base speed", then torque falls off with
        increasing rpm above that (constant-power-ish region) as back-EMF
        approaches supply voltage."""
        if self.stall_torque_Nm is None or self.no_load_rpm is None:
            raise ValueError("stall torque and no load rpm must be set")
        T_max = self.stall_torque_Nm * (1-rpm/self.no_load_rpm)
        return max(0.0, T_max)


class Powertrain:
    def __init__(self, motor_map: MotorMap, wheel_radius_m: float, gear_ratio: float = 1.0):
        self.motor_map = motor_map
        self.wheel_radius_m = wheel_radius_m
        self.gear_ratio = gear_ratio

    def speed_to_rpm(self, v: float) -> float:
        """Vehicle speed (m/s) -> motor rpm, via wheel radius and gear ratio."""
        w_wheel = v / self.wheel_radius_m         # rad/s
        w_motor = w_wheel * self.gear_ratio    # rad/s
        rpm = w_motor * 60 / (2 * np.pi)
        return rpm

    def throttle_to_traction_force(self, throttle: float, v: float) -> float:
        """Implements the throttle convention used throughout this project:
        throttle in [0,1] is a FRACTION OF MAX TORQUE AT THE CURRENT MOTOR
        SPEED, not a fraction of max power or max force. See the note in
        vehicle_long.py::derivatives for why."""
        rpm = self.speed_to_rpm(v)
        T_motor = throttle * self.motor_map.max_torque_at(rpm)
        T_wheel = T_motor * self.gear_ratio
        F_traction = T_wheel / self.wheel_radius_m
        return F_traction
        """Brake should get its own equivalent method (brake_to_force) using a
        speed-independent max brake force rather than a torque-speed curve —
        mechanical brakes don't have a back-EMF-style falloff at high rpm.
        """

    def traction_force_to_electrical_power(self, F_traction: float, v: float) -> dict:
        """Given a required traction force at the wheel and current speed,
        work out motor operating point and electrical power draw."""
        rpm = self.speed_to_rpm(v)
        T_wheel = F_traction * self.wheel_radius_m
        T_motor = T_wheel / self.gear_ratio
        eta = self.motor_map.efficiency_at(T_motor, rpm)
        w_motor = rpm * 2 * np.pi / 60  # Convert RPM to rad/s
        P_motor = T_motor * w_motor
        P_elec = P_motor / eta
        return {"T_motor": T_motor, "rpm": rpm, "eta": eta, "P_motor": P_motor, "P_elec": P_elec}
