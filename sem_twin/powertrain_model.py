import numpy as np
import scipy.interpolate as interp


class MotorMap:
    def __init__(self, rpm_points: np.ndarray, torque_points: np.ndarray,
                 efficiency: np.ndarray, stall_torque_Nm: float = None,
                 no_load_rpm: float = None):
        self.rpm_points = rpm_points        # 1D grid, rpm
        self.torque_points = torque_points   # 1D grid, Nm
        self.efficiency = efficiency        # 2D array, shape (len(torque_points), len(rpm_points))
        self.stall_torque_Nm = stall_torque_Nm
        self.no_load_rpm = no_load_rpm

    def efficiency_at(self, torque: float, rpm: float) -> float:
        torque = np.clip(torque, self.torque_points.min(), self.torque_points.max())
        rpm = np.clip(rpm, self.rpm_points.min(), self.rpm_points.max())
        interpolater = interp.RegularGridInterpolator((self.torque_points, self.rpm_points), self.efficiency, bounds_error=False, fill_value=None)
        return float(interpolater([torque, rpm]))

    def max_torque_at(self, rpm: float) -> float:
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
        w_wheel = v / self.wheel_radius_m         # rad/s
        w_motor = w_wheel * self.gear_ratio    # rad/s
        rpm = w_motor * 60 / (2 * np.pi)
        return rpm

    def throttle_to_traction_force(self, throttle: float, v: float) -> float:
        rpm = self.speed_to_rpm(v)
        T_motor = throttle * self.motor_map.max_torque_at(rpm)
        T_wheel = T_motor * self.gear_ratio
        F_traction = T_wheel / self.wheel_radius_m
        return F_traction

    def traction_force_to_electrical_power(self, F_traction: float, v: float) -> dict:
        rpm = self.speed_to_rpm(v)
        T_wheel = F_traction * self.wheel_radius_m
        T_motor = T_wheel / self.gear_ratio
        eta = self.motor_map.efficiency_at(T_motor, rpm)
        w_motor = rpm * 2 * np.pi / 60  # Convert RPM to rad/s
        P_motor = T_motor * w_motor
        P_elec = P_motor / eta
        return {"T_motor": T_motor, "rpm": rpm, "eta": eta, "P_motor": P_motor, "P_elec": P_elec}
