import numpy as np


class BatteryState:
    def __init__(self, soc: float, voltage: float, current: float):
        self.soc = soc          # 0-1
        self.voltage = voltage  # terminal voltage, V
        self.current = current  # A, positive = discharging


class SimpleBattery:
    def __init__(self, capacity_Ah: float, nominal_voltage_V: float, soc0: float = 1.0):
        self.capacity_As = capacity_Ah * 3600.0
        self.nominal_voltage_V = nominal_voltage_V
        self.soc = soc0

    def step(self, power_elec_W: float, dt_s: float) -> BatteryState:
        current = power_elec_W / self.nominal_voltage_V   # (constant-voltage assumption — this is the simplification you remove in ECMBattery)
        delta_soc = - (current * dt_s) / self.capacity_As
        self.soc += delta_soc
        delta_soc = np.clip(self.soc, 0.0, 1.0)
        return BatteryState(soc=self.soc, voltage=self.nominal_voltage_V, current=current)

class ECMBattery:

    def __init__(self, capacity_Ah: float, r_int_ohm: float,
                 ocv_soc_table: dict, soc0: float = 1.0):
        self.capacity_As = capacity_Ah * 3600.0
        self.r_int = r_int_ohm
        self.ocv_soc_table = ocv_soc_table   # {"soc": [...], "ocv_v": [...]}
        self.soc = soc0

    def ocv_at(self, soc: float) -> float:
        return np.interp(soc, self.ocv_soc_table['soc'], self.ocv_soc_table['ocv_v'])

    def solve_current_for_power(self, power_req_W: float) -> float:
        # P = V_term * I = (OCV - I*R_int) * I
        # R_int*I^2 - OCV*I + P = 0
        OCV = self.ocv_at(self.soc)
        I = (OCV - np.sqrt(OCV**2 - 4*self.r_int*power_req_W)) / (2*self.r_int)
        return I

    def step(self, power_elec_W: float, dt: float) -> BatteryState:
        current = self.solve_current_for_power(power_elec_W)
        delta_soc = -(current * dt) / self.capacity_As
        self.soc += delta_soc
        self.soc = np.clip(self.soc, 0.0, 1.0)
        voltage = self.ocv_at(self.soc) - current * self.r_int
        return BatteryState(soc=self.soc, voltage=voltage, current=current)