import numpy as np
import casadi as ca

def ocp_dynamics_function(vehicle_params, motor_map, track, battery_params,
                          wheel_radius_m: float, gear_ratio: float, brake_max_force: float) -> ca.Function:

    # track interpolation functions
    gradient_interp = ca.interpolant("gradient", "linear", [track.s.tolist()], track.gradient.tolist())
    v_limit_interp = ca.interpolant("v_limit", "linear", [track.s.tolist()], track.v_limit.tolist())
    curv_interp = ca.interpolant("curvature", "linear", [track.s.tolist()], track.curvature.tolist())

    # motor interpolation functions
    eff_flat = motor_map.efficiency.flatten(order = "F").tolist()
    eff_interp = ca.interpolant("eff", "linear",
                                 [motor_map.torque_points.tolist(), motor_map.rpm_points.tolist()], eff_flat)

    # symbolic variable
    # v = ca.MX.sym("v")
    Ek = ca.MX.sym("Ek")
    t = ca.MX.sym("t")
    s = ca.MX.sym("s")
    soc = ca.MX.sym("soc")
    throttle = ca.MX.sym("throttle")
    brake = ca.MX.sym("brake")

    m_eff = vehicle_params.mass_kg + vehicle_params.rotational_inertia_kg

    # v_safe = ca.fmax(v, 0.05)
    E_k_safe = ca.fmax(Ek, 1e-3)
    v = ca.sqrt(2.0 * E_k_safe / m_eff)

    theta = gradient_interp(s)

    # road load forces
    F_aero  = (vehicle_params.air_density_kg_m3 * vehicle_params.Cd * vehicle_params.frontal_area_m2 / m_eff) * E_k_safe 
    F_roll  = vehicle_params.Crr * vehicle_params.mass_kg * 9.81 * ca.cos(theta)
    F_grad = vehicle_params.mass_kg * 9.81 * ca.sin(theta)

    # E_k = 0.5 * vehicle_params.mass_kg * (v)**2

    # motor twin
    rpm = (v/wheel_radius_m) * gear_ratio * 60 / (2 * ca.pi)
    T_max = ca.fmax(0.0, motor_map.stall_torque_Nm * (1 - rpm / motor_map.no_load_rpm))
    # traction force formula
    T_motor = throttle * T_max
    F_motor = T_motor * gear_ratio / wheel_radius_m
    F_brake = brake * brake_max_force

    F_net = F_motor - F_brake - F_aero - F_roll - F_grad

    # derivatives (space domain)
    # dv_ds = (F_net / vehicle_params.m_eff) / v_safe
    dEk_ds = F_net
    ds_ds = 1.0
    dt_ds = 1 / v

    T_wheel = F_motor * wheel_radius_m
    T_m = T_wheel / gear_ratio
    eta = ca.fmax(eff_interp(ca.vertcat(T_m, rpm)), 0.3)
    w_motor = rpm * 2 * ca.pi / 60  # Convert RPM to rad/s
    P_motor = T_m * w_motor
    P_elec = P_motor / eta

    ocv_table = battery_params["ocv_soc_table"]
    ocv_interp = ca.interpolant("ocv", "linear", [ocv_table["soc"]], ocv_table["ocv_v"])

    # P = V_term * I = (OCV - I*R_int) * I
    # R_int*I^2 - OCV*I + P = 0
    OCV = ocv_interp(soc)
    r_int = battery_params["r_int_ohm"]
    # nominal_v = battery_params["nominal_voltage_V"]

    discriminant = OCV**2 - 4*r_int*P_elec
    I = (OCV - ca.sqrt(ca.fmax(discriminant, 0.0))) / (2*r_int)

    capacity_As = battery_params["capacity_Ah"] * 3600

    dsoc_ds = - (I * dt_ds) / capacity_As
    # I = P_elec / nominal_v   # see note below re: SimpleBattery vs ECM
    # dsoc_ds = -I * dt_ds / capacity_As 

    state = ca.vertcat(Ek, t, soc)
    rates = ca.vertcat(dEk_ds, dt_ds, dsoc_ds)

    return ca.Function("f", [state, s, throttle, brake], [rates], 
                       ["state", "s", "throttle", "brake"], ["rates"])

def ocp_rk4(f: ca.Function) -> ca.Function:
    state = ca.MX.sym("state", 3)
    s0 = ca.MX.sym("s0")
    s1 = ca.MX.sym("s1")
    throttle = ca.MX.sym("throttle")
    brake = ca.MX.sym("brake")
    ds = s1 - s0

    k1 = f(state, s0, throttle, brake)
    k2 = f(state + ds/2 * k1, s0 + ds/2, throttle, brake)
    k3 = f(state + ds/2 * k2, s0 + ds/2, throttle, brake)
    k4 = f(state + ds * k3, s1, throttle, brake)

    state_next = state + (ds/6) * (k1 + 2*k2 + 2*k3 + k4)
    return ca.Function("rk4_step", [state, s0, s1, throttle, brake], [state_next])


