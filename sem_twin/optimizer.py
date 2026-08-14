import numpy as np
import casadi as ca
from sem_twin.vehicle_long import LongitudinalVehicle
from sem_twin.strategies import ConstantSpeedStrategy
from sem_twin.ocp_dynamics import ocp_dynamics_function, ocp_rk4
from scipy.optimize import minimize
from sem_twin.sim_runner import run_lap


def optimize_strategy_params(strategy_factory, param_bounds, vehicle, track,
                              battery_factory, lap_time_limit=None):
    """Search over a strategy's free parameters to minimise energy (or some
    other objective) for one lap."""

    def cost(params):
        vehicle.battery = battery_factory()
        strategy = strategy_factory(params)
        log = run_lap(vehicle, strategy, track)
        energy_used = np.trapezoid(log.battery_power_W, log.t)
        lap_time = log.t[-1]
        # energy_cost = 1.0 - log.soc[-1]
        if lap_time_limit is not None and lap_time > lap_time_limit:
            penalty = 1e6 * (lap_time - lap_time_limit)
            return energy_used + penalty
            # energy_cost += 1e6 * (lap_time - lap_time_limit)
        return energy_used

    x0 = np.mean(param_bounds, axis=1)  # start in the middle of the bounds
    result = minimize(cost, x0, bounds=param_bounds, method="Nelder-Mead")
    optimal_params = result.x
    optimal_strategy = strategy_factory(optimal_params)
    vehicle.battery = battery_factory()  # reset battery for final run
    log = run_lap(vehicle, optimal_strategy, track)
    return optimal_params, log


def build_ocp(vehicle, vehicle_params, motor_map, battery_params, track,
               wheel_radius_m, gear_ratio, brake_max_force, N: int = 200):
    v_limit_interp = ca.interpolant("v_limit", "linear", [track.s.tolist()], track.v_limit.tolist())

    opti = ca.Opti()
    s_nodes = np.linspace(0, track.length_m, N+1)

    m_eff = vehicle_params.mass_kg + vehicle_params.rotational_inertia_kg

    # Decision variables — Ek replaces v as the state
    Ek = opti.variable(N+1)
    t = opti.variable(N+1)
    soc = opti.variable(N+1)
    throttle = opti.variable(N)
    brake = opti.variable(N)

    f = ocp_dynamics_function(vehicle_params, motor_map, track, battery_params,
                               wheel_radius_m, gear_ratio, brake_max_force)
    rk4_step = ocp_rk4(f)

    avg_speed_min_kmh = 15.0
    avg_speed_min_ms = avg_speed_min_kmh * 1000 / 3600
    lap_time_limit = track.length_m / avg_speed_min_ms

    v_limit_at_nodes = np.array([float(v_limit_interp(s)) for s in s_nodes])
    Ek_limit_at_nodes = 0.5 * m_eff * v_limit_at_nodes**2     # v_limit -> Ek_limit
    v_min = 0.5
    Ek_min = 0.5 * m_eff * v_min**2                            # same floor, in Ek terms

    # 1. State bounds & boundary conditions — all in Ek now
    opti.subject_to(opti.bounded(Ek_min, Ek, Ek_limit_at_nodes))
    opti.subject_to(Ek[0] == Ek_min)
    opti.subject_to(t[0] == 0.0)
    opti.subject_to(soc[0] == 1.0)
    opti.subject_to(t[N] <= lap_time_limit)

    # 2. Control bounds — unchanged
    opti.subject_to(opti.bounded(0.0, throttle, 1.0))
    opti.subject_to(opti.bounded(0.0, brake, 1.0))

    # 3. Dynamic shooting constraints — state vector is now [Ek, t, soc]
    for i in range(N):
        state_i = ca.vertcat(Ek[i], t[i], soc[i])
        state_next_predicted = rk4_step(state_i, s_nodes[i], s_nodes[i+1], throttle[i], brake[i])
        opti.subject_to(Ek[i+1] == state_next_predicted[0])
        opti.subject_to(t[i+1] == state_next_predicted[1])
        opti.subject_to(soc[i+1] == state_next_predicted[2])

    for k in range(N):
        grad_k = track.gradient_at(s_nodes[k])
        smooth_ramp = 0.5 * (1.0 + ca.tanh((grad_k - 0.02) / 0.005))
        min_throttle = 0.3 * smooth_ramp
        opti.subject_to(throttle[k] >= min_throttle)   # <-- uses `i`, not `k`

    # 5. Objective — unchanged, soc is still soc
    opti.minimize(1.0 - soc[N])

    # 6. Initial guess — convert v_guess to Ek_guess
    log = run_lap(vehicle, ConstantSpeedStrategy(target_v=15.0), track)
    v_guess = np.interp(s_nodes, log.s, log.v)
    v_guess = np.maximum(v_min, v_guess)
    Ek_guess = 0.5 * m_eff * v_guess**2

    t_guess = np.interp(s_nodes, log.s, log.t)
    soc_guess = np.interp(s_nodes, log.s, log.soc)

    opti.set_initial(Ek, Ek_guess)
    opti.set_initial(t, t_guess)
    opti.set_initial(soc, soc_guess)
    opti.set_initial(throttle, 0.3)
    opti.set_initial(brake, 0.0)

    opti.solver("ipopt", {}, {
                  "max_iter": 3000,
                  "tol": 1e-4,
                  "acceptable_tol": 1e-3,
                  "acceptable_iter": 100,   # accept if "acceptable" criteria hold for 15 consecutive iterations
                  "print_level": 5,
              })
    try:
        sol = opti.solve()
        return sol, {"Ek": Ek, "t": t, "soc": soc, "throttle": throttle, "brake": brake}
    except RuntimeError:
        Ek_debug = opti.debug.value(Ek)
        v_debug = np.sqrt(2 * Ek_debug / m_eff)
        print("v at each node (from Ek):", v_debug)
        print("min v:", v_debug.min(), "at node", v_debug.argmin())
        raise