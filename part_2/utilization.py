"""Thrust utilisation for Project Part 2, Simulation 5."""

import numpy as np


def thrust_utilization(logs, thrusters) -> np.ndarray:
    """Calculate thrust utilisation [%] at each time step."""
    u = np.asarray(logs.u, dtype=float)  # Actual thrust [N]
    u_max = np.array([thruster.u_max for thruster in thrusters], dtype=float)

    if u.shape != (len(logs.t), len(u_max)):
        raise ValueError("logs.u must contain one column per thruster and one row per time step.")
    if len(u_max) == 0 or np.any(u_max <= 0):
        raise ValueError("Maximum thrusts must be positive.")
    if not np.isfinite(u).all() or not np.isfinite(u_max).all():
        raise ValueError("Thrust values must be finite.")

    return 100.0 * np.sum(np.abs(u), axis=1) / np.sum(u_max)


def average_utilization(logs, thrusters, t_from: float | None = None) -> float:
    """Calculate average utilisation [%] from t_from (default: second half)."""
    t = np.asarray(logs.t, dtype=float)
    if t.size == 0:
        raise ValueError("The simulation log is empty.")

    if t_from is None:
        t_from = (t[0] + t[-1]) / 2

    selected = t >= t_from
    if not np.any(selected):
        raise ValueError("No samples in the selected time interval.")

    U_T = thrust_utilization(logs, thrusters)
    return float(np.mean(U_T[selected]))
