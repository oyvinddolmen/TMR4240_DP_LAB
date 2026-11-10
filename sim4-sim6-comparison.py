"""
sim4-sim6-comparison.py

Compare EKF closed-loop station keeping in the design sea (Simulation 4)
and the heavier sea (Simulation 6). The primary comparison uses 0-600 s
for both runs. The complete 0-1000 s heavy-sea run is reported separately.

Run from the TMR4240_DP_LAB project directory:
    python3.11 sim4-sim6-comparison.py

Results are printed in the terminal; no CSV or figures are saved.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Do not open the scenario plotting windows.

import run_case_part_2 as runner


EVAL_TIME = 600.0
LF_WINDOW_SECONDS = 30.0
THRUST_LIMITS_KN = np.array([32.0, 80.0, 80.0])
THRUST_RATE_LIMITS_KNS = np.array([4.0, 10.0, 10.0])
AZIMUTH_RATE_LIMIT_DEGS = 12.0


def get_3dof(values):
    """Select North/East/heading or surge/sway/yaw from 6-DOF histories."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D history, got shape {data.shape}.")
    if data.shape[1] >= 6:
        return data[:, [0, 1, 5]]
    if data.shape[1] == 3:
        return data
    raise ValueError(f"Expected 3 or 6 columns, got {data.shape[1]}.")


def get_observer_history(logs, kind):
    """Find the logged observer estimate, allowing common field names."""
    candidates = (
        ["eta_hat", "eta_est", "eta_estimated", "eta_obs", "eta_lf",
         "est_eta", "observer_eta", "eta_observer", "eta_filtered"]
        if kind == "eta" else
        ["nu_hat", "nu_est", "nu_estimated", "nu_obs", "nu_lf",
         "est_nu", "observer_nu", "nu_observer", "nu_filtered"]
    )
    for field in candidates:
        values = getattr(logs, field, None)
        if values is not None:
            data = np.asarray(values)
            if data.ndim == 2 and data.shape[1] in (3, 6):
                return get_3dof(data), field

    available = sorted(
        field for field in vars(logs)
        if any(s in field.lower() for s in ("est", "hat", "obs", "lf"))
    ) if hasattr(logs, "__dict__") else []
    raise AttributeError(
        f"Could not find logged EKF {kind} estimates. "
        f"Possible relevant fields: {available}. "
        "Check the field names in simulation/plotter_part_2.py "
        "(plot_observer) and add the correct field to candidates."
    )


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def smooth_1d(values, width):
    """Centred moving average with endpoint padding (no zero-padding bias)."""
    if width <= 1:
        return np.asarray(values, dtype=float).copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(np.asarray(values, dtype=float), (left, right), mode="edge")
    sums = np.cumsum(np.concatenate(([0.0], padded)))
    return (sums[width:] - sums[:-width]) / width


def lf_pose(truth, dt):
    """Approximate low-frequency truth with a centred 30 s moving average."""
    width = max(1, int(round(LF_WINDOW_SECONDS / dt)))
    if width % 2 == 0:
        width += 1
    output = np.column_stack([
        smooth_1d(truth[:, 0], width),
        smooth_1d(truth[:, 1], width),
        np.arctan2(
            smooth_1d(np.sin(truth[:, 2]), width),
            smooth_1d(np.cos(truth[:, 2]), width),
        ),
    ])
    return output


def metrics(logs, end_time):
    truth_all = get_3dof(logs.eta)
    n = len(truth_all)
    time = (np.asarray(logs.t, dtype=float).reshape(-1)
            if hasattr(logs, "t") else np.arange(n) * runner.DT)
    if len(time) != n:
        raise ValueError("Time and pose histories have different lengths.")
    dt = float(np.median(np.diff(time)))
    if abs(dt - runner.DT) > 1e-5:
        raise ValueError(f"Unexpected logged time step {dt} s.")
    ids = np.flatnonzero((time >= -1e-8) & (time <= end_time + 1e-8))
    if len(ids) < 2 or time[ids[-1]] < end_time - 2 * dt:
        raise ValueError(f"The requested interval 0-{end_time:g} s is unavailable.")

    truth = truth_all[ids]
    desired = get_3dof(logs.sp)[ids]
    estimate_all, eta_field = get_observer_history(logs, "eta")
    estimate = estimate_all[ids]
    lf_truth = lf_pose(truth, dt)

    total_error = truth - desired
    total_error[:, 2] = wrap(total_error[:, 2])
    lf_error = lf_truth - desired
    lf_error[:, 2] = wrap(lf_error[:, 2])
    observer_error = estimate - lf_truth
    observer_error[:, 2] = wrap(observer_error[:, 2])

    total_pos = np.linalg.norm(total_error[:, :2], axis=1)
    lf_pos = np.linalg.norm(lf_error[:, :2], axis=1)
    estimate_pos = np.linalg.norm(observer_error[:, :2], axis=1)

    thrust_kn = np.asarray(logs.u, dtype=float)[ids, :3] / 1000.0
    thrust_rate = np.diff(thrust_kn, axis=0) / dt
    alpha_rad = np.asarray(logs.alpha, dtype=float)[ids, :3]
    az_rate = np.rad2deg(np.diff(np.unwrap(alpha_rad, axis=0), axis=0) / dt)

    desired_wrench = get_3dof(logs.tau_d)[ids]
    applied_wrench = get_3dof(logs.tau_thr)[ids]
    residual = (desired_wrench - applied_wrench) / 1000.0

    result = {
        "RMS total position error [m]": rms(total_pos),
        "Max total position error [m]": float(np.max(total_pos)),
        "RMS total heading error [deg]": rms(np.rad2deg(total_error[:, 2])),
        "Max total heading error [deg]": float(np.max(np.abs(np.rad2deg(total_error[:, 2])))),
        "RMS LF position error [m]": rms(lf_pos),
        "Max LF position error [m]": float(np.max(lf_pos)),
        "RMS LF heading error [deg]": rms(np.rad2deg(lf_error[:, 2])),
        "RMS EKF position error vs LF [m]": rms(estimate_pos),
        "RMS EKF heading error vs LF [deg]": rms(np.rad2deg(observer_error[:, 2])),
        "Mean total absolute thrust [kN]": float(np.mean(np.sum(np.abs(thrust_kn), axis=1))),
        "Mean thrust activity [kN/s]": float(np.mean(np.sum(np.abs(thrust_rate), axis=1))),
        "Max bow thrust [kN]": float(np.max(np.abs(thrust_kn[:, 0]))),
        "Max azimuth 1 thrust [kN]": float(np.max(np.abs(thrust_kn[:, 1]))),
        "Max azimuth 2 thrust [kN]": float(np.max(np.abs(thrust_kn[:, 2]))),
        "RMS Fx residual [kN]": rms(residual[:, 0]),
        "RMS Fy residual [kN]": rms(residual[:, 1]),
        "RMS Mz residual [kNm]": rms(residual[:, 2]),
    }

    try:
        nu_est_all, nu_field = get_observer_history(logs, "nu")
        nu_true = get_3dof(logs.nu)[ids]
        nu_est = nu_est_all[ids]
        nu_lf = np.column_stack([
            smooth_1d(nu_true[:, i], max(1, int(round(LF_WINDOW_SECONDS / dt)) | 1))
            for i in range(3)
        ])
        vel_error = nu_est - nu_lf
        result["RMS EKF surge error vs LF [m/s]"] = rms(vel_error[:, 0])
        result["RMS EKF sway error vs LF [m/s]"] = rms(vel_error[:, 1])
        result["RMS EKF yaw-rate error vs LF [deg/s]"] = rms(np.rad2deg(vel_error[:, 2]))
    except AttributeError as err:
        nu_field = None
        print(f"Note: velocity metrics skipped. {err}")

    # A rate is treated as 'at its limit' within 0.01 of the prescribed value.
    # Fractions refer to simulation time steps, not continuous physical time.
    for i, label in enumerate(("bow", "azimuth 1", "azimuth 2")):
        rate = np.abs(thrust_rate[:, i])
        result[f"Max {label} thrust rate [kN/s]"] = float(np.max(rate))
        result[f"{label} time at thrust-rate limit [%]"] = float(
            100 * np.mean(rate >= THRUST_RATE_LIMITS_KNS[i] - 0.01)
        )
        result[f"{label} time at thrust limit [%]"] = float(
            100 * np.mean(np.abs(thrust_kn[:, i]) >= THRUST_LIMITS_KN[i] - 0.01)
        )
        if i > 0:
            rotation = np.abs(az_rate[:, i])
            result[f"Max {label} rotation rate [deg/s]"] = float(np.max(rotation))
            result[f"{label} time at rotation-rate limit [%]"] = float(
                100 * np.mean(rotation >= AZIMUTH_RATE_LIMIT_DEGS - 0.01)
            )

    return result, eta_field, nu_field


def print_table(title, left, right, left_name, right_name):
    width = 96
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'Metric':<57}{left_name:>18}{right_name:>18}")
    print("-" * width)
    for key in left:
        if key in right:
            print(f"{key:<57}{left[key]:>18.3f}{right[key]:>18.3f}")


def main():
    if runner.selected_observer() != "kalman":
        raise RuntimeError('Set SELECTED_OBSERVER = "kalman" in run_case_part_2.py first.')

    # Match the closed-loop EKF stage of Simulation 4 exactly.
    cfg4 = runner.Part2SimConfig(
        dt=runner.DT, T=EVAL_TIME, use_observer=True,
        observer_type=runner.selected_observer(),
    )
    # Match Simulation 6 exactly, except for the prescribed longer duration and sea.
    cfg6 = runner.Part2SimConfig(
        dt=runner.DT, T=1000.0, use_observer=True,
        observer_type=runner.selected_observer(),
    )
    required_flags = {
        "use_reference": True,
        "use_controller": True,
        "thruster_dynamics": True,
        "use_sensor_noise": False,
    }
    for flag, expected in required_flags.items():
        if getattr(cfg4, flag) != getattr(cfg6, flag):
            raise RuntimeError(f"Different configuration for {flag} in the two runs.")
        if getattr(cfg4, flag) != expected:
            raise RuntimeError(f"Expected {flag}={expected} for both simulations.")
    print("Observer: EKF (kalman), same gains in both runs")
    print("Reference/controller/thruster settings match across both runs.")
    print("Sensor noise setting:", getattr(cfg4, "use_sensor_noise", "not available"))

    print("\nRunning Simulation 4 closed-loop EKF (design sea, 600 s)...")
    logs4 = runner.run(cfg4, np.zeros(6), runner.environment())
    print("Running Simulation 6 closed-loop EKF (heavy sea, 1000 s)...")
    logs6 = runner.run(
        cfg6, np.zeros(6),
        runner.environment(hs=runner.ROBUST_HS, tp=runner.ROBUST_TP),
    )

    sim4, eta4, nu4 = metrics(logs4, EVAL_TIME)
    sim6, eta6, nu6 = metrics(logs6, EVAL_TIME)
    full6, _, _ = metrics(logs6, 1000.0)

    print(f"\nObserver logs: pose={eta4}/{eta6}; velocity={nu4}/{nu6}")
    print_table(
        "FAIR COMPARISON: 0-600 s, DESIGN SEA vs HEAVY SEA",
        sim4, sim6, "Sim 4 (0-600)", "Sim 6 (0-600)",
    )
    print_table(
        "SIMULATION 6: FIRST 600 s vs FULL 1000 s (NOT THE PRIMARY COMPARISON)",
        sim6, full6, "Sim 6 (0-600)", "Sim 6 (0-1000)",
    )
    print("\nLF truth: approximately 30 s centred moving average of true pose/velocity.")
    print("Mean thrust activity: mean sum of absolute actual thrust rates.")
    print("'Time at limit' = percent of sampled steps within 0.01 of the limit.")
    print("These are approximate limit-activity indicators, not exact commanded saturation times.")
    print("No CSV files or figures were generated by this script.")


if __name__ == "__main__":
    main()
