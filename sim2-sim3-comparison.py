
"""
compare_sim2_sim3.py

Calculates and compares results from Simulations 2 and 3.

Run from the project directory:
    python3.11 sim2-sim3-comparison.py
"""

import numpy as np
import matplotlib

# Prevent simulation figures from opening during calculations
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_case_part_2 as runner


def get_pose(data):
    """Extract North, East and heading from 3-DOF or 6-DOF data."""
    data = np.asarray(data, dtype=float)

    if data.shape[1] >= 6:
        return data[:, [0, 1, 5]]

    return data[:, [0, 1, 2]]


def rms(values):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values**2))


def calculate_metrics(logs):
    """Calculate tracking errors and thruster activity."""

    actual = get_pose(logs.eta)
    desired = get_pose(logs.sp)

    n = len(actual)

    # Time
    if hasattr(logs, "t"):
        t = np.asarray(logs.t, dtype=float)[:n]
    else:
        t = np.arange(n) * runner.DT

    dt = float(np.median(np.diff(t)))

    # Position error
    north_error = actual[:, 0] - desired[:, 0]
    east_error = actual[:, 1] - desired[:, 1]

    position_error = np.sqrt(
        north_error**2 + east_error**2
    )

    # Heading error, wrapped to +/-180 degrees
    heading_difference = actual[:, 2] - desired[:, 2]

    heading_error = np.rad2deg(
        np.arctan2(
            np.sin(heading_difference),
            np.cos(heading_difference)
        )
    )

    # Use the last 120 seconds of each 300-second leg.
    # This prevents large errors immediately after setpoint changes
    # from dominating the station-keeping results.
    hold = 300.0
    steady_mask = (t % hold) >= 180.0

    # Actual thrust in kN
    thrust = np.asarray(logs.u, dtype=float)[:n, :3] / 1000.0

    # Change in thrust per second
    thrust_rate = np.diff(thrust, axis=0) / dt

    # Thruster activity: sum of absolute thrust values
    total_abs_thrust = np.sum(np.abs(thrust), axis=1)

    # Total change in actual thrust per second
    thrust_activity = np.sum(
        np.abs(thrust_rate), axis=1
    )

    # Commanded and applied wrench
    commanded = get_pose(logs.tau_d)[:n]
    applied = get_pose(logs.tau_thr)[:n]

    residual = commanded - applied

    results = {
        "RMS position error [m]":
            rms(position_error),

        "Max position error [m]":
            np.max(position_error),

        "RMS heading error [deg]":
            rms(heading_error),

        "Max heading error [deg]":
            np.max(np.abs(heading_error)),

        "RMS position error, steady [m]":
            rms(position_error[steady_mask]),

        "Max position error, steady [m]":
            np.max(position_error[steady_mask]),

        "RMS heading error, steady [deg]":
            rms(heading_error[steady_mask]),

        "Max heading error, steady [deg]":
            np.max(np.abs(heading_error[steady_mask])),

        "Max bow thrust [kN]":
            np.max(np.abs(thrust[:, 0])),

        "Max azimuth 1 thrust [kN]":
            np.max(np.abs(thrust[:, 1])),

        "Max azimuth 2 thrust [kN]":
            np.max(np.abs(thrust[:, 2])),

        "Mean total absolute thrust [kN]":
            np.mean(total_abs_thrust),

        "Mean thrust activity [kN/s]":
            np.mean(thrust_activity),

        "Max bow thrust rate [kN/s]":
            np.max(np.abs(thrust_rate[:, 0])),

        "Max azimuth 1 thrust rate [kN/s]":
            np.max(np.abs(thrust_rate[:, 1])),

        "Max azimuth 2 thrust rate [kN/s]":
            np.max(np.abs(thrust_rate[:, 2])),

        "RMS Fx residual [kN]":
            rms(residual[:, 0]) / 1000.0,

        "RMS Fy residual [kN]":
            rms(residual[:, 1]) / 1000.0,

        "RMS Mz residual [kNm]":
            rms(residual[:, 2]) / 1000.0,
    }

    return results


def main():

    print("\nRunning Simulation 2...")
    logs2 = runner.sim2()
    plt.close("all")

    print("\nRunning Simulation 3...")
    logs3 = runner.sim3()
    plt.close("all")

    if logs2 is None or logs3 is None:
        raise RuntimeError(
            "Check that both sim2() and sim3() return logs."
        )


    # Check setpoints, reference and actual vessel position
    for name, logs in [("SIM 2", logs2), ("SIM 3", logs3)]:
        print(f"\n=== REFERENCE CHECK: {name} ===")

        if hasattr(logs, "t"):
            times = np.asarray(logs.t)
        else:
            times = np.arange(len(logs.eta)) * runner.DT

        for target_time in [0, 1, 299, 301, 599, 601]:
            k = np.argmin(np.abs(times - target_time))

            print(f"\nt = {times[k]:.1f} s")

            for field in ["cmd", "sp", "eta"]:
                if hasattr(logs, field):
                    pose = get_pose(getattr(logs, field))[k]
                    print(
                        f"{field:4s}: "
                        f"N={pose[0]:7.2f} m, "
                        f"E={pose[1]:7.2f} m, "
                        f"heading={np.rad2deg(pose[2]):7.2f} deg"
                    )


    sim2 = calculate_metrics(logs2)
    sim3 = calculate_metrics(logs3)

    # Print the results to the terminal
    print("\n" + "=" * 85)
    print("QUANTITATIVE COMPARISON: SIMULATION 2 VS SIMULATION 3")
    print("=" * 85)

    print(
        f"{'Metric':<43}"
        f"{'Sim 2':>15}"
        f"{'Sim 3':>15}"
    )

    print("-" * 85)

    for key in sim2:
        print(
            f"{key:<43}"
            f"{sim2[key]:>15.3f}"
            f"{sim3[key]:>15.3f}"
        )

    print("\n" + "=" * 85)
    print("Steady = last 120 seconds of each 300-second leg.")
    print("Thrust activity = mean sum of absolute thrust rates.")
    print("This is not the formal thrust-utilisation metric.")


if __name__ == "__main__":
    main()
