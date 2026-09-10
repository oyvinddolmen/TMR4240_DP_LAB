"""Plots specific to Part 2: observer, wind field, waves, thruster set-points."""
import matplotlib.pyplot as plt
import numpy as np


def _moving_average(x, dt, window=30.0):
    """Centred moving average with edge padding (no phase shift, same length)."""
    k = max(int(round(window / dt)), 1)
    xp = np.pad(np.asarray(x, float), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, "valid")


def plot_observer(logs, figsize=(11, 7)):
    """Compare true, measured, and estimated horizontal states.

    Velocities are NOT measured on the vessel: the "true + noise" traces in
    the right column are validation signals only (``models/sensors.py``), the
    observer never receives them.  See :func:`plot_observer_error` for the
    estimation errors, which are usually more informative than this overlay.
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    pose = ((0, "North [m]", 1.0), (1, "East [m]", 1.0),
            (5, "Heading [deg]", 180.0 / np.pi))
    vel = ((0, "u [m/s]", 1.0), (1, "v [m/s]", 1.0),
           (5, "r [deg/s]", 180.0 / np.pi))
    for row, (idx, label, scale) in enumerate(pose):
        axes[row, 0].plot(logs.t, scale * logs.eta[:, idx], label="true")
        axes[row, 0].plot(logs.t, scale * logs.eta_measured[:, idx], ".", ms=2, label="measured")
        axes[row, 0].plot(logs.t, scale * logs.eta_est[:, idx], "--", label="estimated")
        axes[row, 0].set_ylabel(label)
    for row, (idx, label, scale) in enumerate(vel):
        axes[row, 1].plot(logs.t, scale * logs.nu[:, idx], label="true")
        axes[row, 1].plot(logs.t, scale * logs.nu_measured[:, idx], ".", ms=2,
                          label="true + noise (validation only)")
        axes[row, 1].plot(logs.t, scale * logs.nu_est[:, idx], "--", label="estimated")
        axes[row, 1].set_ylabel(label)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle(f"Observer comparison: {logs.observer_name}")
    fig.tight_layout()
    return fig


def plot_observer_error(logs, figsize=(11, 7), window=30.0):
    """Estimation errors: estimate minus truth, and estimate minus the
    low-frequency truth (``window``-second moving average of the true state).

    Left column: North, East, heading; right column: u, v, r.  The
    "vs. truth" error contains the wave-frequency motion the observer is
    meant to remove, the "vs. LF truth" error shows how well the
    low-frequency state — what the controller needs — is reconstructed.
    """
    dt = float(logs.t[1] - logs.t[0])
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    rows = ((0, "North error [m]", 1.0), (1, "East error [m]", 1.0),
            (5, "Heading error [deg]", 180.0 / np.pi))
    for row, (idx, label, scale) in enumerate(rows):
        true = np.unwrap(logs.eta[:, idx]) if idx == 5 else logs.eta[:, idx]
        est = np.unwrap(logs.eta_est[:, idx]) if idx == 5 else logs.eta_est[:, idx]
        e_tot = est - true
        e_lf = est - _moving_average(true, dt, window)
        if idx == 5:
            e_tot, e_lf = _wrap_pi(e_tot), _wrap_pi(e_lf)
        axes[row, 0].plot(logs.t, scale * e_tot, lw=0.6, alpha=0.6, label="estimate - truth")
        axes[row, 0].plot(logs.t, scale * e_lf, label=f"estimate - LF truth ({window:.0f} s mean)")
        axes[row, 0].set_ylabel(label)
    vel = ((0, "u error [m/s]", 1.0), (1, "v error [m/s]", 1.0), (5, "r error [deg/s]", 180.0 / np.pi))
    for row, (idx, label, scale) in enumerate(vel):
        e_tot = logs.nu_est[:, idx] - logs.nu[:, idx]
        e_lf = logs.nu_est[:, idx] - _moving_average(logs.nu[:, idx], dt, window)
        axes[row, 1].plot(logs.t, scale * e_tot, lw=0.6, alpha=0.6, label="estimate - truth")
        axes[row, 1].plot(logs.t, scale * e_lf, label="estimate - LF truth")
        axes[row, 1].set_ylabel(label)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle(f"Observer estimation errors: {logs.observer_name}")
    fig.tight_layout()
    return fig



def _wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def plot_wind_field(logs, figsize=(10, 7)):
    """Part 2 wind field: total, mean+slow and gust speed; direction; loads."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes[0].plot(logs.t, logs.U_w, label="total U")
    axes[0].plot(logs.t, logs.U_w_mean, "--", label="mean + slow (U_mean)")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(logs.t, logs.U_w_gust)
    axes[1].set_ylabel("Gust [m/s]")
    # unwrapped so that a direction near +-180 deg does not jump between the two
    axes[2].plot(logs.t, np.degrees(np.unwrap(logs.beta_w)), label="NED (towards)")
    axes[2].plot(logs.t, np.degrees(np.unwrap(logs.alpha_w)), "--", label="relative (BODY)")
    axes[2].set_ylabel("Direction [deg]\n(unwrapped)")
    axes[2].legend(loc="best", fontsize=8)
    axes[3].plot(logs.t, logs.tau_w6[:, 0] * 1e-3, label="Fx [kN]")
    axes[3].plot(logs.t, logs.tau_w6[:, 1] * 1e-3, label="Fy [kN]")
    axes[3].plot(logs.t, logs.tau_w6[:, 5] * 1e-3, label="Mz [kNm]")
    axes[3].set_ylabel("BODY loads")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Wind field (Part 2)")
    fig.tight_layout()
    return fig


def plot_waves(logs, figsize=(10, 6)):
    """Wave loads on the hull (Part 2): first+second order BODY wrench."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    rows = [("Fx [kN]", 0, 1e-3), ("Fy [kN]", 1, 1e-3), ("Mz [kNm]", 5, 1e-3)]
    for ax, (label, idx, scale) in zip(axes, rows):
        ax.plot(logs.t, logs.tau_wave[:, idx] * scale)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Wave loads (BODY)")
    fig.tight_layout()
    return fig


def plot_thruster_setpoints(logs, figsize=(11, 7)):
    """Per-thruster set-point vs. actual (Part 2, Simulation 2).

    Left column: commanded thrust (allocator) against the thrust the
    rate-limited, saturated actuator actually delivered.  Right column: the
    same for the azimuth angle.  One row per thruster.
    """
    n = len(logs.thruster_names)
    fig, axes = plt.subplots(n, 2, figsize=figsize, sharex=True, squeeze=False)
    for i, name in enumerate(logs.thruster_names):
        ax = axes[i, 0]
        ax.plot(logs.t, logs.u_cmd[:, i] * 1e-3, label="set-point (allocator)")
        ax.plot(logs.t, logs.u[:, i] * 1e-3, "--", label="actual (thruster dynamics)")
        ax.set_ylabel(f"{name}\nthrust [kN]")
        ax = axes[i, 1]
        # unwrapped: an azimuth resting near +-180 deg would otherwise appear to
        # swing by 360 deg every time the wrapped angle changes sign
        ax.plot(logs.t, np.degrees(np.unwrap(logs.alpha_cmd[:, i])), label="set-point")
        ax.plot(logs.t, np.degrees(np.unwrap(logs.alpha[:, i])), "--", label="actual")
        ax.set_ylabel("azimuth [deg]\n(unwrapped)")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Thruster set-points vs. actual")
    fig.tight_layout()
    return fig


def plot_wrench_residual(logs, figsize=(11, 6)):
    """Desired-minus-applied BODY wrench for actuator-limit assessment.

    The residual is computed from logged signals rather than stored as a
    separate simulator state: ``r_tau = tau_d - tau_thr``.  This makes the
    transient effect of thrust and azimuth slew, and any persistent
    infeasibility, directly visible in Part 2, Simulations 2 and 3.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    residual = logs.tau_d - logs.tau_thr
    rows = (("Fx residual [kN]", 0), ("Fy residual [kN]", 1),
            ("Mz residual [kNm]", 5))
    for ax, (label, idx) in zip(axes, rows):
        ax.plot(logs.t, residual[:, idx] * 1e-3)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(r"Actuator wrench residual $r_\tau=\tau_d-\tau_{thr}$")
    fig.tight_layout()
    return fig
