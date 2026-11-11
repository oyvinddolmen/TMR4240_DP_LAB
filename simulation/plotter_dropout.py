"""Report figures for the EKF measurement-dropout case in Simulation 7."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _mark_dropout(axes, dropout, *, label="No pose measurements"):
    for ax in np.asarray(axes).flat:
        ax.axvspan(*dropout, color="0.5", alpha=0.15,
                   label=label, zorder=0)
        for boundary in dropout:
            ax.axvline(boundary, color="0.4", ls="--", lw=0.8)
        ax.grid(True, alpha=0.3)


def plot_dropout_trajectory(logs, dropout, figsize=(8, 7)):
    """True trajectory by measurement availability, plus the EKF LF estimate."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    start, stop = np.searchsorted(logs.t, dropout)
    segments = (
        (slice(0, start + 1), "tab:blue", "True: before dropout"),
        (slice(start, stop + 1), "tab:red",
         f"True: no measurements ({dropout[0]:g}-{dropout[1]:g} s)"),
        (slice(stop, None), "tab:green", "True: measurements restored"),
    )
    for section, color, label in segments:
        ax.plot(logs.eta[section, 1], logs.eta[section, 0], color=color,
                lw=1.2, label=label)
    ax.plot(logs.eta_est[:, 1], logs.eta_est[:, 0], "--", color="0.25",
            lw=1.0, label="EKF LF estimate")
    ax.plot(logs.cmd[0, 1], logs.cmd[0, 0], "kx", ms=9, label="Setpoint")
    ax.plot(logs.eta[0, 1], logs.eta[0, 0], "o", color="tab:blue",
            ms=5, label="Start")
    ax.plot(logs.eta[-1, 1], logs.eta[-1, 0], "s", color="tab:green",
            ms=5, label=f"End ({logs.t[-1]:g} s)")
    for index, text in ((start, f"{dropout[0]:g} s"), (stop, f"{dropout[1]:g} s")):
        point = (logs.eta[index, 1], logs.eta[index, 0])
        ax.plot(*point, "o", color="black", ms=4)
        ax.annotate(text, point, xytext=(28, 14), textcoords="offset points",
                    fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color="0.25", lw=0.7))
    ax.set(xlabel="East [m]", ylabel="North [m]",
           title="Simulation 7: EKF station-keeping trajectory\n"
                 f"All pose measurements lost from {dropout[0]:g} to {dropout[1]:g} s; noise off")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_dropout_actuator_usage(logs, thrusters, dropout, figsize=(12, 8)):
    """Actual/commanded thrust as percent of rated thrust, and azimuth angle."""
    fig, axes = plt.subplots(len(thrusters), 2, figsize=figsize, sharex=True,
                             squeeze=False, layout="constrained")
    for row, thruster in enumerate(thrusters):
        commanded = 100.0 * np.abs(logs.u_cmd[:, row]) / thruster.u_max
        actual = 100.0 * np.abs(logs.u[:, row]) / thruster.u_max
        axes[row, 0].plot(logs.t, commanded, color="tab:orange", lw=0.9,
                          label="Commanded")
        axes[row, 0].plot(logs.t, actual, color="tab:blue", lw=1.0,
                          label="Actual")
        axes[row, 0].set_ylabel(f"{thruster.name}\nThrust usage [%]")
        axes[row, 0].set_ylim(0.0, max(10.0, 1.1 * max(actual.max(), commanded.max())))
        axes[row, 1].plot(logs.t, np.degrees(np.unwrap(logs.alpha_cmd[:, row])),
                          color="tab:orange", lw=0.9, label="Commanded")
        axes[row, 1].plot(logs.t, np.degrees(np.unwrap(logs.alpha[:, row])),
                          color="tab:blue", lw=1.0, label="Actual")
        axes[row, 1].set_ylabel("Azimuth [deg]\n(unwrapped)")
    _mark_dropout(axes, dropout)
    for ax in axes.flat:
        ax.set_xlim(logs.t[0], logs.t[-1])
    axes[0, 0].set_title("Thrust usage (100% = rated thrust)")
    axes[0, 1].set_title("Azimuth angle")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Simulation 7: actuator usage during measurement dropout", fontsize=13)
    return fig


def plot_dropout_pose_comparison(logs, dropout, figsize=(12, 8)):
    """Compare only the EKF LF estimate and the true pose including waves."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             layout="constrained")
    for ax, idx, label in zip(axes, (0, 1, 5),
                               ("North [m]", "East [m]", "Heading [deg]")):
        truth = logs.eta[:, idx]
        low_frequency = logs.eta_est[:, idx]
        if idx == 5:
            truth, low_frequency = (
                np.degrees(np.unwrap(values)) for values in (truth, low_frequency)
            )
        ax.plot(logs.t, truth, color="tab:blue", lw=1.0, label="True (with waves)")
        ax.plot(logs.t, low_frequency, "--", color="tab:orange", lw=1.2,
                label="EKF LF estimate")
        ax.set_ylabel(label)
        ax.set_xlim(logs.t[0], logs.t[-1])
    _mark_dropout(axes, dropout, label=None)
    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Simulation 7: true and estimated position and heading\n"
                 f"All pose measurements unavailable: {dropout[0]:g}-{dropout[1]:g} s (noise off)",
                 fontsize=13)
    return fig


def plot_dropout_bias(logs, dropout, figsize=(12, 8)):
    """Estimated NED environmental force and yaw-moment bias during dropout."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             layout="constrained")
    for ax, idx, label in zip(axes, (0, 1, 5),
                               ("North bias [kN]", "East bias [kN]", "Yaw bias [kNm]")):
        ax.plot(logs.t, logs.bias_est[:, idx] * 1e-3, color="tab:blue", lw=1.0)
        ax.set_ylabel(label)
        ax.set_xlim(logs.t[0], logs.t[-1])
    _mark_dropout(axes, dropout, label=None)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Simulation 7: estimated environmental bias (EKF)\n"
                 f"All pose measurements unavailable: {dropout[0]:g}-{dropout[1]:g} s; "
                 "measurement noise off", fontsize=13)
    return fig


def save_sim7_dropout_plots(logs, thrusters, output_dir, *, dropout):
    """Save the three report PDFs and keep figures open for interactive use."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"pdf.fonttype": 42}):
        figures = (
            ("sim7_trajectory.pdf", plot_dropout_trajectory(logs, dropout)),
            ("sim7_actuator_usage.pdf", plot_dropout_actuator_usage(logs, thrusters, dropout)),
            ("sim7_position_estimates.pdf", plot_dropout_pose_comparison(logs, dropout)),
        )
        for filename, figure in figures:
            path = output_dir / filename
            figure.savefig(path, format="pdf", bbox_inches="tight")
            paths.append(path)
    return paths
