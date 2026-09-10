"""Thrust utilisation and capability plots for Project Part 2 (Simulation 5).

Definitions (from the project description):

* **Thrust utilisation** at time t is the sum of the magnitudes of the
  individual thruster forces, as a percentage of the sum of the maximum
  nominal thrusts of all active thrusters,

      u_t(t) = 100 * sum_i |u_i(t)| / sum_i u_max,i .

* **Average thrust utilisation** is the mean of u_t(t) over the evaluation
  window (by default the second half of the run, so transients are excluded).

* Three polar plots: the utilisation for **all** directions; the **DP
  watch-circle** view, which hides directions whose maximum **low-frequency**
  excursion (a 30 s centred moving average — the station-keeping motion a
  DP system checks on its screen) exceeds the watch-circle limits (5 m /
  5 deg); and the **operability** view, which hides directions whose maximum
  **total** motion (raw trajectory including the wave-frequency oscillation —
  what a connected gangway experiences) exceeds the operability limits (3 m /
  3 deg).  The sweep computes and reports both deviation measures per
  direction; ``deviation`` selects which one drives ``feasible`` (the
  watch-circle mask by default) and ``plot_capability(total_limits=...)``
  adds the operability view.

The sweep itself is generic: you provide ``run(direction_rad)`` — a function
that builds the environment (wind, waves, current — all from the same
direction), runs your simulator and returns the logs.  The sweep bookkeeping,
the deviation measures and the polar plot are provided; **the thrust-utilisation
metrics are yours to implement** in ``part_2/utilization.py``
(``thrust_utilization``, ``average_utilization`` carry the definitions as
TODOs) — Part 2, Simulation 5 raises ``NotImplementedError`` until you do.

Typical use (see ``run_case_part_2.py``, Simulation 5)::

    from simulation.capability import capability_sweep, plot_capability

    def run(direction):
        ...build Wind/Waves/Current from `direction`, run sim, return logs

    result = capability_sweep(run, thrusters, step_deg=10.0,
                              pos_limit=5.0, psi_limit_deg=5.0)
    plot_capability(result)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

# The two thrust-utilisation metrics are student work (Part 2, Simulation 5)
# and live with the other Part 2 blocks; they are re-exported here so that
# ``capability_sweep`` and existing imports keep working.
from part_2.utilization import average_utilization, thrust_utilization  # noqa: F401


def _wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _moving_average(x: np.ndarray, dt: float, window: float) -> np.ndarray:
    """Centred moving average with edge padding (no phase shift, same length).

    The first and last ``window/2`` seconds are computed from edge-padded
    data and are biased toward the endpoint samples; use
    :func:`_interior_selection` to exclude them when taking extrema."""
    k = max(int(round(window / dt)), 1)
    xp = np.pad(np.asarray(x, float), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, "valid")


def _interior_selection(sel: np.ndarray, dt: float, window: float) -> np.ndarray:
    """Restrict a boolean selection to samples whose averaging window is fully
    supported by data (half a window trimmed at both ends).  Falls back to the
    original selection if the trim would leave nothing (very short runs)."""
    k = max(int(round(window / dt)), 1)
    n = len(sel)
    interior = np.zeros(n, dtype=bool)
    lo, hi = k // 2, n - (k - 1 - k // 2)
    if hi > lo:
        interior[lo:hi] = True
    out = sel & interior
    return out if np.any(out) else sel


def low_frequency_motion(logs, window: float = 30.0) -> np.ndarray:
    """(n, 3) low-frequency North, East and (unwrapped) heading: the logged
    trajectory smoothed with a ``window``-second centred moving average."""
    dt = float(logs.t[1] - logs.t[0])
    return np.column_stack([_moving_average(logs.eta[:, 0], dt, window),
                            _moving_average(logs.eta[:, 1], dt, window),
                            _moving_average(np.unwrap(logs.eta[:, 5]), dt, window)])


def max_deviation(logs, t_from: float | None = None, *, deviation: str = "total",
                  window: float = 30.0) -> tuple[float, float]:
    """(max position deviation [m], max heading deviation [rad]) from the setpoint
    over ``t >= t_from`` (default: second half).

    ``deviation="total"`` (default) evaluates the raw logged motion including
    the wave-frequency oscillation — the vessel motion an operation such as a
    gangway transfer experiences.  ``"lf"`` evaluates the low-frequency motion
    (see :func:`low_frequency_motion`) — DP station-keeping performance.  For
    ``"lf"`` the first and last half filter window are excluded from the
    maximum, because the edge-padded moving average is biased there (its value
    would depend on the wave phase at the ends of the run)."""
    if deviation not in ("lf", "total"):
        raise ValueError("deviation must be 'lf' or 'total'")
    t_from = 0.5 * logs.t[-1] if t_from is None else t_from
    sel = logs.t >= t_from
    if deviation == "lf":
        lf = low_frequency_motion(logs, window)
        N, E, psi_u = lf[:, 0], lf[:, 1], lf[:, 2]
        sel = _interior_selection(sel, float(logs.t[1] - logs.t[0]), window)
    else:
        N, E, psi_u = logs.eta[:, 0], logs.eta[:, 1], logs.eta[:, 5]
    pos = np.hypot(N[sel] - logs.sp[sel, 0], E[sel] - logs.sp[sel, 1])
    psi = np.abs(_wrap_pi(psi_u[sel] - logs.sp[sel, 5]))
    return float(np.max(pos)), float(np.max(psi))


@dataclass
class CapabilityResult:
    directions_deg: np.ndarray
    utilization: np.ndarray            # average utilisation [%] per direction
    max_pos_dev: np.ndarray            # [m], of the motion selected by `deviation`
    max_psi_dev: np.ndarray            # [rad], of the motion selected by `deviation`
    pos_limit: float
    psi_limit: float                   # [rad]
    max_pos_dev_total: np.ndarray = None   # [m]  raw motion (operational/gangway measure)
    max_psi_dev_total: np.ndarray = None    # [rad]
    max_pos_dev_lf: np.ndarray = None       # [m]  low-frequency motion (DP performance)
    max_psi_dev_lf: np.ndarray = None       # [rad]
    logs: dict = field(default_factory=dict)   # direction_deg -> logs (optional)
    deviation: str = "lf"              # "lf" or "total": motion the limits were applied to

    @property
    def feasible(self) -> np.ndarray:
        """Directions where the operational limits were respected (on the
        motion selected by ``deviation``)."""
        return (self.max_pos_dev <= self.pos_limit) & (self.max_psi_dev <= self.psi_limit)

    @property
    def feasible_total(self) -> np.ndarray:
        """``pos_limit``/``psi_limit`` applied to the TOTAL (raw) motion.

        Note which limits these are: the ones stored on this result, i.e. the
        watch circles the sweep was run with.  This is NOT the operability
        view — that one applies the smaller operability limits to the total
        motion, and ``plot_capability`` computes it from its own
        ``total_limits`` argument.  If you want the operability mask, compare
        ``max_pos_dev_total``/``max_psi_dev_total`` against those limits
        yourself.
        """
        return ((self.max_pos_dev_total <= self.pos_limit)
                & (self.max_psi_dev_total <= self.psi_limit))

    @property
    def feasible_lf(self) -> np.ndarray:
        """``pos_limit``/``psi_limit`` applied to the low-frequency motion.

        With the sweep's default ``deviation="lf"`` this is the same mask as
        ``feasible``; it is kept separate so both motions can be compared
        against the same limits after a ``deviation="total"`` sweep.
        """
        return ((self.max_pos_dev_lf <= self.pos_limit)
                & (self.max_psi_dev_lf <= self.psi_limit))


def capability_sweep(run, thrusters, *, step_deg: float = 10.0,
                     pos_limit: float = 5.0, psi_limit_deg: float = 5.0,
                     t_from: float | None = None, keep_logs: bool = False,
                     verbose: bool = True, deviation: str = "lf",
                     window: float = 30.0) -> CapabilityResult:
    """Run ``run(direction_rad)`` for every environmental direction and collect
    the average thrust utilisation and the maximum deviations.

    ``direction`` is the NED direction the environment comes FROM, in radians,
    0 = North, increasing clockwise; the sweep covers [0, 360) in ``step_deg``
    increments.  Both deviation measures are computed and stored per
    direction — low-frequency (DP station-keeping, the watch-circle measure
    and default mask) and total (raw motion, the operational/gangway
    measure); ``deviation`` selects the one the mask (``feasible``) is
    applied to, see :func:`max_deviation`.  ``window`` is the moving-average
    length [s].
    """
    dirs = np.arange(0.0, 360.0, step_deg)
    util = np.empty(len(dirs))
    dpos_tot = np.empty(len(dirs))
    dpsi_tot = np.empty(len(dirs))
    dpos_lf = np.empty(len(dirs))
    dpsi_lf = np.empty(len(dirs))
    logs_kept = {}
    for i, d in enumerate(dirs):
        logs = run(np.deg2rad(d))
        util[i] = average_utilization(logs, thrusters, t_from)
        dpos_tot[i], dpsi_tot[i] = max_deviation(logs, t_from, deviation="total")
        dpos_lf[i], dpsi_lf[i] = max_deviation(logs, t_from, deviation="lf", window=window)
        if keep_logs:
            logs_kept[float(d)] = logs
        dpos_i, dpsi_i = (dpos_tot[i], dpsi_tot[i]) if deviation == "total" else (dpos_lf[i], dpsi_lf[i])
        if verbose:
            reasons = [lab for ok, lab in ((dpos_i <= pos_limit, "position"),
                                           (dpsi_i <= np.deg2rad(psi_limit_deg), "heading")) if not ok]
            verdict = "ok" if not reasons else "masked (" + ", ".join(reasons) + ")"
            print(f"direction {d:5.1f} deg: utilisation {util[i]:5.1f} %, "
                  f"max total dev {dpos_tot[i]:.2f} m / {np.rad2deg(dpsi_tot[i]):.2f} deg, "
                  f"max LF dev {dpos_lf[i]:.2f} m / {np.rad2deg(dpsi_lf[i]):.2f} deg -> {verdict} ({deviation})")
    dpos, dpsi = (dpos_tot, dpsi_tot) if deviation == "total" else (dpos_lf, dpsi_lf)
    return CapabilityResult(dirs, util, dpos, dpsi, pos_limit, np.deg2rad(psi_limit_deg),
                            max_pos_dev_total=dpos_tot, max_psi_dev_total=dpsi_tot,
                            max_pos_dev_lf=dpos_lf, max_psi_dev_lf=dpsi_lf,
                            logs=logs_kept, deviation=deviation)


def plot_capability(result: CapabilityResult, figsize=None, total_limits=None, rmax=None):
    """Polar plots of the average thrust utilisation: all directions, the
    directions inside the operational limits of the sweep (``result.feasible``,
    by default the DP watch circles on the low-frequency motion), and — if
    ``total_limits=(pos_m, psi_deg)`` is given — the directions inside these
    limits on the TOTAL motion (the operability view: a gangway or crane
    operation experiences the wave-frequency motion too).  Masked directions
    are not drawn.  ``rmax`` fixes the radial axis (default: 100 % or the data)."""
    n_ax = 3 if total_limits is not None else 2
    fig, axes = plt.subplots(1, n_ax, figsize=figsize or (5.5 * n_ax, 5),
                             subplot_kw={"projection": "polar"})
    theta = np.deg2rad(np.append(result.directions_deg, result.directions_deg[0]))
    util = np.append(result.utilization, result.utilization[0])
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, rmax if rmax is not None else max(100.0, 1.05 * np.nanmax(util)))
        ax.set_rlabel_position(22.5)
        ax.grid(True, alpha=0.4)
    axes[0].plot(theta, util, "-o", ms=3)
    axes[0].set_title("Average thrust utilisation [%]", pad=15)
    masked = np.where(np.append(result.feasible, result.feasible[0]), util, np.nan)
    axes[1].plot(theta, masked, "-o", ms=3)
    axes[1].set_title(f"Inside {result.pos_limit:.0f} m / {np.rad2deg(result.psi_limit):.0f} deg "
                      f"on the {'low-frequency' if result.deviation == 'lf' else 'total'} motion", pad=15)
    if total_limits is not None:
        pos_lim, psi_lim_deg = total_limits
        ok = ((result.max_pos_dev_total <= pos_lim)
              & (result.max_psi_dev_total <= np.deg2rad(psi_lim_deg)))
        masked_t = np.where(np.append(ok, ok[0]), util, np.nan)
        axes[2].plot(theta, masked_t, "-o", ms=3, color="C2")
        axes[2].set_title(f"Inside {pos_lim:.0f} m / {psi_lim_deg:.0f} deg on the total motion",
                          pad=15)
    fig.tight_layout()
    return fig
