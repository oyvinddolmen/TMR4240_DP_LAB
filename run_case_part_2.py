"""
Part 2 scenario runner — the file you edit and run.

    python run_case_part_2.py                 # runs SCENARIO below
    python run_case_part_2.py sim3            # runs one named scenario
    python run_case_part_2.py --list          # lists the scenarios

Every mandatory simulation of the Part 2 description has a preset here
(``sim1`` … ``sim7``), plus ``bonus`` for the optional extra-credit sensor-noise
task; all regular simulations run with sensor noise OFF.  Each preset builds
the configuration, the environment and the set-point sequence, runs
``DPSimulatorPart2`` and plots.  Edit the
presets freely: the environmental numbers are collected at the top, the
per-simulation switches live in each function.  All Part 2 blocks are
imported from ``part_2/`` — the checks (``python check.py --part 2``) build
them with the constructor defaults documented in those templates, so put
your final tuning there, not only in this file.

``SELECTED_OBSERVER`` starts as ``None``: compare both observers in Part 2,
Simulation 4, then set it to the one you chose and justify the choice in the
report. Part 2, Simulations 5-6 use it explicitly and stop with a message
until you have set it. Simulation 4 always compares both observers and
exports six paired report figures to output/pdf/sim4/.
Simulation 7 uses EKF explicitly and exports its measurement-dropout case
to output/pdf/sim7/.
The bonus sensor-noise scenario exports four paired figures to output/pdf/sensor_noise/
and the actuator wrench residual comparison to outputs/sensor_noise/.
A combined noisy raw-feedback versus EKF comparison is saved to output/pdf/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from part_2.config import Part2SimConfig, default_thrusters_part_2
from part_2.controller import DPController
from part_2.current import Current
from part_2.observer import select_observer
from simulation.plotter_part_2 import (plot_heading_decomposition,
                                       plot_observer, plot_observer_error,
                                       plot_thruster_setpoints, plot_waves,
                                       plot_wind_field, plot_wrench_residual,
                                       save_raw_feedback_ekf_comparison,
                                       save_sensor_noise_comparison_plots,
                                       save_sim4_comparison_plots)
from simulation.plotter_dropout import save_sim7_dropout_plots
from part_2.reference import ReferenceModel
from part_2.thrust_allocation import ThrustAllocator
from models.waves import Waves
from part_2.wind import Wind
from simulation.capability import capability_sweep, plot_capability
from simulation.plotter import (plot_current, plot_dashboard, plot_time_histories,
                                plot_wrench, plot_xy)
from simulation.simulation_part_2 import DPSimulatorPart2

# Some presets deliberately open more figures than matplotlib's default
# 20-figure warning threshold: Part 2, Simulation 4 alone plots both observers
# in four conditions.  Raise the threshold so a harmless warning does not read
# as a fault.
plt.rcParams["figure.max_open_warning"] = 60

# ---------------------------------------------------------------------------
# Environmental conditions prescribed for the supplied Gunnerus vessel model
# and its three-thruster configuration.
# ---------------------------------------------------------------------------
DT = 0.1
CURRENT_SPEED, CURRENT_FROM = 0.2, np.deg2rad(90.0)       # 0.2 m/s from east
WIND_SPEED, WIND_FROM = 10.0, np.deg2rad(0.0)             # 10 m/s from north
WAVES_HS, WAVES_TP, WAVES_FROM = 1.5, 8.0, np.deg2rad(45.0)   # design sea, from north-east
ROBUST_HS, ROBUST_TP = 2.5, 10.0                          # Simulation 6 (heavy sea for a vessel of this size, Lpp 34 m)
CAP_WIND, CAP_CURRENT, CAP_HS, CAP_TP = 12.0, 0.2, 1.5, 8.0   # Simulation 5
CAP_POS_LIMIT, CAP_PSI_LIMIT_DEG = 5.0, 5.0               # DP watch circles (max LF excursion)
CAP_TOTAL_POS_LIMIT, CAP_TOTAL_PSI_LIMIT_DEG = 3.0, 3.0   # operability limits (max total motion)
SEED = 123

# TODO (students): the observer used by Part 2, Simulations 5-6 is YOUR choice, made on
# the evidence you produce in Part 2, Simulation 4.  Compare both observers
# there first, then set this to "nonlinear_passive" or "kalman" and justify
# the choice in the report.  It is None until you choose: every preset that
# needs it stops with a message instead of picking one for you.
SELECTED_OBSERVER = "kalman"    # nonlinear_passive or kalman

# Initial observer error used by the convergence demonstration in Part 2,
# Simulation 4.  The engine resets every observer at the TRUE initial pose, so
# an estimate that starts on top of the truth has nothing to converge from.
# The value below is an EXAMPLE (5 m north, 5 m west, 10 deg of heading):
# choose your own, and say in the report what it is and why it is a fair test.
OBSERVER_OFFSET = np.array([5.0, -5.0, 0.0, 0.0, 0.0, np.deg2rad(10.0)])

# Absolute so the runner and notebooks always write to the same folders.
SIM4_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf" / "sim4"
SIM7_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf" / "sim7"
SENSOR_NOISE_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf" / "sensor_noise"
SENSOR_NOISE_RESIDUAL_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "sensor_noise"
SENSOR_NOISE_RAW_EKF_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf"

SCENARIO = "sim2"          # default scenario when run without arguments


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def selected_observer() -> str:
    """The observer you chose in Part 2, Simulation 4 (``SELECTED_OBSERVER``).

    Raises ``NotImplementedError`` while the choice is still unmade, so a
    preset never silently runs with an observer you did not pick.
    """
    if SELECTED_OBSERVER is None:
        raise NotImplementedError(
            "set SELECTED_OBSERVER at the top of run_case_part_2.py to the "
            "observer you selected in Part 2, Simulation 4 — "
            "\"nonlinear_passive\" or \"kalman\" — and justify the choice in "
            "the report; run Part 2, Simulation 4 first if you have not yet")
    return SELECTED_OBSERVER


def environment(*, current=True, wind=True, waves=True, gust=True,
                hs=WAVES_HS, tp=WAVES_TP, seed=SEED):
    """Standard Part 2 environment; switch components off with False."""
    cur = Current(CURRENT_SPEED, CURRENT_FROM, semantics="from") if current else Current()
    wnd = (Wind(WIND_SPEED, WIND_FROM, semantics="from", sigma_slow=0.5, gust=gust,
                sigma_dir=np.deg2rad(2.0), dir_limit=np.deg2rad(5.0), seed=seed)
           if wind else Wind())
    wav = Waves(hs=hs, tp=tp, direction=WAVES_FROM, seed=seed) if waves else None
    return cur, wnd, wav


def four_corner_commands(cfg: Part2SimConfig, hold: float):
    """Set-point sequence of the four-corner test, ``hold`` seconds per leg."""
    corners = [[50.0, 0.0, 0.0], [50.0, -50.0, 0.0], [50.0, -50.0, -np.pi / 4],
               [0.0, -50.0, -np.pi / 4], [0.0, 0.0, 0.0]]
    n_steps = int(round(cfg.T / cfg.dt)) + 1
    per_leg = int(round(hold / cfg.dt))
    eta_cmd = np.zeros((n_steps, 6))
    for i, c in enumerate(corners):
        rows = slice(i * per_leg, n_steps if i == len(corners) - 1 else (i + 1) * per_leg)
        eta_cmd[rows, 0], eta_cmd[rows, 1], eta_cmd[rows, 5] = c
    return eta_cmd


def make_sim(cfg: Part2SimConfig, controller=None):
    thrusters = default_thrusters_part_2()
    return DPSimulatorPart2(
        cfg, controller or DPController(), thrusters,
        reference=ReferenceModel(dt=cfg.dt),
        allocator=ThrustAllocator(thrusters),
        observer=select_observer(cfg.observer_type, **cfg.observer_kwargs),
    )


class WrenchSequence:
    """Stand-in controller: a piecewise-constant desired wrench.

    ``segments`` is a list of ``(duration_s, [Fx, Fy, Mz])``; the last segment
    holds to the end of the run. Used by Simulation 4 for its fixed-wrench
    observer test.
    """

    def __init__(self, segments):
        self.segments = [(float(d), np.asarray(w, dtype=float).reshape(3)) for d, w in segments]

    def compute(self, t, dt, eta, nu, eta_ref, nu_ref, acc_ref):
        t_seg = 0.0
        wrench = self.segments[-1][1]
        for duration, w in self.segments:
            if t < t_seg + duration:
                wrench = w
                break
            t_seg += duration
        tau = np.zeros(6)
        tau[0], tau[1], tau[5] = wrench
        return tau


def run(cfg, eta_cmd, env, controller=None):
    current, wind, waves = env
    sim = make_sim(cfg, controller)
    sim.reset_state()
    return sim.run(eta_cmd, current=current, wind=wind, waves=waves)


def standard_plots(logs, *, setpoints=False, observer=False, environment_plots=False,
                   save_thrust_pdf=False):
    """Show the standard figures and return them by name for optional export."""
    figures = {
        "dashboard": plot_dashboard(logs, save_thrust_pdf=save_thrust_pdf),
        "time_histories": plot_time_histories(logs),
        "trajectory": plot_xy(logs),
    }
    if setpoints:
        figures["wrench"] = plot_wrench(logs)
        figures["wrench_residual"] = plot_wrench_residual(logs)
        figures["thruster_setpoints"] = plot_thruster_setpoints(logs)
    if observer:
        figures["observer"] = plot_observer(logs)
        figures["observer_errors"] = plot_observer_error(logs)
    if environment_plots:
        figures["wind_field"] = plot_wind_field(logs)
        figures["current"] = plot_current(logs)
        figures["waves"] = plot_waves(logs)
    return figures


# ---------------------------------------------------------------------------
# Mandatory simulations
# ---------------------------------------------------------------------------

def sim1():
    """Part 2, Simulation 1 — environmental loads: free drift, DP off, 300 s; plots
    the wind field (total, mean+slow and gust speed, direction), current and
    wave loads you built."""
    cfg = Part2SimConfig(dt=DT, T=300.0, use_controller=False, use_reference=False)
    logs = run(cfg, np.zeros(6), environment())
    plot_time_histories(logs)
    plot_xy(logs)
    plot_wind_field(logs)
    plot_current(logs)
    plot_waves(logs)
    return logs


def sim2():
    """Part 2, Simulation 2 — four-corner DP test with constrained Part 2
    thruster dynamics enabled, and with environment and observer disabled."""
    hold = 300.0
    cfg = Part2SimConfig(dt=DT, T=5 * hold, use_reference=True,
                         use_observer=False, thruster_dynamics=True)
    logs = run(cfg, four_corner_commands(cfg, hold),
               environment(current=False, wind=False, waves=False))
    standard_plots(logs, setpoints=True)
    return logs


def sim3():
    """Part 2, Simulation 3 — the prescribed four-corner DP test in the Part 2,
    Simulation 1 environment, with raw measurements fed back (no observer).

    This deliberately keeps the reference model and constrained thrusters on:
    the purpose is to validate the integrated DP system under current, detailed
    wind and waves before observers are introduced in Simulation 4.
    """
    hold = 300.0
    cfg = Part2SimConfig(dt=DT, T=5 * hold, use_observer=False)
    logs = run(cfg, four_corner_commands(cfg, hold), environment())
    standard_plots(logs, setpoints=True, environment_plots=True)
    return logs


def sim4():
    """Part 2, Simulation 4 — observer selection, in three parts: (a) fixed desired
    wrench [2 2 2]e3 in the Part 2, Simulation 1 environment, with and without waves,
    for both observers; (b) the same run started with OBSERVER_OFFSET so the estimate
    has something to converge from; (c) closed-loop station keeping with raw
    measurements and each observer. Six paired report PDFs are saved to
    SIM4_OUTPUT_DIR on every call, with NPO left and EKF right."""
    out = {}
    fixed = WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
    variants = (("with waves", dict(waves=True)), ("without waves", dict(waves=False)))
    for kind in ("nonlinear_passive", "kalman"):
        for label, env_kw in variants:
            cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True,
                                 observer_type=kind)
            logs = run(cfg, np.zeros(6), environment(**env_kw), controller=fixed)
            plot_observer(logs).suptitle(f"Observer {kind}, {label}")
            plot_observer_error(logs).suptitle(f"Observer {kind}, {label} — estimation errors")
            plot_heading_decomposition(logs)
            out[(kind, label)] = logs
    # Convergence: start the estimate away from the truth (the engine resets it
    # ON the truth), then watch the error decay.  Quote the convergence time.
    for kind in ("nonlinear_passive", "kalman"):
        cfg = Part2SimConfig(dt=DT, T=300.0, use_reference=False, use_observer=True,
                             observer_type=kind)
        sim = make_sim(cfg, fixed)
        sim.reset_state()
        sim.observer.reset(OBSERVER_OFFSET)
        current, wind, waves = environment()
        logs = sim.run(np.zeros(6), current=current, wind=wind, waves=waves)
        plot_observer(logs).suptitle(f"Observer {kind}, offset start — convergence")
        plot_observer_error(logs).suptitle(
            f"Observer {kind}, offset start — estimation errors")
        out[(kind, "offset start")] = logs
    # Run both actual observers, independently of the selection for sim5-6.
    # Keep the existing output keys used by notebooks ("nonlinear", "kalman").
    for label, use_obs, kind in (("raw measurements", False, "kalman"),
                                 ("nonlinear", True, "nonlinear_passive"),
                                 ("kalman", True, "kalman")):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=use_obs,
                             observer_type=kind)
        logs = run(cfg, np.zeros(6), environment())
        standard_plots(logs, setpoints=True, observer=use_obs,
                       save_thrust_pdf=not use_obs)
        plt.gcf().suptitle(f"Closed loop — {label}")
        out[("closed loop", label)] = logs
    paths = save_sim4_comparison_plots(out, SIM4_OUTPUT_DIR)
    print(f"Simulation 4: saved {len(paths)} comparison PDFs to {SIM4_OUTPUT_DIR}")
    return out


def sim5(step_deg: float = 10.0):
    """Part 2, Simulation 5 — average thrust-utilisation polar plot: co-linear
    wind/waves/current swept around the vessel, DP at the origin, heading 0,
    SELECTED_OBSERVER in the loop.

    Three polar plots: the utilisation for all directions; the directions
    whose maximum low-frequency excursion stays inside the DP watch circles
    (CAP_POS_LIMIT / CAP_PSI_LIMIT_DEG) — the motion a DP system checks on
    its screen; and the directions whose maximum total motion (with the
    wave-frequency oscillation) stays inside the operability limits
    (CAP_TOTAL_POS_LIMIT / CAP_TOTAL_PSI_LIMIT_DEG) — what a motion-critical
    operation such as a gangway transfer experiences."""
    thrusters = default_thrusters_part_2()

    def run_direction(direction):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=True, observer_type=selected_observer())
        env = (Current(CAP_CURRENT, direction, semantics="from"),
               Wind(CAP_WIND, direction, semantics="from", sigma_slow=0.5, gust=True,
                    sigma_dir=np.deg2rad(2.0), seed=SEED),
               Waves(hs=CAP_HS, tp=CAP_TP, direction=direction, seed=SEED))
        return run(cfg, np.zeros(6), env)

    result = capability_sweep(run_direction, thrusters, step_deg=step_deg,
                              pos_limit=CAP_POS_LIMIT, psi_limit_deg=CAP_PSI_LIMIT_DEG,
                              deviation="lf")
    plot_capability(result, total_limits=(CAP_TOTAL_POS_LIMIT, CAP_TOTAL_PSI_LIMIT_DEG))
    return result


def sim6():
    """Part 2, Simulation 6 — observer robustness: heavy sea, station keeping 1000 s,
    SELECTED_OBSERVER in the loop."""
    cfg = Part2SimConfig(dt=DT, T=1000.0, use_observer=True, observer_type=selected_observer())
    logs = run(cfg, np.zeros(6), environment(hs=ROBUST_HS, tp=ROBUST_TP))
    standard_plots(logs, observer=True)
    return logs


def sim7():
    """Part 2, Simulation 7 — EKF station keeping for 400 s, with all pose
    measurements lost for 200 <= t < 300 s. Standard current, wind and waves;
    no measurement noise. Save trajectory, actuator usage and pose estimates
    to output/pdf/sim7/."""
    cfg = Part2SimConfig(dt=DT, T=400.0, use_observer=True,
                         observer_type="kalman", use_sensor_noise=False,
                         measurement_dropout=(200.0, 300.0))
    logs = run(cfg, np.zeros(6), environment())
    paths = save_sim7_dropout_plots(
        logs, default_thrusters_part_2(), SIM7_OUTPUT_DIR,
        dropout=cfg.measurement_dropout)
    print(f"Simulation 7: saved {len(paths)} PDFs to {SIM7_OUTPUT_DIR}")
    return logs


def bonus_sensor_noise():
    """Extra credit (3 points) — sensor noise: repeat the Part 2, Simulation 4
    observer comparison and the closed-loop raw-vs-observer run with
    measurement noise ON (fixed course noise levels, models/sensors.py)."""
    out = {}
    fixed = WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
    for kind in ("nonlinear_passive", "kalman"):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True,
                             observer_type=kind, use_sensor_noise=True)
        logs = run(cfg, np.zeros(6), environment(), controller=fixed)
        plot_observer(logs).suptitle(f"Bonus: observer {kind}, waves + sensor noise")
        plot_observer_error(logs).suptitle(f"Bonus: observer {kind}, waves + sensor noise — estimation errors")
        out[kind] = logs
    for label, use_obs, kind in (("raw noisy measurements", False, "kalman"),
                                 ("observer kalman", True, "kalman"),
                                 ("observer nonlinear_passive", True, "nonlinear_passive")):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=use_obs,
                             observer_type=kind, use_sensor_noise=True)
        logs = run(cfg, np.zeros(6), environment())
        standard_plots(logs, setpoints=True, observer=use_obs)
        plt.gcf().suptitle(f"Bonus closed loop, sensor noise on — {label}")
        out[("closed loop", label)] = logs
    paths = save_sensor_noise_comparison_plots(
        out, SENSOR_NOISE_OUTPUT_DIR,
        residual_output_dir=SENSOR_NOISE_RESIDUAL_OUTPUT_DIR)
    #print(f"Sensor noise: saved {len(paths)} comparison PDFs to {SENSOR_NOISE_OUTPUT_DIR}")
    return out

SCENARIOS = {"sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4,
             "sim5": sim5, "sim6": sim6, "sim7": sim7, "bonus": bonus_sensor_noise}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--list" in argv:
        for k, f in SCENARIOS.items():
            print(f"{k}: {f.__doc__.splitlines()[0]}")
        return
    name = argv[0] if argv else SCENARIO
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario {name!r}; try --list")
    try:
        SCENARIOS[name]()
    except NotImplementedError as exc:
        # A block you have not written yet, or a choice you have not made.
        # Stop with the message rather than a traceback, and still show
        # whatever the scenario managed to plot before it stopped.
        print(f"\n{name}: stopped — {exc}")
        if plt.get_fignums():
            print(f"showing the {len(plt.get_fignums())} figure(s) produced "
                  "before that point.")
            plt.show()
        return
    plt.show()
    print("Simulation finished.")


if __name__ == "__main__":
    main()
