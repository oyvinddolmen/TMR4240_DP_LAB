"""Part 2 checks as pytest tests, plus unit tests for the provided
Part 2 infrastructure (``simulation/capability.py`` and the observer
measurement contract of ``simulation/simulation_part_2.py``).

Each check test wraps one check from ``simulation/checks_part_2.py``.  A
subsystem whose template placeholder is still in place is reported as
skipped, not failed, so a fresh template starts green.  The capability tests
do not depend on any student code.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from simulation.capability import max_deviation
from simulation.checks_part_2 import ALL_CHECKS, run_check


@pytest.mark.parametrize("key", list(ALL_CHECKS), ids=list(ALL_CHECKS))
def test_part_2(key):
    result = run_check(key)
    detail = "\n".join([result.name, *result.details])
    if result.status == "NOT IMPLEMENTED":
        pytest.skip(detail)
    assert result.passed, detail


# ---------------------------------------------------------------------------
# simulation/capability.py — deviation measures (no student code involved)
# ---------------------------------------------------------------------------

def _oscillation_logs(T=600.0, dt=0.1, amp_deg=5.0, period=9.0, phase=0.0):
    """Synthetic station-keeping logs: pure heading oscillation, zero position."""
    t = np.arange(0.0, T + dt / 2, dt)
    eta = np.zeros((len(t), 6))
    eta[:, 5] = np.deg2rad(amp_deg) * np.sin(2.0 * np.pi / period * t + phase)
    return SimpleNamespace(t=t, eta=eta, sp=np.zeros((len(t), 6)))


def test_max_deviation_total_sees_the_oscillation():
    _, psi = max_deviation(_oscillation_logs(), deviation="total")
    assert np.degrees(psi) == pytest.approx(5.0, abs=0.05)


def test_max_deviation_default_is_total():
    logs = _oscillation_logs()
    assert max_deviation(logs) == max_deviation(logs, deviation="total")


def test_max_deviation_lf_attenuates_the_oscillation():
    # A 30 s moving average of a 9 s, 5 deg oscillation leaves
    # |sin(w W/2)/(w W/2)| * 5 deg = 0.41 deg in the interior.
    _, psi = max_deviation(_oscillation_logs(), deviation="lf", window=30.0)
    assert 0.3 <= np.degrees(psi) <= 0.5


def test_max_deviation_lf_is_end_phase_independent():
    # Without trimming half a filter window at the ends, the edge-padded
    # moving average makes the reported maximum depend on the wave phase at
    # the end of the run (0.5-2.3 deg for this signal).
    maxima = [np.degrees(max_deviation(_oscillation_logs(phase=ph),
                                       deviation="lf", window=30.0)[1])
              for ph in np.linspace(0.0, 2.0 * np.pi, 9)]
    assert max(maxima) <= 0.5
    assert max(maxima) - min(maxima) <= 0.1


def test_max_deviation_lf_short_run_falls_back():
    # A run shorter than the filter window must still return a finite value.
    pos, psi = max_deviation(_oscillation_logs(T=20.0), deviation="lf", window=30.0)
    assert np.isfinite(pos) and np.isfinite(psi)


# ---------------------------------------------------------------------------
# simulation/simulation_part_2.py — the observer measurement contract.
# Velocities are NOT measured on the vessel: the observer must receive the
# measured POSE only (plus the previous step's desired wrench) and the
# controller must receive the observer's ESTIMATES.  These tests fail if
# anyone wires velocity measurements into the observer or bypasses the
# estimates — do not weaken them.
# ---------------------------------------------------------------------------

_NU_MARKER = np.array([0.111, 0.222, 0.0, 0.0, 0.0, 0.333])


class _SpyObserver:
    name = "spy"

    def __init__(self):
        self.received = []          # (t, dt, eta_measured, tau_est) per step

    def reset(self, eta0=None):
        self.received = []

    def step(self, t, dt, eta_measured, tau_est):
        # No *args/**kwargs on purpose: an engine that passes anything beyond
        # (t, dt, eta_measured, tau_est) makes this call — and the test — fail.
        self.received.append((float(t), float(dt),
                              np.array(eta_measured, dtype=float),
                              np.array(tau_est, dtype=float)))
        return np.array(eta_measured, dtype=float), _NU_MARKER.copy(), np.zeros(6)


class _SpyController:
    def __init__(self, tau=8.5e3):
        self.tau = float(tau)
        self.seen = []              # (eta, nu) the controller was given

    def compute(self, t, dt, eta, nu, eta_ref, nu_ref=None, acc_ref=None):
        self.seen.append((np.array(eta, dtype=float), np.array(nu, dtype=float)))
        out = np.zeros(6)
        out[0] = self.tau           # nonzero so tau_est carries information
        return out


def _spy_run(T=3.0):
    from part_2.config import Part2SimConfig, default_thrusters_part_2
    from simulation.simulation_part_2 import DPSimulatorPart2

    obs, ctl = _SpyObserver(), _SpyController()
    cfg = Part2SimConfig(dt=0.1, T=T, use_observer=True, use_reference=False,
                         use_sensor_noise=True)
    sim = DPSimulatorPart2(cfg, ctl, default_thrusters_part_2(), observer=obs)
    sim.reset_state()
    logs = sim.run(np.zeros(6))
    return obs, ctl, logs


def test_observer_receives_the_measured_pose_only():
    obs, _, logs = _spy_run()
    assert len(obs.received) == len(logs.t)
    eta_seen = np.array([r[2] for r in obs.received])
    assert eta_seen.shape == (len(logs.t), 6)
    # exactly the sensor output, which (noise on) is NOT the true state
    assert np.array_equal(eta_seen, logs.eta_measured)
    assert not np.allclose(eta_seen, logs.eta)


def test_observer_receives_the_previous_desired_wrench():
    obs, ctl, logs = _spy_run()
    tau_seen = np.array([r[3] for r in obs.received])
    assert np.allclose(tau_seen[0], 0.0)                    # nothing before t = 0
    assert np.array_equal(tau_seen[1:], logs.tau_d[:-1])    # tau_d delayed one step
    assert np.allclose(tau_seen[1:, 0], ctl.tau)


def test_controller_receives_the_observer_estimates():
    obs, ctl, logs = _spy_run()
    nu_ctrl = np.array([nu for _, nu in ctl.seen])
    # the controller sees the observer's velocity ESTIMATE (marker), never the
    # simulated or "measured" velocity
    assert np.allclose(nu_ctrl, _NU_MARKER)
    eta_ctrl = np.array([eta for eta, _ in ctl.seen])
    assert np.array_equal(eta_ctrl, logs.eta_measured)      # = spy estimate = measurement
    assert np.array_equal(logs.nu_est, np.tile(_NU_MARKER, (len(logs.t), 1)))
