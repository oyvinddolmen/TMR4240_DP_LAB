"""EKF prediction, recovery and simulator routing when pose measurements fail."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from part_2.config import Part2SimConfig, default_thrusters_part_2
from part_2.controller import DPController
from part_2.observer import KalmanFilterObserver
from simulation.simulation_part_2 import DPSimulatorPart2


@pytest.mark.parametrize("missing", [None, np.full(6, np.nan)], ids=["None", "NaN"])
def test_missing_measurements_predict_state_and_covariance(missing):
    observer = KalmanFilterObserver()
    observer.x[9:12] = [0.25, -0.1, 0.05]
    position_variance = observer.P[6, 6]
    estimate = observer.step(0.0, 0.1, missing, np.zeros(6))
    assert estimate.eta[[0, 1, 5]] == pytest.approx([0.025, -0.01, 0.005])
    assert observer.P[6, 6] > position_variance
    assert np.isfinite(observer.x).all()
    assert np.isfinite(observer.P).all()
    assert np.allclose(observer.P, observer.P.T)
    assert np.linalg.eigvalsh(observer.P).min() >= -1e-8
    assert np.isnan(observer.last_innovation).all()


def test_measurements_resume_correction_after_outage():
    observer = KalmanFilterObserver()
    for step in range(100):
        observer.step(step * 0.1, 0.1, None, np.zeros(6))
    prediction = deepcopy(observer)
    prediction.step(10.0, 0.1, None, np.zeros(6))
    measurement = np.array([3.0, -2.0, 0.0, 0.0, 0.0, 0.1])
    y = measurement[[0, 1, 5]]
    observer.step(10.0, 0.1, measurement, np.zeros(6))
    assert np.linalg.norm(y - observer.H @ observer.x) < np.linalg.norm(y - prediction.H @ prediction.x)
    assert observer.P[6, 6] < prediction.P[6, 6]
    assert np.isfinite(observer.last_innovation).all()
    assert np.linalg.eigvalsh(observer.P).min() >= -1e-8


def test_partial_measurements_use_only_available_components():
    observer = KalmanFilterObserver()
    prediction = deepcopy(observer)
    prediction.step(0.0, 0.1, None, np.zeros(6))
    measurement = np.array([np.nan, np.nan, 0.0, 0.0, 0.0, 0.1])
    observer.step(0.0, 0.1, measurement, np.zeros(6))
    assert np.isnan(observer.last_innovation[:2]).all()
    assert np.isfinite(observer.last_innovation[2])
    assert observer.P[8, 8] < prediction.P[8, 8]
    assert np.isfinite(observer.x).all()


def test_simulator_drops_pose_at_start_and_restores_at_stop():
    cfg = Part2SimConfig(dt=0.1, T=0.6, observer_type="kalman",
                         use_sensor_noise=False, measurement_dropout=(0.2, 0.4))
    sim = DPSimulatorPart2(cfg, DPController(), default_thrusters_part_2())
    logs = sim.run(np.zeros(6))
    pose = logs.eta_measured[:, [0, 1, 5]]
    assert np.isnan(pose).all(axis=1).tolist() == [False, False, True, True, False, False, False]
    assert np.allclose(pose[[0, 1, 4, 5, 6]], logs.eta[[0, 1, 4, 5, 6]][:, [0, 1, 5]])
    for values in (logs.eta, logs.eta_est, logs.nu_est, logs.tau_d, logs.u):
        assert np.isfinite(values).all()


@pytest.mark.parametrize("interval", [(-1.0, 0.2), (0.4, 0.2), (0.1, 0.8), (np.nan, 0.3)])
def test_invalid_dropout_intervals_are_rejected(interval):
    cfg = Part2SimConfig(T=0.6, measurement_dropout=interval)
    with pytest.raises(ValueError, match="measurement_dropout"):
        DPSimulatorPart2(cfg, DPController(), default_thrusters_part_2())


def test_sim7_configuration_and_exports_do_not_depend_on_selected_observer():
    import run_case_part_2 as rc

    logs = SimpleNamespace()
    with patch.object(rc, "run", return_value=logs) as run, \
            patch.object(rc, "save_sim7_dropout_plots", return_value=[]) as save, \
            patch.object(rc, "SELECTED_OBSERVER", None):
        assert rc.sim7() is logs
    cfg, command, _ = run.call_args.args
    assert cfg.T == 400.0
    assert cfg.observer_type == "kalman" and cfg.use_observer
    assert not cfg.use_sensor_noise
    assert cfg.measurement_dropout == (200.0, 300.0)
    assert np.array_equal(command, np.zeros(6))
    assert save.call_args.kwargs["dropout"] == cfg.measurement_dropout
