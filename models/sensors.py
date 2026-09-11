"""Measurement layer used by the Part 2 simulator.

The observer receives pose measurements only (GNSS position, gyrocompass
heading) — vessel velocities are NOT measured on board; estimating them from
the pose is the observer's job.  A "measured" velocity is still generated
here and is never passed to an observer.  It has two uses: the validation
plots, and the no-observer baseline runs (Part 2, Simulations 2 and 3, and the
raw closed-loop comparison of Simulation 4), which feed it straight to the
controller.  That baseline is the idealised one Simulation 4 asks you to
compare with pose-only observer feedback: with noise off the velocity it uses
equals the simulated velocity exactly, while with noise on (the extra-credit
task) the baseline controller sees a NOISY velocity, which is part of what
that comparison exposes.

Measurement noise is switched on or off per simulation with
``Part2SimConfig(use_sensor_noise=...)`` — off by default, so the mandatory
simulations run with ideal measurements.  The noise levels themselves are
fixed course parameters (below): they represent a DP-grade sensor suite
(GNSS position, gyrocompass heading) and are sized so that a well-tuned
observer can handle them.  They are not student tuning knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from simulation.utils import wrap_angle_pi


# Fixed 1-sigma measurement noise of the sensor suite (course parameters).
# Pose: GNSS North/East and gyrocompass heading.  The observer never receives
# a velocity measurement; the velocity noise below reaches the validation plots
# and, in the no-observer baseline runs, the controller itself.
SENSOR_ETA_STD = np.array([0.20, 0.20, 0.0, 0.0, 0.0, np.deg2rad(0.20)])
SENSOR_NU_STD = np.array([0.02, 0.02, 0.0, 0.0, 0.0, np.deg2rad(0.05)])


@dataclass(frozen=True)
class SensorConfig:
    """Sensor settings: only the noise switch and the seed are selectable.

    The standard deviations are the fixed course parameters above; they are
    intentionally not constructor arguments.
    """

    noise: bool = False
    seed: int = 1234
    eta_std: np.ndarray = field(
        default_factory=lambda: SENSOR_ETA_STD.copy(), init=False, repr=False
    )
    nu_std: np.ndarray = field(
        default_factory=lambda: SENSOR_NU_STD.copy(), init=False, repr=False
    )


class VesselSensors:
    """Turn the true vessel state into (optionally noisy) measurements."""

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        self._validate()
        self.reset()

    def _validate(self):
        if self.config.eta_std.shape != (6,) or self.config.nu_std.shape != (6,):
            raise ValueError("sensor standard deviations must have six elements")

    def reset(self):
        self.rng = np.random.default_rng(self.config.seed)

    def measure(self, eta: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eta_m = np.asarray(eta, dtype=float).reshape(6).copy()
        nu_m = np.asarray(nu, dtype=float).reshape(6).copy()
        if self.config.noise:
            eta_m += self.rng.normal(0.0, self.config.eta_std)
            nu_m += self.rng.normal(0.0, self.config.nu_std)
        eta_m[5] = wrap_angle_pi(eta_m[5])
        return eta_m, nu_m
