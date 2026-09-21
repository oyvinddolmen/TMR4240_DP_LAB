"""Irregular-wave loads for Project Part 2.

The model discretizes an ITTC/modified Pierson--Moskowitz spectrum and uses
the Gunnerus force RAOs and drift-force QTFs from the corrected database
``data/gunnerus_vessel.json`` (why it is corrected: ``data/GUNNERUS_WAVE_DATA.md``;
regression tests: ``notebooks/part_2_waves_verification.ipynb``, section 7).
The result is a callable compatible with ``DPSimulatorPart2.run(waves=...)``.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Callable

import numpy as np
from mcsimpy.utils import pipi
from mcsimpy.waves.wave_loads import WaveLoad
from mcsimpy.waves.wave_spectra import ModifiedPiersonMoskowitz


class _WaveLoadComplexInterp(WaveLoad):
    """``WaveLoad`` with an interpolation-safe force-RAO lookup.

    The parent class interpolates the stored force-RAO *phase* between the two
    neighbouring 10-degree heading columns as a plain number.  Phase is an
    angle: whenever two adjacent columns lie on opposite sides of the +-180
    degree wrap (or on different 360-degree branches), the straight-line blend
    takes the long way around the circle and can even invert the sign of the
    load mid-interval.  As the vessel oscillates in the waves, the relative
    heading sweeps back and forth across such a seam, and the phase error
    rectifies the wave-frequency load into a spurious slowly varying force or
    moment.

    This subclass interpolates the complex response

        H(omega, beta) = amp * exp(1j * phase)

    and returns ``(|H|, angle(H))``: the blend is then independent of the
    stored phase branch and always takes the shortest angular path.  Values at
    the grid headings are unchanged.

    Note: the second-order (Newman) loads require ``qtf_interp_angles=True``
    (the mcsimpy default) — ``second_order_loads`` indexes a 360-entry heading
    grid that exists only in that branch.  Do not disable it.
    """

    def _rao_interp(self, rel_angle):
        deg = np.rad2deg(self._qtf_angles)
        index_lb = np.argmin(
            np.abs(deg - np.floor(np.rad2deg(rel_angle[:, None]) / 10.0) * 10.0), axis=1)
        index_ub = np.where(index_lb < len(self._qtf_angles) - 1, index_lb + 1, 0)
        freq_ind = np.arange(0, self._N)
        h_lb = (self._forceRAOamp[:, freq_ind, index_lb]
                * np.exp(1j * self._forceRAOphase[:, freq_ind, index_lb]))
        h_ub = (self._forceRAOamp[:, freq_ind, index_ub]
                * np.exp(1j * self._forceRAOphase[:, freq_ind, index_ub]))
        theta1, theta2 = self._qtf_angles[index_lb], self._qtf_angles[index_ub]
        scale = pipi(rel_angle - theta1) / pipi(theta2 - theta1)
        h = h_lb + (h_ub - h_lb) * scale
        return np.abs(h), np.angle(h)


class Waves:
    """Long-crested irregular sea and resulting 6-DOF BODY wrench.

    Parameters
    ----------
    hs, tp:
        Significant wave height [m] and peak period [s].
    direction:
        NED direction in radians.  By default this is the direction the waves
        come *from* (0 = North, pi/2 = East), as normally stated in weather
        conditions.  Set ``direction_is_from=False`` when supplying the
        propagation direction used by MSS/mcsimpy.  Pass a callable
        ``direction(t) -> rad`` for a sea whose direction changes with time;
        it is evaluated at every call and applied with :meth:`set_direction`.
    n_components:
        Number of frequency components used to discretize the spectrum.
    seed:
        Seed for the random phases (and frequency jitter), making
        comparisons reproducible.
    random_frequencies:
        Perturb each component frequency uniformly within its spectral bin,
        exactly like ``rand_freq`` in the MSS ``Waves`` block.  With an
        equispaced grid the difference frequencies are all multiples of
        ``domega``, so the slowly-varying drift loads would repeat exactly
        every ``2*pi/domega`` seconds (about a minute for typical sea
        states); the jitter removes that artifact.  Disable only when you
        need a strictly periodic signal for debugging.

    Notes
    -----
    The current vessel plant integrates surge, sway and yaw only, but the full
    six-component wave wrench is returned and logged.  Heave, roll and pitch
    do not affect the present 3-DOF vessel state.
    """

    def __init__(
        self,
        hs: float = 1.5,
        tp: float = 8.0,
        direction: float | Callable[[float], float] = np.deg2rad(45.0),
        *,
        direction_is_from: bool = True,
        n_components: int = 20,
        seed: int = 123,
        random_frequencies: bool = True,
        omega_min: float | None = None,
        omega_max: float | None = None,
        config_file: str | None = None,
    ) -> None:
        if hs <= 0.0 or tp <= 0.0:
            raise ValueError("hs and tp must be positive")
        if n_components < 2:
            raise ValueError("n_components must be at least 2")

        wp = 2.0 * np.pi / tp
        # This range resolves the energetic part of the ITTC spectrum without
        # placing components at omega=0, where the spectrum is singular.
        lo = 0.25 * wp if omega_min is None else float(omega_min)
        hi = 3.0 * wp if omega_max is None else float(omega_max)
        if not 0.0 < lo < hi:
            raise ValueError("require 0 < omega_min < omega_max")

        self.hs = float(hs)
        self.tp = float(tp)
        self._direction_is_from = bool(direction_is_from)
        self._direction_fn = direction if callable(direction) else None
        direction0 = float(direction(0.0)) if callable(direction) else float(direction)
        self.direction_from = direction0 if direction_is_from else direction0 - np.pi
        self.direction = float(
            (direction0 + np.pi if direction_is_from else direction0) % (2.0 * np.pi)
        )
        bin_centers = np.linspace(lo, hi, n_components)
        _, spectral_density = ModifiedPiersonMoskowitz(bin_centers)(hs, tp)
        domega = bin_centers[1] - bin_centers[0]
        # Amplitudes are taken from the spectrum at the bin centers,
        # a_i = sqrt(2 S(w_i) dw), as in MSS Wave_init.m.
        self.amplitudes = np.sqrt(2.0 * spectral_density * domega)
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, n_components)
        self.frequencies = bin_centers
        if random_frequencies:
            self.frequencies = bin_centers + rng.uniform(
                -0.5 * domega, 0.5 * domega, n_components
            )
        self.angles = np.full(n_components, self.direction)

        rao_file = config_file
        if rao_file is None:
            # Corrected Gunnerus hydrodynamic database. The table shipped
            # with mcsimpy has corrupt force-RAO phases for relative headings
            # 0-180 deg and a drift-table layout that silently zeroes the yaw
            # drift moment — see data/GUNNERUS_WAVE_DATA.md.
            rao_file = str(Path(__file__).resolve().parent.parent
                           / "data" / "gunnerus_vessel.json")
        # mcsimpy prints a three-line QTF construction banner.  Suppress that
        # implementation detail so directional sweeps keep readable output.
        with contextlib.redirect_stdout(io.StringIO()):
            self._load = _WaveLoadComplexInterp(
                self.amplitudes,
                self.frequencies,
                self.phases,
                self.angles,
                rao_file,
            )

    def set_direction(self, direction: float, *, direction_is_from: bool | None = None) -> None:
        """Change the wave direction of all components (radians, NED).

        ``direction_is_from`` defaults to the convention chosen at
        construction.  The spectrum, amplitudes and phases are unchanged, so
        the sea state stays statistically the same while its direction turns;
        the loads follow the new direction from the next call.
        """
        is_from = self._direction_is_from if direction_is_from is None else direction_is_from
        d = float(direction)
        self.direction_from = d if is_from else d - np.pi
        self.direction = float((d + np.pi if is_from else d) % (2.0 * np.pi))
        self.angles[:] = self.direction            # same array object as WaveLoad._angles

    def __call__(self, t: float, eta: np.ndarray, nu: np.ndarray | None = None) -> np.ndarray:
        """Return BODY ``[Fx, Fy, Fz, Mx, My, Mz]`` in N and Nm."""
        if self._direction_fn is not None:
            self.set_direction(self._direction_fn(float(t)))
        eta6 = np.asarray(eta, dtype=float).reshape(6)
        return np.asarray(self._load(float(t), eta6), dtype=float).reshape(6)
