"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


def _davenport_spectrum(f: np.ndarray, Uref: float, k: float) -> np.ndarray:
    """
    Davenport along-wind gust spectrum (the spectrum chosen for the
    gust component.

    f    : frequency [Hz]
    Uref : reference (mean) wind speed [m/s]
    k    : surface drag coefficient (~0.003 for open sea)
    """
    x = 1200.0 * f / Uref
    return 4.0 * k * Uref ** 2 * x ** 2 / (f * (1.0 + x ** 2) ** (4.0 / 3.0))


class Wind:
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {})
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        self.seed = seed

        # --- load the coefficient table once, reused every step() call ---
        self._alpha_deg, self._C6 = load_wind_coefficients()

        # --- RNG, shared by every stochastic component so a single seed
        #     reproduces the whole run (Part 1 convention) ---
        self._rng = np.random.default_rng(self.seed)

        # --- mean direction, converted ONCE to "towards" (Part 1
        #     convention: "from" is meteorological default, internally we
        #     always work with the direction the wind blows TOWARDS) ---
        self._beta_towards_mean = (self.beta + np.pi if self.semantics == "from"
                                    else self.beta)

        # --- slowly varying SPEED (Part 1): state of the mean-reverting
        #     (Ornstein-Uhlenbeck) process, dU/dt + U/tau_slow = w(t) ---
        self._slow_component = 0.0  # current deviation from mean_speed [m/s]

        # --- slowly varying DIRECTION (new in Part 2): same OU recipe as
        #     the slow speed, applied to a direction offset instead, then
        #     hard-clipped every step to +/- dir_limit ---
        self._dbeta = 0.0  # current direction offset [rad]

        # --- GUST speed (new in Part 2): precompute a fixed set of
        #     frequencies, spectral amplitudes and random phases ONCE
        #     (seeded), then sum them at whatever t is passed to step().
        #     This keeps the gust a pure, reproducible function of time. ---
        gp = self.gust_params ##Dictionnary
        self.gust_n_harmonics = int(gp.get("n_harmonics", 50)) ##50 different harmonics
        self.gust_f_min = float(gp.get("f_min", 1.0 / 600.0))  # 10 min period, slowest gust
        self.gust_f_max = float(gp.get("f_max", 1.0))          # 1 s period, fastest gust
        self.gust_k = float(gp.get("k", 0.003))                # open-sea drag coeff
        Uref = max(self.mean_speed, 0.1)  # Davenport spectrum reference speed, not to divide by 0

        if self.gust:
            f = np.logspace(np.log10(self.gust_f_min),
                             np.log10(self.gust_f_max),
                             self.gust_n_harmonics)
            df = np.gradient(f)  # bin widths for log-spaced frequencies
            S = _davenport_spectrum(f, Uref, self.gust_k)
            self._gust_f = f
            self._gust_amp = np.sqrt(2.0 * S * df)
            self._gust_phase = self._rng.uniform(0.0, 2.0 * np.pi, size=f.size)
        else:
            self._gust_f = np.array([])
            self._gust_amp = np.array([])
            self._gust_phase = np.array([])

    def _gust_speed(self, t: float) -> float:
        """Zero-mean gust speed at time t, from the precomputed harmonics."""
        if not self.gust or self._gust_f.size == 0:
            return 0.0
        return float(np.sum(self._gust_amp *
                             np.cos(2.0 * np.pi * self._gust_f * t + self._gust_phase)))

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        # ------------------------------------------------------------
        # 1) Ambient wind speed: mean + slowly-varying component
        #    (identical to Part 1).
        # ------------------------------------------------------------
        if self.sigma_slow > 0.0 and self.tau_slow > 0.0:
            w_k = self._rng.normal(0.0, 1.0)  # N(0, 1)
            # dU/dt + U/tau_slow = w   =>   Euler-Maruyama step. The driving
            # noise intensity is scaled so the PROCESS itself (not the raw
            # noise) has stationary standard deviation sigma_slow, i.e.
            # Var[U_slow] = sigma_sde^2 * tau_slow / 2 = sigma_slow^2.
            sigma_slow_sde = self.sigma_slow / np.sqrt(self.tau_slow / 2.0)
            self._slow_component += (
                -self._slow_component / self.tau_slow * dt
                + sigma_slow_sde * np.sqrt(dt) * w_k
            )
        U_mean = self.mean_speed + self._slow_component

        # --- gust speed, added on top of the mean+slow speed (Part 2) ---
        U_gust = self._gust_speed(t)
        U_ambient = max(U_mean + U_gust, 0.0)  # speed cannot be negative

        # ------------------------------------------------------------
        # 2) Ambient direction: mean ("towards") + bounded slowly varying
        #    offset (Part 2), no gust component on direction.
        # ------------------------------------------------------------
        if self.sigma_dir > 0.0 and self.tau_dir > 0.0:
            w_beta = self._rng.normal(0.0, 1.0)
            sigma_dir_sde = self.sigma_dir / np.sqrt(self.tau_dir / 2.0)
            self._dbeta += (
                -self._dbeta / self.tau_dir * dt
                + sigma_dir_sde * np.sqrt(dt) * w_beta
            )
            self._dbeta = np.clip(self._dbeta, -self.dir_limit, self.dir_limit)

        beta_towards = self._beta_towards_mean + self._dbeta
        unit_towards = np.array([np.cos(beta_towards), np.sin(beta_towards)])

        # ------------------------------------------------------------
        # 3) Ambient wind as a NED vector [V_N, V_E] (Part 1).
        # ------------------------------------------------------------
        V_wind_ned = U_ambient * unit_towards

        # ------------------------------------------------------------
        # 4) Relative wind in the BODY frame (Part 1): rotate the NED
        #    wind vector into BODY with J^T(psi), then subtract the
        #    vessel's own BODY-frame velocity (u, v).
        # ------------------------------------------------------------
        psi = eta[5]
        cpsi, spsi = np.cos(psi), np.sin(psi)
        J_T = np.array([[cpsi, spsi],
                         [-spsi, cpsi]])
        V_wind_body = J_T @ V_wind_ned
        V_rw_body = V_wind_body - nu[0:2]

        U_rw = float(np.linalg.norm(V_rw_body))
        alpha_rw = float(np.arctan2(V_rw_body[1], V_rw_body[0]))  # rad, (-pi, pi)

        # ------------------------------------------------------------
        # 5) Interpolate the coefficient table at alpha_rw (periodic,
        #    360 deg) and compute the six generalized loads (Part 1).
        # ------------------------------------------------------------
        alpha_rw_deg = np.degrees(alpha_rw)
        C = np.array([
            np.interp(alpha_rw_deg, self._alpha_deg, self._C6[:, i], period=360.0)
            for i in range(6)
        ])
        tau_w6 = (U_rw ** 2) * C

        info = {
            "U": U_ambient,
            "beta_ned": float(np.mod(beta_towards, 2 * np.pi)),
            "alpha_body": alpha_rw,
            "U_mean": U_mean,
            "U_gust": U_gust,
        }
        return tau_w6, info