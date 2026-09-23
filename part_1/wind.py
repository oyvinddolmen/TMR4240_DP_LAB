"""
Wind template

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
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
----------------------
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


class Wind:
    """Template for student wind model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    seed : random seed for the slow component, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = sigma_slow / np.sqrt(120/2)          # TEMPORARY CHANGE FROM ØYVIND
        self.tau_slow = float(tau_slow)
        self.seed = seed

        # --- load the coefficient table once, reused every step() call ---
        self._alpha_deg, self._C6 = load_wind_coefficients()

        # --- state for the slowly-varying wind speed component ---
        # A first-order (Ornstein-Uhlenbeck-like) process: it wanders
        # randomly but "relaxes" back towards zero with time constant
        # tau_slow, so the wind speed changes smoothly rather than
        # jumping around every time step.
        self._rng = np.random.default_rng(self.seed)
        self._slow_component = 0.0  # current deviation from mean_speed [m/s]

    def _towards_ned(self) -> np.ndarray:
        """Ambient wind direction as a unit vector [N, E], the direction
        the wind blows TOWARDS (converting from "from" semantics if needed)."""
        ##BETA IN RADIANS
        beta_towards = self.beta + np.pi if self.semantics == "from" else self.beta
        return np.array([np.cos(beta_towards), np.sin(beta_towards)]), beta_towards

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        # ------------------------------------------------------------
        # 1) Ambient wind speed: mean + slowly-varying component.
        # ------------------------------------------------------------
        if self.sigma_slow > 0.0 and self.tau_slow > 0.0:
            w_k = self._rng.normal(0.0, 1.0)#Gaussian with mean =0 and deviation=1
            # dU/dt+mu*U=w
            #mu=1/tau_slow
            self._slow_component += (
                -self._slow_component / self.tau_slow * dt
                + self.sigma_slow * np.sqrt(dt) * w_k
            )
        U_ambient = max(self.mean_speed + self._slow_component, 0.0)#prevent the wind from being negativ, 0 if it happens.

        # ------------------------------------------------------------
        # 2) Ambient wind as a NED vector [V_N, V_E], direction "towards".
        # ------------------------------------------------------------
        unit_towards, beta_towards = self._towards_ned()
        V_wind_ned = U_ambient * unit_towards  # [V_N, V_E]

        # ------------------------------------------------------------
        # 3) Relative wind in the BODY frame:
        #    rotate the NED wind vector into BODY with J^T(psi),
        #    then subtract the vessel's own BODY-frame velocity (u, v).
        # ------------------------------------------------------------
        psi = eta[5]
        cpsi, spsi = np.cos(psi), np.sin(psi)
        # J^T(psi), top-left 2x2 block of the 3-DOF kinematic Jacobian's
        # transpose (rotates NED vectors into the BODY frame):
        J_T = np.array([[cpsi, spsi],
                         [-spsi, cpsi]])
        V_wind_body = J_T @ V_wind_ned          # ambient wind, BODY frame
        V_rw_body = V_wind_body - nu[0:2]       # relative wind, BODY frame

        U_rw = float(np.linalg.norm(V_rw_body))
        alpha_rw = float(np.arctan2(V_rw_body[1], V_rw_body[0]))  # rad, (-pi, pi)

        # ------------------------------------------------------------
        # 4) Interpolate the coefficient table at alpha_rw (periodic, 360 deg)
        #    and compute the six generalized loads.
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
        }
        return tau_w6, info
