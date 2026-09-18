"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from typing import Tuple
import numpy as np
from simulation.utils import wrap_angle_pi

# Per-axis tuning parameters live with the rest of the Part 1 configuration.
from part_1.config import RefAxisConfig


class ReferenceModel:
    """
    Template for student reference model.

    The default implementation is pass-through, so eta_ref = eta_cmd and the
    reference velocities/accelerations are zero.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Using the second-order low-pass filter from Part 1 task description
        eta_cmd = np.asarray(eta_cmd, dtype=float).reshape(6)

        # ------------------- North and East ------------------
        wn_xy = self.cfg_xy.wn
        zeta_xy = self.cfg_xy.zeta

        # calculating x'' for North and East
        # no need to rotate nu to NED frame, because nu_ref = [X_ref', Y_ref', psi_ref']
        for i in [0, 1]:
            error = eta_cmd[i] - self.eta_ref[i]

            self.acc_ref[i] = (
                wn_xy**2 * error
                - 2.0 * zeta_xy * wn_xy * self.nu_ref[i]
            )

            # forward euler discretization
            self.nu_ref[i] += dt * self.acc_ref[i]
            self.eta_ref[i] += dt * self.nu_ref[i]

        # -------------------- Yaw -------------------------
        wn_psi = self.cfg_psi.wn
        zeta_psi = self.cfg_psi.zeta

        psi_error = wrap_angle_pi(eta_cmd[5] - self.eta_ref[5])

        self.acc_ref[5] = (
            wn_psi**2 * psi_error
            - 2.0 * zeta_psi * wn_psi * self.nu_ref[5]
        )

        # forward euler to discretize
        self.nu_ref[5] += dt * self.acc_ref[5]
        self.eta_ref[5] += dt * self.nu_ref[5]

        return (
            self.eta_ref.copy(),
            self.nu_ref.copy(),
            self.acc_ref.copy(),
        )
