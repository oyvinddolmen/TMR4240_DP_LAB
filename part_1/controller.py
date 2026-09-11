"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
from part_1.config import TuningParameters
from simulation.utils import Rz, wrap_angle_pi, ned_to_body_xy

class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):
        # Tuning parameters
        self.tuningParameters = TuningParameters()

        # Integral states
        self.integral_ned = np.zeros(2)
        self.integral_psi = 0.0

    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        pass

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,   
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        # TODO: Replace this placeholder with your DP controller.
        # Return the (6,) desired BODY wrench — fill in tau_d[0] = Fx,
        # tau_d[1] = Fy, tau_d[5] = Mz and leave the rest zero.

        # NOTE: reference parameters comes in NED frame, eta and nu comes in BODY frame. Return tau_desired in BODY frame


        # ---------- PID-REGULATOR ------------

        # Extracting states:
        N     = eta[0]
        E     = eta[1]
        psi   = eta[5]
        N_d   = eta_ref[0]
        E_d   = eta_ref[1]
        psi_d = eta_ref[5]

        # Position errors in NED frame:
        e_ned = np.array([
            N_d - N,
            E_d - E
        ])

        # Heading error wrapped to (-pi, pi]
        e_psi = wrap_angle_pi(psi_d - psi)

        # REGULATOR - integral effect
        self.integral_ned += e_ned * dt
        self.integral_psi += e_psi * dt

        # REGULATOR - derivative effect
        u = nu[0]
        v = nu[1]
        r = nu[5]

        J = Rz(psi) 
        nu_ned = J @ np.array([u, v, r]) # nu comes in BODY frame so must be transformed to NED
        nu_ref_ned = np.array([
            nu_ref[0],
            nu_ref[1],
            nu_ref[5]
        ])
        e_dot_ned = nu_ref_ned[:2] - nu_ned[:2]
        e_dot_psi = nu_ref_ned[2] - nu_ned[2]

        # PID regulator for NED
        # north and east
        F_ned_P = self.tuningParameters.Kp @ e_ned
        F_ned_I = self.tuningParameters.Ki @ self.integral_ned
        F_ned_D = self.tuningParameters.Kd @ e_dot_ned
        F_ned = F_ned_P + F_ned_I + F_ned_D

        # yaw part:
        Mz_P = self.tuningParameters.Kp_psi * e_psi
        Mz_I = self.tuningParameters.Ki_psi * self.integral_psi
        Mz_D = self.tuningParameters.Kd_psi * e_dot_psi
        Mz = Mz_P + Mz_I + Mz_D


        # Transform the control input from NED -> BODY
        F_body = ned_to_body_xy(F_ned[:2], psi)
        Fx = F_body[0]
        Fy = F_body[1]

        # create return variable tau_desired
        tau_d = np.zeros(6)
        tau_d[0] = Fx
        tau_d[1] = Fy
        tau_d[5] = Mz

        # TODO: anti-integrator-windupø

        return tau_d
