"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray
    wave_pose: np.ndarray | None = None


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the nonlinear passive observer.
        return self._placeholder(eta_measured)


class KalmanFilterObserver(Observer):
    """15-state EKF for 3-DOF low-frequency DP state estimation.

    State:
        x = [xi, xi_dot, eta, nu, bias], all reduced 3-DOF vectors.

        xi      : wave-frequency pose contribution, NED
        xi_dot  : derivative of wave-frequency contribution
        eta     : low-frequency pose [N, E, psi], NED
        nu      : [u, v, r], BODY
        bias    : slowly varying environmental bias, NED

    Input:
        eta_measured : 6-DOF measured pose
        tau_est      : previous-step desired BODY wrench
    """

    name = "kalman"

    # Reduced indices in the simulator's 6-DOF vectors.
    IDX = np.array([0, 1, 5])

    def __init__(
        self,
        omega_w=0.8,                    # natural frequency of the modelled waves
        zeta_w=0.7,                     # dampening ratio of wave model
        T_bias=(300.0, 300.0, 300.0),   # time constant of enviroment bias, b
        
        # Process noise parameters: large value -> trust measurement more
        q_w_pose=1e-6,                  
        q_w_rate=1e-4,                      
        q_eta_N = 1e-6,                        
        q_eta_E = 1e-6,
        q_eta_psi = 1e-7,
        q_nu_u = 1e-4,
        q_nu_v = 1e-5,
        q_nu_r = 1e-7,
        q_bias=1e-2,

        # Measurement noise
        R_diag=(0.05**2, 0.05**2, np.deg2rad(0.1)**2),  
    ):
        self.omega_w = float(omega_w)   # for second order wave frequency oscillator
        self.zeta_w = float(zeta_w)     # -||-
        self.T_bias = np.asarray(T_bias, dtype=float).reshape(3)
        self.q_w_pose = float(q_w_pose)
        self.q_w_rate = float(q_w_rate)
        self.q_eta = np.array([
            q_eta_N,
            q_eta_E,
            q_eta_psi
        ])
        self.q_nu = np.array([
            q_nu_u,
            q_nu_v,
            q_nu_r
        ])
        self.q_bias = float(q_bias)

        self.R = np.diag(np.asarray(R_diag, dtype=float))
        if np.any(self.R.diagonal() <= 0.0):
            raise ValueError("R_diag must contain positive variances.")
        if np.any(self.T_bias <= 0.0):
            raise ValueError("Bias time constants must be positive.")
        if self.omega_w <= 0.0 or self.zeta_w <= 0.0:
            raise ValueError("omega_w and zeta_w must be positive.")

        # Gunnerus matrices 
        self.M = np.array([
            [6.007e5, 0.0,     0.0],
            [0.0,     7.067e5, -4.733e5],
            [0.0,    -5.712e5,  5.456e7],
        ])
        self.D = np.diag([1.118e3, 2.229e4, 1.950e6])

        # Solve linear systems rather than forming inv(M).
        self.M_inv_D = np.linalg.solve(self.M, self.D)
        self.M_inv = np.linalg.solve(self.M, np.eye(3))

        self.H = np.zeros((3, 15))
        self.H[:, 0:3] = np.eye(3)   # wave pose contribution
        self.H[:, 6:9] = np.eye(3)   # low-frequency pose

        self.reset()

    @staticmethod
    def _wrap(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _J(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    # dJ/dpsi
    @staticmethod
    def _dJ_dpsi(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [-s, -c, 0.0],
            [ c, -s, 0.0],
            [0.0, 0.0, 0.0],
        ])

    # dJ^T/dpsi
    @staticmethod
    def _dJT_dpsi(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [-s,  c, 0.0],
            [-c, -s, 0.0],
            [0.0, 0.0, 0.0],
        ])

    def reset(self, eta0=None):
        """Reset estimates and covariance before each simulation."""
        self.x = np.zeros(15)

        if eta0 is not None:
            eta0 = np.asarray(eta0, dtype=float).reshape(6)
            self.x[6:9] = eta0[self.IDX]

        # Initial covariance: high uncertainty in velocity and bias.
        p0_diag = np.array(
            [1.0, 1.0, 0.1,       # xi
             1.0, 1.0, 0.1,       # xi_dot
             0.1, 0.1, 0.02,      # eta
             1.0, 1.0, 0.1,       # nu
             1e8, 1e8, 1e10]      # bias [units are N and Nm]
        )
        self.P = np.diag(p0_diag)

        self.initialized = eta0 is not None

    def _continuous_model(self, x, tau):
        """Return f(x, tau) and the continuous-time Jacobian F."""
        xi = x[0:3]
        xi_dot = x[3:6]
        eta = x[6:9]
        nu = x[9:12]
        bias = x[12:15]

        psi = eta[2]
        J = self._J(psi)
        JT = J.T

        # Second-order wave-frequency oscillator.
        w = self.omega_w
        zeta = self.zeta_w

        # dynamics model - time continous
        f = np.zeros(15)
        f[0:3] = xi_dot
        f[3:6] = -(w**2) * xi - 2.0 * zeta * w * xi_dot         # xi''
        f[6:9] = J @ nu                                         # Low-frequency vessel kinematics in NED.
        f[9:12] = self.M_inv @ (-self.D @ nu + JT @ bias + tau) # vessel dynamics
        f[12:15] = -bias / self.T_bias                          # slowly varying bias in NED.

        # Analytic continuous-time Jacobian.
        F = np.zeros((15, 15))        
        F[0:3, 3:6] = np.eye(3)                                 # wave oscillator
        F[3:6, 0:3] = -(w**2) * np.eye(3)
        F[3:6, 3:6] = -2.0 * zeta * w * np.eye(3)
        F[6:9, 8] = self._dJ_dpsi(psi) @ nu                     # eta_dot = J(psi) @ nu
        F[6:9, 9:12] = J
        F[9:12, 8] = (self.M_inv @ self._dJT_dpsi(psi) @ bias)  # nu_dot = ...
        F[9:12, 9:12] = -self.M_inv_D
        F[9:12, 12:15] = self.M_inv @ JT
        F[12:15, 12:15] = -np.diag(1.0 / self.T_bias)           # bias decay

        return f, F

    def _process_covariance(self, dt):
        """Discrete process covariance for the Euler model."""
        q_diag = np.array(
            [self.q_w_pose] * 3
            + [self.q_w_rate] * 3
            + self.q_eta.tolist()
            + self.q_nu.tolist()
            + [self.q_bias] * 3
        )
        return np.diag(q_diag * dt)     # Q_D = Q_c * dt, 15x15 covariance matrix

    def step(self, t, dt, eta_measured, tau_est):
        dt = float(dt)

        eta_measured = np.asarray(eta_measured, dtype=float).reshape(6)
        tau_est = np.asarray(tau_est, dtype=float).reshape(6)
        y = eta_measured[self.IDX]
        tau = tau_est[self.IDX]

        # Initialise the LF pose from the first available measurement.
        if not self.initialized:
            self.x[6:9] = y
            self.initialized = True

        # 1. PREDICTION
        f, F = self._continuous_model(self.x, tau)

        x_pred = self.x + dt * f
        x_pred[8] = self._wrap(x_pred[8])

        Fd = np.eye(15) + dt * F
        Qd = self._process_covariance(dt)

        P_pred = Fd @ self.P @ Fd.T + Qd
        P_pred = 0.5 * (P_pred + P_pred.T)          # secures that P:pred is symmentric

        # 2. MEASUREMENT CORRECTION
        y_pred = self.H @ x_pred
        innovation = y - y_pred
        innovation[2] = self._wrap(innovation[2])

        # K = P_pred H.T inv(S), calculated without explicit inverse.
        S = self.H @ P_pred @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ P_pred).T

        self.x = x_pred + K @ innovation
        self.x[8] = self._wrap(self.x[8])

        # Joseph covariance update for numerical robustness.
        I_KH = np.eye(15) - K @ self.H
        self.P = (I_KH @ P_pred @ I_KH.T+ K @ self.R @ K.T)
        self.P = 0.5 * (self.P + self.P.T)

        # 3. RETURN THE FULL 6-DOF INTERFACE
        eta_out = np.zeros(6)
        nu_out = np.zeros(6)
        bias_out = np.zeros(6)

        eta_out[self.IDX] = self.x[6:9]
        nu_out[self.IDX] = self.x[9:12]
        bias_out[self.IDX] = self.x[12:15]

        return ObserverEstimate(
            eta=eta_out,
            nu=nu_out,
            bias=bias_out,
            wave_pose=self._expand_reduced(self.x[0:3]),
        )

    @classmethod
    def _expand_reduced(cls, value):
        """Expand a reduced [N, E, psi] vector into the 6-DOF interface."""
        out = np.zeros(6)
        out[cls.IDX] = value
        return out


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

