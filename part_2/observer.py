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
    """Nonlinear Passive Observer for 3-DOF DP."""

    name = "nonlinear_passive"
    IDX = np.array([0, 1, 5])  # North, East og heading

    def __init__(
        self,
        omega_w=2.0 * np.pi / 8.0,
        zeta_w=0.1,
        zeta_desired=1.0,
        notch_factor=2.0,
        T_bias=(300.0, 300.0, 300.0),
        k4=(8000.0, 8000.0, 1.5e6),
        bias_gain_ratio=0.1,
    ):

        # Parametere for bølgefilteret
        self.omega_w = float(omega_w)
        self.zeta_w = float(zeta_w)
        self.zeta_desired = float(zeta_desired)
        self.omega_c = notch_factor * self.omega_w

        # Tidskonstanter og gains for bias-estimering
        self.T_bias = np.asarray(T_bias, dtype=float)
        self.k4 = np.asarray(k4, dtype=float)
        self.bias_gain_ratio = float(bias_gain_ratio)

        if not (self.omega_w > 0 and
                0 < self.zeta_w < self.zeta_desired and
                notch_factor > 1):
            raise ValueError("Invalid wave-filter parameters.")

        if np.any(self.T_bias <= 0) or np.any(self.k4 <= 0):
            raise ValueError("T_bias and k4 must be positive.")

        if not (0 < self.bias_gain_ratio < self.omega_w):
            raise ValueError("Invalid bias gain ratio.")

        # Massematrise for Gunnerus fra prosjektbeskrivelsen
        M_data = np.array([
            [6.007e5, 0.0, 0.0],
            [0.0, 7.067e5, -4.733e5],
            [0.0, -5.712e5, 5.456e7]
        ])

        # Symmetrisering for passivitetsbasert design
        self.M = 0.5 * (M_data + M_data.T)

        if np.min(np.linalg.eigvalsh(self.M)) <= 0:
            raise ValueError("Mass matrix must be positive definite.")

        # Lineær dempningsmatrise
        self.D = np.diag([1.118e3, 2.229e4, 1.950e6])

        I = np.eye(3)
        Z = np.zeros((3, 3))
        w = self.omega_w

        # Bølgemodell: ligning (7.3) og (7.4)
        self.Aw = np.block([
            [Z, I],
            [-w**2 * I, -2 * self.zeta_w * w * I]
        ])

        # Henter bølgebevegelsen fra bølgetilstandene
        self.Cw = np.hstack((Z, I))

        # Observer gains etter ligning (7.32)-(7.34)
        dzeta = self.zeta_desired - self.zeta_w

        self.K1 = np.vstack((
            -2 * dzeta * (self.omega_c / w) * I,
            2 * w * dzeta * I
        ))

        self.K2 = self.omega_c * I

        # Gains for bias og hastighetskorreksjon
        self.K4 = np.diag(self.k4)
        self.K3 = self.bias_gain_ratio * self.K4

        # Full passivitet krever også kontroll av KYP-betingelsen
        self.reset()

    @staticmethod
    def _wrap(angle):
        # Holder heading mellom -pi og pi
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _J(psi):
        # Rotasjonsmatrise fra body til NED
        c = np.cos(psi)
        s = np.sin(psi)

        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

    def reset(self, eta0=None):
        # Nullstiller observerens 15 tilstander
        self.xi = np.zeros(6)      # Bølgebevegelse
        self.eta = np.zeros(3)     # LF-posisjon og heading
        self.nu = np.zeros(3)      # Hastighet i body-frame
        self.bias = np.zeros(3)    # Miljøkrefter i NED

        self.initialized = eta0 is not None

        # Initialiserer med skipets startposisjon
        if eta0 is not None:
            eta0 = np.asarray(eta0, dtype=float).reshape(6)
            self.eta = eta0[self.IDX].copy()
            self.eta[2] = self._wrap(self.eta[2])

    def _derivatives(self, y, tau):
        # Predikert måling: LF-posisjon + bølgebevegelse
        y_hat = self.eta + self.Cw @ self.xi

        # Innovasjon: forskjell mellom målt og estimert posisjon
        y_tilde = y - y_hat
        y_tilde[2] = self._wrap(y_tilde[2])

        J = self._J(self.eta[2])

        # Bølgetilstander: ligning (7.18a)
        dxi = self.Aw @ self.xi + self.K1 @ y_tilde

        # Lavfrekvent posisjon: ligning (7.18b)
        deta = J @ self.nu + self.K2 @ y_tilde

        # Langsomt varierende bias: ligning (7.18c)
        dbias = (
            -self.bias / self.T_bias
            + self.K3 @ y_tilde
        )

        # Hastighetsdynamikk: ligning (7.18d)
        # G = 0 fordi skipet ikke har mooring stiffness
        wrench = (
            -self.D @ self.nu
            + J.T @ self.bias
            + tau
            + J.T @ self.K4 @ y_tilde
        )

        # Løser M * dnu = wrench
        dnu = np.linalg.solve(self.M, wrench)

        return dxi, deta, dnu, dbias

    def step(self, t, dt, eta_measured, tau_est):
        # Sjekker tidssteget
        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite.")

        # Henter North, East og heading fra målingene
        y = np.asarray(
            eta_measured, dtype=float
        ).reshape(6)[self.IDX].copy()

        # Henter surge, sway og yaw-moment fra forrige steg
        tau = np.asarray(
            tau_est, dtype=float
        ).reshape(6)[self.IDX].copy()

        if not np.all(np.isfinite(y)):
            raise ValueError("Invalid pose measurement.")

        if not np.all(np.isfinite(tau)):
            raise ValueError("Invalid wrench input.")

        # Initialiserer ved første måling hvis nødvendig
        if not self.initialized:
            self.eta = y.copy()
            self.initialized = True

        # Beregner observerens differensialligninger
        dxi, deta, dnu, dbias = self._derivatives(y, tau)

        # Integrerer tilstandene med Forward Euler
        self.xi += dt * dxi
        self.eta += dt * deta
        self.nu += dt * dnu
        self.bias += dt * dbias

        # Sørger for korrekt heading
        self.eta[2] = self._wrap(self.eta[2])

        # Konverterer estimatene tilbake til 6-DOF
        eta_out = np.zeros(6)
        nu_out = np.zeros(6)
        bias_out = np.zeros(6)

        eta_out[self.IDX] = self.eta
        nu_out[self.IDX] = self.nu
        bias_out[self.IDX] = self.bias

        # Returnerer estimert posisjon, hastighet og bias
        return ObserverEstimate(
            eta=eta_out,
            nu=nu_out,
            bias=bias_out
        )


class KalmanFilterObserver(Observer):
    """15-state nonlinear EKF

    State ordering (all angles in rad, all forces in SI):
        x[ 0: 3] = xi_p   = [N_w, E_w, psi_w]   wave-frequency pose
        x[ 3: 6] = xi_v   = d(xi_p)/dt          wave-frequency pose rate
        x[ 6: 9] = eta    = [N, E, psi]        low-frequency NED pose
        x[ 9:12] = nu     = [u, v, r]         LF BODY velocity
        x[12:15] = b      = [b_N,b_E,b_psi]   slowly varying NED bias

    Continuous model:
        dot(xi_p) = xi_v
        dot(xi_v) = -Omega^2 xi_p - 2*zeta*Omega xi_v + w_wave
        dot(eta)  = J(psi) nu
        dot(nu)   = M^-1 (-D nu + J(psi)^T b + tau_previous)
        dot(b)    = -T_b^-1 b + w_bias
        y         = eta + xi_p + v

    Forward Euler for state, F_k = I + dt * df/dx for covariance,
    Q_k = dt * Q_c (white continuous-time process noise spectral density).
    Corrected covariance is updated using the Joseph form.
    """

    name = "kalman"

    # Observer model coefficients 
    DEFAULT_M = np.array([
        [6.007e5, 0.0,      0.0],
        [0.0,     7.067e5, -4.733e5],
        [0.0,    -5.712e5,  5.456e7],
    ])
    DEFAULT_D = np.diag([1.118e3, 2.229e4, 1.950e6])

    @staticmethod
    def _wrap(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _rotation(psi: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]])

    @staticmethod
    def _drotation(psi: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[-s, -c, 0.0],
                         [ c, -s, 0.0],
                         [0.0, 0.0, 0.0]])

    def __init__(
        self,
        *,
        tp: float = 8.0,
        wave_zeta: float = 0.15,
        wave_std: tuple[float, float, float] = (0.30, 0.30, np.deg2rad(1.0)),
        bias_tau: tuple[float, float, float] = (600.0, 600.0, 600.0),
        bias_std: tuple[float, float, float] = (1.0e4, 1.0e4, 1.0e5),
        acceleration_noise: tuple[float, float, float] = (0.005, 0.005, np.deg2rad(0.01)),
        measurement_std: tuple[float, float, float] = (0.0, 0.0, 0.0),
        eta0_std: tuple[float, float, float] = (1.0, 1.0, np.deg2rad(3.0)),
        nu0_std: tuple[float, float, float] = (0.05, 0.05, np.deg2rad(0.15)),
        M: np.ndarray | None = None,
        D: np.ndarray | None = None,
    ):
        self.M = np.asarray(self.DEFAULT_M if M is None else M, dtype=float).reshape(3, 3).copy()
        self.D = np.asarray(self.DEFAULT_D if D is None else D, dtype=float).reshape(3, 3).copy()
        self.M_inv = np.linalg.inv(self.M)
        self.MinvD = self.M_inv @ self.D

        self.omega = 2.0 * np.pi / float(tp)
        self.wave_zeta = float(wave_zeta)
        self.wave_std = np.asarray(wave_std, dtype=float).reshape(3).copy()
        self.bias_tau = np.asarray(bias_tau, dtype=float).reshape(3).copy()
        self.bias_std = np.asarray(bias_std, dtype=float).reshape(3).copy()
        accel_std = np.asarray(acceleration_noise, dtype=float).reshape(3)
        meas_std = np.asarray(measurement_std, dtype=float).reshape(3)
        eta0_std = np.asarray(eta0_std, dtype=float).reshape(3)
        nu0_std = np.asarray(nu0_std, dtype=float).reshape(3)

        self.omega2 = np.full(3, self.omega ** 2)
        self.wave_damping = np.full(3, 2.0 * self.wave_zeta * self.omega)

        # Output y = eta + C_w xi, where xi = [xi_p, xi_v] and C_w = [I, 0].
        self.Cw = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.H = np.zeros((3, 15))
        self.H[:, 0:3] = np.eye(3)
        self.H[:, 6:9] = np.eye(3)
        self.I15 = np.eye(15)

        # Continuous-time WHITE-NOISE intensity Qc (not a per-step covariance).
        # The oscillator acceleration intensity is chosen so that its stationary
        # displacement standard deviation is wave_std for the chosen zeta/omega:
        # q_wave = 4*zeta*omega**3*wave_std**2.
        # Ornstein-Uhlenbeck b has q_bias = 2*bias_std**2 / bias_tau.
        self.Qc = np.zeros((15, 15))
        self.Qc[3:6, 3:6] = np.diag(4.0 * self.wave_zeta * self.omega**3 * self.wave_std**2)
        self.Qc[9:12, 9:12] = np.diag(accel_std**2)
        self.Qc[12:15, 12:15] = np.diag(2.0 * self.bias_std**2 / self.bias_tau)

        # Mandatory simulations use ideal pose measurements -> R = 0.
        # For prescribed extra-credit GNSS/gyro noise, pass:
        # measurement_std=(0.20, 0.20, np.deg2rad(0.20)).
        self.R = np.diag(meas_std**2)

        self.P0 = np.diag(np.r_[
            self.wave_std**2,
            (self.omega * self.wave_std)**2,
            eta0_std**2,
            nu0_std**2,
            self.bias_std**2,
        ])
        self.reset()

    def reset(self, eta0: np.ndarray | None = None):
        """Reset for a simulation run; initialize LF pose from TRUE eta0."""
        self.x = np.zeros(15)
        if eta0 is not None:
            pose = np.asarray(eta0, dtype=float).reshape(6)
            self.x[6:9] = pose[[0, 1, 5]]
            self.x[8] = self._wrap(self.x[8])
        self.P = self.P0.copy()
        self.Qk = None  # Set at each step because dt may be 0.1 s or 0.01 s
        self.last_innovation = np.zeros(3)

    def _dynamics_and_jacobian(self, x: np.ndarray, tau: np.ndarray):
        """Continuous drift f(x,tau) and its exact analytic Jacobian A=df/dx."""
        xi_p = x[0:3]
        xi_v = x[3:6]
        psi = x[8]
        nu = x[9:12]
        b = x[12:15]

        J = self._rotation(psi)
        dJ = self._drotation(psi)
        f = np.zeros(15)
        f[0:3] = xi_v
        f[3:6] = -self.omega2 * xi_p - self.wave_damping * xi_v
        f[6:9] = J @ nu
        f[9:12] = self.M_inv @ (-self.D @ nu + J.T @ b + tau)
        f[12:15] = -b / self.bias_tau

        A = np.zeros((15, 15))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = -np.diag(self.omega2)
        A[3:6, 3:6] = -np.diag(self.wave_damping)
        A[6:9, 9:12] = J
        A[6:9, 8] = dJ @ nu
        A[9:12, 9:12] = -self.MinvD
        A[9:12, 12:15] = self.M_inv @ J.T
        A[9:12, 8] = self.M_inv @ (dJ.T @ b)
        A[12:15, 12:15] = -np.diag(1.0 / self.bias_tau)
        return f, A

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        dt = float(dt)
        eta6 = np.asarray(eta_measured, dtype=float).reshape(6)
        tau6 = np.asarray(tau_est, dtype=float).reshape(6)
        y = eta6[[0, 1, 5]]  # NED pose, measured; NEVER read measured velocity
        tau = tau6[[0, 1, 5]]  # BODY previous desired wrench (N,N,Nm)

        # 1) Nonlinear time update (explicit Euler), using tau_c[k-1].
        f, A = self._dynamics_and_jacobian(self.x, tau)
        x_pred = self.x + dt * f
        x_pred[8] = self._wrap(x_pred[8])
        F = self.I15 + dt * A
        self.Qk = dt * self.Qc
        P_pred = F @ self.P @ F.T + self.Qk
        P_pred = 0.5 * (P_pred + P_pred.T)

        # 2) Wrapped innovation, y_hat = eta_LF + eta_WF.
        innovation = y - self.H @ x_pred
        innovation[2] = self._wrap(innovation[2])
        S = self.H @ P_pred @ self.H.T + self.R
        S = 0.5 * (S + S.T)
        # K = P_pred H^T S^-1, solved as a 3x3 linear system.
        K = np.linalg.solve(S, self.H @ P_pred).T

        # 3) Measurement correction.  Joseph form preserves covariance PSD.
        self.x = x_pred + K @ innovation
        self.x[8] = self._wrap(self.x[8])
        IKH = self.I15 - K @ self.H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_innovation = innovation.copy()

        # Expand 3DOF estimates to the simulator's expected 6DOF layout:
        # [surge/sway/heave/roll/pitch/yaw] with unused channels equal zero.
        wave_pose_out = np.zeros(6)
        eta_out = np.zeros(6)
        nu_out = np.zeros(6)
        bias_out = np.zeros(6)
        wave_pose_out[[0, 1, 5]] = self.x[0:3]
        eta_out[[0, 1, 5]] = self.x[6:9]
        nu_out[[0, 1, 5]] = self.x[9:12]
        bias_out[[0, 1, 5]] = self.x[12:15]
        return ObserverEstimate(
            eta=eta_out,
            nu=nu_out,
            bias=bias_out,
            wave_pose=wave_pose_out
        )


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