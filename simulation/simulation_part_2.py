"""Independent Part 2 DP simulation wiring.

Gunnerus and generic actuator utilities are shared with Part 1, but every
student-editable subsystem is imported from or injected from ``part_2``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from models.gunnerus_3dof import Gunnerus3DOF
from models.sensors import SensorConfig, VesselSensors
from models.thruster_dynamics import ThrusterConfig, ThrusterSet
from part_2.config import Part2SimConfig
from part_2.controller import DPController
from part_2.current import Current
from part_2.observer import Observer, ObserverEstimate, select_observer
from part_2.reference import ReferenceModel
from part_2.thrust_allocation import ThrustAllocator
from part_2.wind import Wind
from simulation.plant import LoadInput, body_wrench
from simulation.utils import Rz, to_3dof, to_6dof


@dataclass
class Part2Logs:
    # Common fields intentionally match Part 1 so shared plotters still work.
    t: np.ndarray
    eta: np.ndarray
    nu: np.ndarray
    eta_measured: np.ndarray
    nu_measured: np.ndarray
    eta_est: np.ndarray
    nu_est: np.ndarray
    bias_est: np.ndarray
    sp: np.ndarray
    cmd: np.ndarray
    tau_d: np.ndarray
    tau_thr: np.ndarray
    tau_total: np.ndarray
    tau_w6: np.ndarray
    tau_wave: np.ndarray
    u_cmd: np.ndarray
    alpha_cmd: np.ndarray
    u: np.ndarray
    alpha: np.ndarray
    cur_body: np.ndarray
    Uc: np.ndarray
    beta_c: np.ndarray
    U_w: np.ndarray
    alpha_w: np.ndarray
    beta_w: np.ndarray
    U_w_mean: np.ndarray        # mean + slowly varying wind speed (info["U_mean"])
    U_w_gust: np.ndarray        # gust component (info["U_gust"])
    thruster_names: list[str]
    observer_name: str
    # Compatibility/optional controller diagnostics.
    I_ned: np.ndarray
    I_psi: np.ndarray
    tau_P: np.ndarray
    tau_I: np.ndarray
    tau_D: np.ndarray
    nu_ref: np.ndarray
    acc_ref: np.ndarray


class DPSimulatorPart2:
    """Part 2 signal chain.

    ``sensor -> selected observer -> controller -> allocation -> constrained
    thrusters -> Gunnerus`` with current, detailed wind, and wave loads acting
    on the vessel.  Observer input uses the previous controller desired wrench
    (the ``tau_est`` signal before thruster dynamics): at step ``k`` the
    observer receives ``tau_d[k-1]`` (zero at ``k = 0``).

    The observer receives the measured POSE only — velocities are never
    measured; the observer must estimate them from position and heading.
    With ``use_observer=False`` the loop instead feeds the exact simulated
    state (including velocity) back to the controller, an idealisation kept
    for the no-observer baseline simulations.

    ``reset_state`` calls ``observer.reset(eta0)`` with the TRUE initial pose,
    so the observer starts from the vessel's actual position (there is no
    convergence transient from a wrong initial estimate); the same is done for
    the reference model.  The observer is built as
    ``select_observer(cfg.observer_type, **cfg.observer_kwargs)`` when none is
    injected, so a retuned observer can be selected from the configuration.
    """

    def __init__(
        self,
        cfg: Part2SimConfig,
        controller: DPController,
        thrusters: list[ThrusterConfig],
        *,
        reference: ReferenceModel | None = None,
        allocator: ThrustAllocator | None = None,
        sensors: VesselSensors | None = None,
        observer: Observer | None = None,
        pkl_path: str | None = None,
    ):
        if cfg.dt not in (0.01, 0.1):
            raise ValueError("Part 2 requires fixed dt=0.01 or dt=0.1 s")
        self.cfg = cfg
        self.controller = controller
        self.vessel = Gunnerus3DOF(cfg.dt, cfg.method, pkl_path)
        self.thrusters = ThrusterSet(thrusters, dynamics=cfg.thruster_dynamics)
        self.allocator = allocator or ThrustAllocator(thrusters)
        self.sensors = sensors or VesselSensors(
            SensorConfig(noise=cfg.use_sensor_noise)
        )
        self.reference = reference if cfg.use_reference else None
        if cfg.use_reference and self.reference is None:
            self.reference = ReferenceModel(dt=cfg.dt)
        self.observer = observer
        if cfg.use_observer and self.observer is None:
            self.observer = select_observer(cfg.observer_type,
                                            **(getattr(cfg, "observer_kwargs", None) or {}))
        self.thruster_names = [x.name for x in thrusters]
        self.reset_state()

    def reset_state(self, eta0=None, nu0=None):
        self.vessel.set_eta(np.zeros(3) if eta0 is None else to_3dof(eta0))
        self.vessel.set_nu(np.zeros(3) if nu0 is None else to_3dof(nu0))
        self.thrusters.reset()
        self.sensors.reset()
        eta6 = to_6dof(self.vessel.get_eta())
        for component in (self.controller, self.reference, self.observer):
            if component is not None and hasattr(component, "reset"):
                try:
                    component.reset(eta6)
                except TypeError:
                    component.reset()

    def run(self, eta_cmd, *, current=None, wind=None, waves: LoadInput = None):
        cfg = self.cfg
        current = current or Current()
        wind = wind or Wind()
        n = int(round(cfg.T / cfg.dt)) + 1
        t = np.arange(n) * cfg.dt
        six = lambda: np.zeros((n, 6))
        eta_h, nu_h, eta_mh, nu_mh = six(), six(), six(), six()
        eta_eh, nu_eh, bias_h = six(), six(), six()
        sp_h, cmd_h, tau_dh, tau_th, tau_total = six(), six(), six(), six(), six()
        tau_wh, tau_waveh = six(), six()
        tau_P, tau_I, tau_D = six(), six(), six()
        nu_ref_h, acc_ref_h = np.full((n, 6), np.nan), np.full((n, 6), np.nan)
        nth = len(self.thruster_names)
        u_cmd_h, a_cmd_h = np.zeros((n, nth)), np.zeros((n, nth))
        u_h, a_h = np.zeros((n, nth)), np.zeros((n, nth))
        cur_body, Uc, beta_c = np.zeros((n, 2)), np.zeros(n), np.zeros(n)
        Uw, alpha_w, beta_w = np.zeros(n), np.zeros(n), np.zeros(n)
        Uw_mean, Uw_gust = np.zeros(n), np.zeros(n)
        I_ned, I_psi = np.zeros((n, 2)), np.zeros(n)
        tau_est = np.zeros(6)
        commands = np.asarray(eta_cmd, dtype=float)
        if commands.shape not in ((6,), (n, 6)):
            raise ValueError(
                "eta_cmd must be a (6,) set-point or an (N_steps, 6) time series with "
                f"N_steps = round(T/dt) + 1 = {n}; got shape {commands.shape}")

        for k, tk in enumerate(t):
            eta = to_6dof(self.vessel.get_eta())
            nu = to_6dof(self.vessel.get_nu())
            eta_m, nu_m = self.sensors.measure(eta, nu)
            eta_h[k], nu_h[k] = eta, nu
            eta_mh[k], nu_mh[k] = eta_m, nu_m

            if cfg.use_observer:
                estimate = self.observer.step(tk, cfg.dt, eta_m, tau_est)
                if not isinstance(estimate, ObserverEstimate):
                    estimate = ObserverEstimate(*estimate)
                eta_ctrl, nu_ctrl = estimate.eta, estimate.nu
                bias_h[k] = estimate.bias
            else:
                eta_ctrl, nu_ctrl = eta_m, nu_m
            eta_eh[k], nu_eh[k] = eta_ctrl, nu_ctrl

            cmd = commands.reshape(6) if commands.ndim == 1 else commands[k].reshape(6)
            cmd_h[k] = cmd
            if self.reference is None:
                sp, nu_ref, acc_ref = cmd, np.zeros(6), np.zeros(6)
            else:
                sp, nu_ref, acc_ref = self.reference.step(tk, cfg.dt, cmd)
                # Shared velocity plots expect BODY u/v, while the reference
                # model and controller interface use NED Ndot/Edot.
                uv_body = Rz(eta_ctrl[5]).T @ np.asarray(nu_ref)[:3]
                nu_ref_h[k] = 0.0
                nu_ref_h[k, 0:2] = uv_body[0:2]
                nu_ref_h[k, 5] = nu_ref[5]
                acc_ref_h[k] = acc_ref
            sp_h[k] = sp

            if cfg.use_controller:
                tau_d = np.asarray(self.controller.compute(
                    tk, cfg.dt, eta_ctrl, nu_ctrl, sp, nu_ref, acc_ref
                ), dtype=float).reshape(6)
            else:
                tau_d = np.zeros(6)                 # DP off: free drift
            tau_est = tau_d.copy()
            tau_dh[k] = tau_d
            pid = getattr(self.controller, "last_pid_body", None)
            if pid is not None:
                tau_P[k], tau_I[k], tau_D[k] = pid["P"], pid["I"], pid["D"]

            if cfg.bypass_actuators:
                u_cmd = u = np.zeros(nth)
                a_cmd = a = self.thrusters.get_angles()
                tau_thr = tau_d
            else:
                u_cmd, a_cmd = self.allocator.allocate(
                    tk, cfg.dt, tau_d, self.thrusters.get_thrusts(),
                    self.thrusters.get_angles()
                )
                u, a, tau3 = self.thrusters.step(u_cmd, a_cmd, cfg.dt)
                tau_thr = to_6dof(tau3)
            u_cmd_h[k], a_cmd_h[k], u_h[k], a_h[k] = u_cmd, a_cmd, u, a
            tau_th[k] = tau_thr
            if hasattr(self.controller, "apply_external_aw"):
                self.controller.apply_external_aw(tau_thr, psi=eta_ctrl[5], dt=cfg.dt)

            tau_w, info = wind.step(tk, cfg.dt, eta, nu)
            tau_w = np.asarray(tau_w, dtype=float).reshape(6)
            info = info or {}
            tau_wave = body_wrench(waves, tk, eta, nu)
            tau_wh[k], tau_waveh[k] = tau_w, tau_wave
            Uw[k] = info.get("U", 0.0)
            alpha_w[k] = info.get("alpha_body", 0.0)
            beta_w[k] = info.get("beta_ned", 0.0)
            Uw_mean[k] = info.get("U_mean", Uw[k])
            Uw_gust[k] = info.get("U_gust", 0.0)
            total = tau_thr + tau_w + tau_wave
            tau_total[k] = total

            nc = np.asarray(current.step(tk, cfg.dt, eta, nu), dtype=float).reshape(6)
            Uc[k], beta_c[k] = np.hypot(nc[0], nc[1]), np.arctan2(nc[1], nc[0])
            cur_body[k] = (Rz(eta[5]).T @ nc[:3])[:2]
            self.vessel.integrate(Uc=Uc[k], beta_c=beta_c[k], tau=to_3dof(total))
            I_ned[k] = getattr(self.controller, "int_ned", np.zeros(2))
            I_psi[k] = getattr(self.controller, "int_psi", 0.0)

        return Part2Logs(
            t, eta_h, nu_h, eta_mh, nu_mh, eta_eh, nu_eh, bias_h,
            sp_h, cmd_h, tau_dh, tau_th, tau_total, tau_wh, tau_waveh,
            u_cmd_h, a_cmd_h, u_h, a_h, cur_body, Uc, beta_c, Uw, alpha_w,
            beta_w, Uw_mean, Uw_gust, self.thruster_names,
            self.observer.name if cfg.use_observer else "off",
            I_ned, I_psi, tau_P, tau_I, tau_D, nu_ref_h, acc_ref_h,
        )
