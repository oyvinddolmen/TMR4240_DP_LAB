"""Automated checks for the Part 2 subsystems (public, requirement-level).

Same statuses and reporting as ``simulation/checks.py`` (Part 1):

* ``PASS`` / ``FAIL``      — the subsystem ran and met / missed the criteria,
* ``NOT IMPLEMENTED``      — the template placeholder is still in place,
* ``ERROR``                — an exception, or the template interface was not
                             followed (the message says what to fix).

Exposed through ``python check.py --part 2``, ``pytest`` (``tests/test_part_2.py``)
and ``notebooks/part_2_demo.ipynb``.

These checks verify *stated requirements and interfaces* only — constructor
contracts, finite outputs, sign conventions, heading wrapping, the actuator
limits, the wind-field requirements of the project text (gust present and
zero-mean, direction variation bounded to +/-5 degrees), and that an observer
actually estimates (its output is not the raw measurement; velocity comes
from the pose — the vessel has no velocity sensor, and the infrastructure
tests in ``tests/test_part_2.py`` fail if velocity measurements are ever
wired into the observer).  They do not grade
performance: how well the observer filters, how fast the estimates converge
and how accurately the closed loop keeps position are assessed in the report
and by the staff.  Passing them therefore means "the interfaces are right and
the design meets the explicit requirements", not "the design is good".

Current, controller and reference model are checked exactly as in Part 1 but
on the ``part_2`` modules.  Constructor contracts are documented in the
``part_2`` templates — keep those signatures working.  Angles are radians,
NED, 0 = North, pi/2 = East.
"""
from __future__ import annotations

import numpy as np

from models.thruster_dynamics import ThrusterSet
from part_2.config import default_thrusters_part_2
from .checks import CheckResult, _Criteria, _state6, _wrench6, wrap_angle_pi

DT = 0.1                    # Part 2 runs at a fixed 0.1 s (or 0.01 s) step
DIR_LIMIT = np.deg2rad(5.0)  # project requirement on the wind-direction excursion


def _as_estimate(est):
    """Accept an ``ObserverEstimate`` or a plain ``(eta, nu, bias)`` tuple —
    the simulator accepts both, so the checks do too."""
    if hasattr(est, "eta"):
        return est
    from part_2.observer import ObserverEstimate
    return ObserverEstimate(*est)


# ---------------------------------------------------------------------------
# Part 1 subsystem checks re-run on the part_2 modules
# ---------------------------------------------------------------------------

def check_current() -> CheckResult:
    """Current model (Part 2 copy): 'from' convention and the rotation profile."""
    from part_2.current import Current

    name = "current (part_2): direction conventions"
    try:
        c = Current(0.5, np.pi / 2, semantics="from")
        v = np.asarray(c.step(0.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    except TypeError as exc:
        return CheckResult(name, "ERROR", [
            f"constructor/step does not follow the template contract: {exc}",
            "expected Current(speed, beta, semantics=...) — see part_2/current.py",
        ])
    if np.allclose(v, 0.0):
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["Current.step() still returns the zero placeholder"])
    crit = _Criteria()
    crit.add("from east -> flows west: [V_N, V_E] = [0, -0.5]",
             np.allclose(v[:2], [0.0, -0.5], atol=1e-6), f"got {np.round(v[:2], 3)}")
    crit.add("components 2-5 of the generalized vector are zero",
             np.allclose(v[2:], 0.0), f"got {np.round(v[2:], 3)}")
    cp = Current(0.5, 0.0, semantics="from", beta_end=np.pi / 2, duration=300.0)
    v0 = np.asarray(cp.step(0.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    v1 = np.asarray(cp.step(300.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    crit.add("rotation start: from north -> [-0.5, 0]",
             np.allclose(v0[:2], [-0.5, 0.0], atol=1e-3), f"got {np.round(v0[:2], 3)}")
    crit.add("rotation end: from east -> [0, -0.5]",
             np.allclose(v1[:2], [0.0, -0.5], atol=1e-3), f"got {np.round(v1[:2], 3)}")
    return crit.result(name)


def check_controller() -> CheckResult:
    """Controller (Part 2 copy): error signs under rotation and heading wrap."""
    from part_2.controller import DPController

    name = "controller (part_2): sign tests and heading wrap"

    def wrench(psi, dN=0.0, dE=0.0, psi_d=None):
        ctl = DPController()
        if hasattr(ctl, "reset"):
            ctl.reset()
        eta = _state6(psi)
        ref = eta.copy()
        ref[0] += dN
        ref[1] += dE
        if psi_d is not None:
            ref[5] = psi_d
        tau = ctl.compute(0.0, DT, eta, np.zeros(6), ref, np.zeros(6), np.zeros(6))
        return np.asarray(tau, float).reshape(6)

    try:
        tau_n = wrench(0.0, dN=10.0)
    except Exception as exc:                                    # noqa: BLE001
        return CheckResult(name, "ERROR", [f"compute() raised: {exc!r}"])
    if np.allclose(tau_n, 0.0):
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["compute() still returns the zero placeholder"])
    crit = _Criteria()
    crit.add("psi=0, error +10 m North: Fx > 0 and dominant",
             tau_n[0] > 0.0 and abs(tau_n[0]) > abs(tau_n[1]),
             f"tau = [{tau_n[0]/1e3:.1f}, {tau_n[1]/1e3:.1f}] kN")
    tau_r = wrench(np.pi / 2, dN=10.0)
    crit.add("psi=90 deg, error +10 m North: Fy < 0 and dominant",
             tau_r[1] < 0.0 and abs(tau_r[1]) > abs(tau_r[0]),
             f"tau = [{tau_r[0]/1e3:.1f}, {tau_r[1]/1e3:.1f}] kN")
    tau_w1 = wrench(np.deg2rad(170.0), psi_d=np.deg2rad(-170.0))
    crit.add("psi=+170 deg -> psi_d=-170 deg: Mz > 0 (shortest way)",
             tau_w1[5] > 0.0, f"Mz = {tau_w1[5]/1e3:.1f} kNm")
    tau_w2 = wrench(np.deg2rad(-170.0), psi_d=np.deg2rad(170.0))
    crit.add("psi=-170 deg -> psi_d=+170 deg: Mz < 0 (shortest way)",
             tau_w2[5] < 0.0, f"Mz = {tau_w2[5]/1e3:.1f} kNm")
    return crit.result(name)


def check_reference() -> CheckResult:
    """Reference model (Part 2 copy): smooth step response and heading wrap."""
    from part_2.reference import ReferenceModel

    name = "reference model (part_2): smooth step and heading wrap"
    rm = ReferenceModel(dt=DT)
    rm.reset(np.zeros(6))
    cmd = np.zeros(6)
    cmd[0] = 10.0
    eta_ref, _, _ = rm.step(0.0, DT, cmd)
    if abs(np.asarray(eta_ref, float).reshape(6)[0] - 10.0) < 0.1:
        return CheckResult(name, "NOT IMPLEMENTED", [
            "step() is still the pass-through placeholder"])
    n = int(600.0 / DT)
    N_ref = np.empty(n)
    N_ref[0] = np.asarray(eta_ref, float).reshape(6)[0]
    for k in range(1, n):
        eta_ref, _, _ = rm.step(k * DT, DT, cmd)
        N_ref[k] = np.asarray(eta_ref, float).reshape(6)[0]
    crit = _Criteria()
    crit.add("smooth start: reference below 1 m one second after a 10 m step",
             N_ref[int(1.0 / DT)] < 1.0,
             f"N_ref(1 s) = {N_ref[int(1.0/DT)]:.2f} m (too fast a reference, wn too high?)")
    crit.add("overshoot below 10% of the step", np.max(N_ref) < 11.0,
             f"max N_ref = {np.max(N_ref):.2f} m")
    crit.add("converges to the setpoint within 600 s", abs(N_ref[-1] - 10.0) < 0.2,
             f"final N_ref = {N_ref[-1]:.2f} m")
    rm2 = ReferenceModel(dt=DT)
    rm2.reset(_state6(3.0))
    psi_hist = np.empty(n)
    for k in range(n):
        eta_ref, _, _ = rm2.step(k * DT, DT, _state6(-3.0))
        psi_hist[k] = np.asarray(eta_ref, float).reshape(6)[5]
    travel = np.sum(np.abs([wrap_angle_pi(d) for d in np.diff(psi_hist)]))
    crit.add("heading 3.0 -> -3.0 rad goes the short way across pi",
             travel < 1.0, f"total travel {travel:.2f} rad (short way is 0.28)")
    return crit.result(name)


# ---------------------------------------------------------------------------
# Wind: Part 1 loads + the Part 2 requirements (gust, bounded direction, ramp)
# ---------------------------------------------------------------------------

def _wind_series(w, T: float, psi: float = 0.0):
    """Run a wind model for T seconds at rest; return Fx, U, U_mean, U_gust, beta."""
    n = int(T / DT)
    Fx, U, Um, Ug, beta = (np.empty(n) for _ in range(5))
    for k in range(n):
        tau, info = w.step(k * DT, DT, _state6(psi), np.zeros(6))
        info = info or {}
        Fx[k] = np.asarray(tau, float).reshape(6)[0]
        U[k], beta[k] = info.get("U", np.nan), info.get("beta_ned", np.nan)
        Um[k], Ug[k] = info.get("U_mean", np.nan), info.get("U_gust", np.nan)
    return Fx, U, Um, Ug, beta


def check_wind() -> CheckResult:
    """Wind model: Part 1 loads, gust present, bounded slow direction."""
    from part_2.wind import Wind, load_wind_coefficients

    name = "wind (part_2): loads, gust, direction variation"
    try:
        w = Wind(10.0, 0.0, semantics="from", seed=0)
        tau, info = w.step(0.0, DT, _state6(), np.zeros(6))
        tau = np.asarray(tau, float).reshape(6)
        Wind(10.0, 0.0, gust=True, gust_params=None, sigma_dir=0.01, tau_dir=300.0,
             dir_limit=DIR_LIMIT, seed=0)
    except TypeError as exc:
        return CheckResult(name, "ERROR", [
            f"constructor/step does not follow the template contract: {exc}",
            "expected Wind(mean_speed, beta, semantics=..., sigma_slow=..., gust=...,"
            " gust_params=..., sigma_dir=..., dir_limit=..., seed=...)"
            " — see part_2/wind.py",
        ])
    if np.allclose(tau, 0.0):
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["Wind.step() still returns the zero placeholder"])
    for key in ("U", "beta_ned", "U_mean", "U_gust"):
        if key not in (info or {}):
            return CheckResult(name, "ERROR", [
                f'info dict returned by step() lacks "{key}" — the Part 2 checks need '
                'U, beta_ned, U_mean and U_gust (see part_2/wind.py)'])

    crit = _Criteria()
    # --- Part 1 behaviour with gust off (loads follow the provided table) ---
    alpha_tab, C6_tab = load_wind_coefficients()
    Fx_exp = 10.0 ** 2 * np.interp(180.0, alpha_tab, C6_tab[:, 0])
    crit.add("gust off, head wind: Fx < 0 and |Fx| = U^2*Cx(180 deg) within 5%",
             tau[0] < 0.0 and abs(tau[0] - Fx_exp) <= 0.05 * abs(Fx_exp),
             f"Fx = {tau[0]/1e3:.2f} kN vs expected {Fx_exp/1e3:.2f} kN")
    we = Wind(10.0, np.pi / 2, semantics="from", seed=0)
    tau_e = np.asarray(we.step(0.0, DT, _state6(), np.zeros(6))[0], float).reshape(6)
    crit.add("gust off, wind from east at psi=0: Fy < 0 and dominant",
             tau_e[1] < 0.0 and abs(tau_e[1]) > abs(tau_e[0]),
             f"Fx = {tau_e[0]/1e3:.2f} kN, Fy = {tau_e[1]/1e3:.2f} kN")
    nu_fwd = np.zeros(6)
    nu_fwd[0] = 2.0
    tau_fwd = np.asarray(Wind(10.0, 0.0, semantics="from", seed=0)
                         .step(0.0, DT, _state6(), nu_fwd)[0], float).reshape(6)
    crit.add("relative wind: sailing into the wind increases |Fx|",
             tau_fwd[0] < tau[0],
             f"Fx = {tau_fwd[0]/1e3:.2f} kN moving vs {tau[0]/1e3:.2f} kN at rest")
    _, _, _, Ug0, beta0 = _wind_series(Wind(10.0, np.pi, semantics="from", seed=0), 120.0)
    crit.add("gust=False (default): U_gust is identically zero",
             np.allclose(np.nan_to_num(Ug0), 0.0),
             f"max |U_gust| = {np.nanmax(np.abs(Ug0)):.3f} m/s")
    crit.add("sigma_dir=0 (default): direction is constant",
             np.nanstd(beta0) < 1e-9, f"std(beta) = {np.rad2deg(np.nanstd(beta0)):.4f} deg")

    # --- Gust component (requirement: present, zero-mean, on top of U_mean) ---
    _, U, Um, Ug, _ = _wind_series(
        Wind(10.0, np.pi, semantics="from", sigma_slow=0.0, gust=True, seed=1), 600.0)
    ok_finite = np.all(np.isfinite(U)) and np.all(np.isfinite(Um)) and np.all(np.isfinite(Ug))
    crit.add("gust=True: U, U_mean, U_gust are finite", ok_finite)
    if ok_finite:
        crit.add("gust=True: U = U_mean + U_gust",
                 np.allclose(U, Um + Ug, atol=1e-6),
                 f"max |U - U_mean - U_gust| = {np.max(np.abs(U - Um - Ug)):.2e} m/s")
        g_std = float(np.std(Ug))
        crit.add("gust is present (std(U_gust) > 0)", g_std > 1e-6, f"std = {g_std:.2f} m/s")
        crit.add("gust is zero-mean (|mean| < std over 600 s)",
                 abs(np.mean(Ug)) < max(g_std, 1e-9), f"mean = {np.mean(Ug):.3f} m/s")

    # --- Slowly varying, bounded direction (requirement: within +/-5 deg) ---
    _, _, _, _, beta = _wind_series(
        Wind(10.0, np.pi, semantics="from", sigma_dir=np.deg2rad(3.0),
             tau_dir=300.0, dir_limit=DIR_LIMIT, seed=2), 1800.0)
    ok_b = np.all(np.isfinite(beta))
    crit.add("sigma_dir>0: beta_ned is finite", ok_b)
    if ok_b:
        # Constructor beta is the FROM direction (pi); logged beta_ned is
        # TOWARDS, so the nominal logged value is 0.  Wrap the excursion.
        exc = np.array([wrap_angle_pi(b - 0.0) for b in beta])
        crit.add("direction varies (std(beta) > 0)", np.std(exc) > 1e-6,
                 f"std = {np.rad2deg(np.std(exc)):.2f} deg")
        crit.add("direction excursion never exceeds dir_limit (5 deg)",
                 np.max(np.abs(exc)) <= DIR_LIMIT + 1e-6,
                 f"max |beta - beta_mean| = {np.rad2deg(np.max(np.abs(exc))):.2f} deg")

    return crit.result(name)


# ---------------------------------------------------------------------------
# Thrust allocation: wrenches reproduced, limits respected
# ---------------------------------------------------------------------------

def check_allocation() -> CheckResult:
    """Thrust allocation: isolated wrenches reproduced, commands within u_max.

    Contract tested here: ``allocate()`` is called repeatedly from the initial
    actuator state (azimuths at ``alpha0``); each command is applied through
    IDEAL actuators and the resulting actuator state is fed back as
    ``u_now`` / ``alpha_now``.  After 30 s of such calls the achieved wrench
    must equal the requested one.  A steady-state allocator therefore passes
    on its first call, and a rate-aware allocator (optional extension: angle
    commands limited to one step of rotation, thrust reduced while an azimuth
    turns) passes once it has converged.  Force and moment errors are judged
    separately (2 % of the request, or 0.5 kN / 5 kNm for a zero request).
    Shaping the transient beyond that is the job of the actuator model.
    """
    from part_2.thrust_allocation import ThrustAllocator

    name = "allocation (part_2): wrenches and thruster limits"
    cfgs = default_thrusters_part_2()
    u_max = np.array([c.u_max for c in cfgs])
    n_calls = int(round(30.0 / DT))          # a rate-aware allocator turns any azimuth within 30 s
    crit = _Criteria()
    cases = [
        ("pure surge 20 kN", _wrench6(Fx=20e3)),
        ("pure sway 20 kN", _wrench6(Fy=20e3)),
        ("pure yaw 200 kNm", _wrench6(Mz=200e3)),
        ("combined 10 kN / 10 kN / 100 kNm", _wrench6(10e3, 10e3, 100e3)),
    ]
    for i, (label, tau_d) in enumerate(cases):
        ts = ThrusterSet(cfgs, dynamics=False)
        alloc = ThrustAllocator(cfgs)
        within = True
        try:
            for k in range(n_calls):
                u_cmd, a_cmd = alloc.allocate(k * DT, DT, tau_d, u_now=ts.get_thrusts(),
                                              alpha_now=ts.get_angles())
                u_cmd = np.asarray(u_cmd, float)
                if i == 0 and k == 0 and np.allclose(u_cmd, 0.0):
                    return CheckResult(name, "NOT IMPLEMENTED",
                                       ["allocate() still returns the zero placeholder"])
                within = within and bool(np.all(np.abs(u_cmd) <= u_max + 1.0))
                _, _, tau_ach = ts.step(u_cmd, a_cmd, DT)
        except Exception as exc:                                # noqa: BLE001
            return CheckResult(name, "ERROR", [f"allocate() raised: {exc!r}"])
        f_req, m_req = tau_d[[0, 1]], tau_d[5]
        err_f = float(np.linalg.norm(tau_ach[:2] - f_req))
        err_m = float(abs(tau_ach[2] - m_req))
        crit.add(f"{label}: force reproduced after convergence (ideal actuators)",
                 err_f <= max(0.02 * np.linalg.norm(f_req), 500.0),
                 f"|dF| = {err_f/1e3:.2f} kN")
        crit.add(f"{label}: moment reproduced after convergence (ideal actuators)",
                 err_m <= max(0.02 * abs(m_req), 5e3),
                 f"|dM| = {err_m/1e3:.2f} kNm")
        crit.add(f"{label}: |u_cmd| within u_max at every call", within,
                 f"final u_cmd = {np.round(u_cmd/1e3, 1)} kN")
    # Infeasible demand: the thrusters saturate at u_max anyway, so the
    # allocator must not command more than the actuators can deliver.
    ts = ThrusterSet(cfgs, dynamics=False)
    u_cmd, a_cmd = ThrustAllocator(cfgs).allocate(
        0.0, DT, _wrench6(Fy=400e3), u_now=ts.get_thrusts(), alpha_now=ts.get_angles())
    u_cmd = np.asarray(u_cmd, float)
    crit.add("infeasible 400 kN sway demand: |u_cmd| <= u_max for every thruster",
             np.all(np.isfinite(u_cmd)) and np.all(np.abs(u_cmd) <= u_max + 1.0),
             f"u_cmd = {np.round(u_cmd/1e3, 1)} kN")
    return crit.result(name)


# ---------------------------------------------------------------------------
# Thrust utilisation (part_2/utilization.py): the two functions students
# implement for Simulation 5, checked on synthetic logs (no sweep needed)
# ---------------------------------------------------------------------------

def check_utilization() -> CheckResult:
    """Thrust utilisation (part_2/utilization.py): definition and averaging window."""
    from types import SimpleNamespace

    from part_2.utilization import average_utilization, thrust_utilization

    name = "thrust utilisation (part_2/utilization.py): definition and averaging window"
    cfgs = default_thrusters_part_2()                    # 32 + 80 + 80 = 192 kN installed
    t = np.arange(0.0, 600.0 + DT / 2, DT)
    half = t >= 300.0
    u = np.zeros((len(t), len(cfgs)))
    u[~half] = [16e3, -40e3, 40e3]                       # 96 kN of 192 kN -> 50 %
    u[half] = [-8e3, 20e3, -20e3]                        # 48 kN of 192 kN -> 25 %
    logs = SimpleNamespace(t=t, u=u)
    try:
        ut = np.asarray(thrust_utilization(logs, cfgs), float)
        avg = float(average_utilization(logs, cfgs))
        avg_all = float(average_utilization(logs, cfgs, t_from=0.0))
    except NotImplementedError:
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["thrust_utilization() / average_utilization() still raise NotImplementedError"])
    except Exception as exc:                                    # noqa: BLE001
        return CheckResult(name, "ERROR", [f"utilisation functions raised: {exc!r}"])
    crit = _Criteria()
    crit.add("one value per time sample", ut.shape == t.shape, f"shape {ut.shape}")
    crit.add("signed thrusts enter as magnitudes: 16 / -40 / 40 kN of 192 kN -> 50 %",
             np.allclose(ut[~half], 50.0, atol=0.01), f"got {ut[~half].mean():.2f} %")
    crit.add("second half -8 / 20 / -20 kN -> 25 %",
             np.allclose(ut[half], 25.0, atol=0.01), f"got {ut[half].mean():.2f} %")
    crit.add("default window is the second half of the run: average = 25 %",
             abs(avg - 25.0) < 0.1, f"got {avg:.2f} %")
    crit.add("t_from=0 averages the whole run: 37.5 %",
             abs(avg_all - 37.5) < 0.2, f"got {avg_all:.2f} %")
    return crit.result(name)


# ---------------------------------------------------------------------------
# Observers (synthetic measurements, no plant): interface-level behaviour
# ---------------------------------------------------------------------------

def _observer_placeholder(obs) -> bool:
    """True if step() still returns the template pass-through."""
    if hasattr(obs, "reset"):
        obs.reset(np.zeros(6))
    for k in range(50):
        eta_m = _state6()
        eta_m[0] = 0.5 * k * DT
        est = _as_estimate(obs.step(k * DT, DT, eta_m, np.zeros(6)))
        if not (np.allclose(np.asarray(est.eta, float), eta_m)
                and np.allclose(np.asarray(est.nu, float), 0.0)):
            return False
    return True


def run_observer(kind: str, eta_meas: np.ndarray, tau: np.ndarray | None = None,
                 eta0: np.ndarray | None = None):
    """Feed a (n, 6) pose-measurement series to a fresh observer; return (eta_est, nu_est)."""
    from part_2.observer import select_observer

    obs = select_observer(kind)
    if hasattr(obs, "reset"):
        obs.reset(np.zeros(6) if eta0 is None else eta0)
    n = len(eta_meas)
    tau = np.zeros((n, 6)) if tau is None else tau
    est_eta, est_nu = np.empty((n, 6)), np.empty((n, 6))
    for k in range(n):
        est = _as_estimate(obs.step(k * DT, DT, eta_meas[k], tau[k]))
        est_eta[k] = np.asarray(est.eta, float).reshape(6)
        est_nu[k] = np.asarray(est.nu, float).reshape(6)
    return est_eta, est_nu


def wf_measurement(T: float = 600.0):
    """Synthetic wave-frequency pose measurement about a zero LF pose."""
    t = np.arange(int(T / DT)) * DT
    eta = np.zeros((len(t), 6))
    eta[:, 0] = 0.5 * np.sin(0.70 * t) + 0.3 * np.sin(0.95 * t + 1.0)
    eta[:, 1] = 0.6 * np.sin(0.75 * t + 0.5)
    eta[:, 5] = np.deg2rad(1.5) * np.sin(0.80 * t + 2.0)
    return t, eta


def _check_observer(kind: str) -> CheckResult:
    from part_2.observer import select_observer

    name = f"observer '{kind}': interface, filtering acts, heading wrap"
    try:
        obs = select_observer(kind)
    except Exception as exc:                                    # noqa: BLE001
        return CheckResult(name, "ERROR", [f"select_observer({kind!r}) raised: {exc!r}"])
    try:
        if _observer_placeholder(obs):
            return CheckResult(name, "NOT IMPLEMENTED",
                               [f"{type(obs).__name__}.step() still returns the pass-through placeholder"])
    except Exception as exc:                                    # noqa: BLE001
        return CheckResult(name, "ERROR", [f"step() raised: {exc!r}"])

    crit = _Criteria()
    # (a) constant drift north at 0.3 m/s, no control force: the velocity
    # estimate must be produced from pose only and point the right way.
    n = int(400.0 / DT)
    eta_m = np.zeros((n, 6))
    eta_m[:, 0] = 0.3 * np.arange(n) * DT
    est_eta, est_nu = run_observer(kind, eta_m)
    sel = slice(n // 2, n)
    crit.add("all estimates finite", np.all(np.isfinite(est_eta)) and np.all(np.isfinite(est_nu)))
    crit.add("constant northward drift: surge velocity estimate is positive (from pose only)",
             np.mean(est_nu[sel, 0]) > 0.0, f"mean u_est = {np.mean(est_nu[sel, 0]):.3f} m/s (true 0.3)")
    crit.add("constant northward drift: position estimate follows the measurement (< 5 m)",
             np.mean(np.abs(est_eta[sel, 0] - eta_m[sel, 0])) < 5.0,
             f"mean |error| = {np.mean(np.abs(est_eta[sel, 0] - eta_m[sel, 0])):.2f} m")

    # (b) wave-frequency measurement: the estimate must be smoother than the
    # measurement (how much is a design/performance question for the report).
    _, eta_w = wf_measurement()
    est_eta, est_nu = run_observer(kind, eta_w)
    sel = slice(len(eta_w) // 2, len(eta_w))
    for idx, lab in ((0, "North"), (1, "East"), (5, "heading")):
        r = float(np.std(est_eta[sel, idx]) / max(np.std(eta_w[sel, idx]), 1e-9))
        crit.add(f"wave filtering acts on {lab}: std(estimate) < std(measurement)",
                 r < 1.0, f"ratio = {r:.2f}")

    # (c) heading wrap: slow turn through +-pi must not produce a 2*pi jump.
    n = int(200.0 / DT)
    r = 0.6 / 200.0
    psi = np.array([wrap_angle_pi(np.pi - 0.3 + r * k * DT) for k in range(n)])
    eta_h = np.zeros((n, 6))
    eta_h[:, 5] = psi
    est_eta, _ = run_observer(kind, eta_h, eta0=_state6(np.pi - 0.3))
    steps = np.abs([wrap_angle_pi(d) for d in np.diff(est_eta[:, 5])])
    final_err = abs(wrap_angle_pi(est_eta[-1, 5] - psi[-1]))
    crit.add("heading through +-pi: no jump > 10 deg between steps (wrapped)",
             np.max(steps) < np.deg2rad(10.0), f"largest step = {np.rad2deg(np.max(steps)):.2f} deg")
    crit.add("heading through +-pi: final heading estimate within 5 deg (wrapped)",
             final_err < np.deg2rad(5.0), f"error = {np.rad2deg(final_err):.2f} deg")
    return crit.result(name)


def check_observer_nonlinear_passive() -> CheckResult:
    """Observer 'nonlinear_passive': interface, filtering acts, heading wrap."""
    return _check_observer("nonlinear_passive")


def check_observer_kalman() -> CheckResult:
    """Observer 'kalman': interface, filtering acts, heading wrap."""
    return _check_observer("kalman")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FAST_CHECKS = {
    "p2_current": check_current,
    "p2_wind": check_wind,
    "p2_allocation": check_allocation,
    "p2_utilization": check_utilization,
    "p2_controller": check_controller,
    "p2_reference": check_reference,
    "p2_observer_npo": check_observer_nonlinear_passive,
    "p2_observer_kf": check_observer_kalman,
}

# All public Part 2 checks are fast.  Closed-loop performance is not checked
# here: it is assessed by the teaching staff from the submitted results.
ALL_CHECKS = dict(FAST_CHECKS)


def run_check(key: str) -> CheckResult:
    """Run one Part 2 check by registry key, converting crashes into ERROR results."""
    func = ALL_CHECKS[key]
    try:
        return func()
    except Exception as exc:                                    # noqa: BLE001
        name = (func.__doc__ or key).splitlines()[0].rstrip(".")
        return CheckResult(name, "ERROR", [f"check crashed: {exc!r}"])


def run_all(fast_only: bool = False) -> list[CheckResult]:
    keys = FAST_CHECKS if fast_only else ALL_CHECKS
    return [run_check(k) for k in keys]
