# TMR4240 Project – Python DP Simulator

<p align="center">
  <img src="assets/gunnerus.jpg" alt="R/V Gunnerus" width="520">
</p>

Python simulation framework for the project in **TMR4240 Marine Control Systems I** (NTNU, Department of Marine Technology): design, implement, and validate a dynamic positioning (DP) system for NTNU's research vessel **R/V Gunnerus**. The vessel is a 3-DOF (surge, sway, yaw) maneuvering model from the [`mcsimpy`](https://github.com/NTNU-MCS/mcsimpy) toolbox. Everything around it — the closed-loop engines, actuators, measurement layer, wave loads, logging, plotting and automated checks — is provided here. **You design and implement the DP system.**

The project runs in **two parts of equal weight**, and Part 2 builds directly on Part 1:

| | Project Part 1 | Project Part 2 |
| --- | --- | --- |
| You implement | current, wind, controller, reference model, thrust allocation | the same five blocks, extended, plus **two observers** and the **thrust-utilisation metrics** |
| Environment | current and a mean, slowly varying wind | that wind **plus gust and slowly varying direction**, and **irregular waves** |
| Actuators | ideal: commands applied exactly | **constrained**: maximum thrust, ramp rate, 2 rpm azimuth slew |
| Feedback | the true vessel state | a **measured pose only**, filtered by your observer |
| Your folder | `part_1/` | `part_2/` |
| Your runner | `run_case_part1.py` | `run_case_part_2.py` |
| Mandatory simulations | Part 1, Simulations 1–4 | Part 2, Simulations 1–7, plus extra credit |

**Where the requirements live.** The two assignment documents, *Project Part 1* and *Project Part 2*, are on Canvas under the *Project* module. They define the modelling, design, simulation and report requirements, and they are what your report is graded against. **This README is the other half:** it explains how the repository is organised, what the provided code guarantees, how to run everything, and how the code maps onto the tasks in those documents. When the two disagree about a number, the assignment wins; when they disagree about a file name or a command, this README wins.

---

## Contents

1. [Quick start](#quick-start)
2. [Who owns what](#who-owns-what)
3. [Repository map](#repository-map)
4. [The provided infrastructure](#the-provided-infrastructure)
5. [Array and frame conventions](#array-and-frame-conventions)
6. [Project Part 1 — what you implement](#project-part-1--what-you-implement)
7. [Project Part 2 — what you implement](#project-part-2--what-you-implement)
8. [Carrying your Part 1 work into Part 2](#carrying-your-part-1-work-into-part-2)
9. [Configuration — nothing is hardcoded](#configuration--nothing-is-hardcoded)
10. [Running simulations](#running-simulations)
11. [Automated checks](#automated-checks)
12. [Recommended workflow](#recommended-workflow)
13. [Notebooks](#notebooks)
14. [Plant API for open-loop tests](#plant-api-for-open-loop-tests)
15. [Updating a repository you have been working in](#updating-a-repository-you-have-been-working-in)
16. [Common pitfalls](#common-pitfalls)
17. [Credits](#credits)

---

## Quick start

Requirements: **Python ≥ 3.10** and **Git** (pip fetches `mcsimpy` from GitHub).

```bash
git clone <repository-url>
cd TMR4240_LAB

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -e .                 # numpy, scipy, matplotlib, mcsimpy, Jupyter, pytest
```

Then confirm both parts are wired up. On a fresh clone every one of your blocks reports `NOT IMPLEMENTED`, which is the expected starting state:

```bash
python run_case_part1.py         # runs the (empty) Part 1 closed loop and plots
python check.py --fast           # Part 1 subsystems: all NOT IMPLEMENTED
python check.py --part 2         # Part 2 subsystems: all NOT IMPLEMENTED
```

The editable install (`-e`) means your edits take effect immediately, with no reinstall. If those commands run, your environment is ready.

---

## Who owns what

Three categories, and the folder tells you which one a file is in:

| Category | Where | What it means |
| --- | --- | --- |
| **Yours to implement** | `part_1/`, `part_2/` | The design work. Every file here is a template with placeholders to replace. Keep the interfaces, change everything else. |
| **Yours to drive** | `run_case_part1.py`, `run_case_part_2.py` | The scenario runners, yours to edit and run. `run_case_part_2.py` ships one preset per Part 2 mandatory simulation; `run_case_part1.py` is a worked example you build your Part 1 scenarios from, and `notebooks/part_1_demo.ipynb` sets up the Part 1 mandatory simulations for you. You also make two choices here that the assignment asks of you (see `SELECTED_OBSERVER` and `OBSERVER_OFFSET` below). |
| **Provided** | `models/`, `simulation/`, `data/`, `check.py`, `tests/`, `notebooks/`, `assets/` | The plant, the engines, the measurement and wave models, the checks and the teaching notebooks. You should not need to change any of it. |

You *may* modify provided code if your design genuinely requires it — but document every such change in your report, and be aware that the automated checks and the infrastructure tests hold the provided contracts. Weakening a check to make your code pass is not a solution.

---

## Repository map

```text
run_case_part1.py        Part 1 scenario runner — yours to edit and run
run_case_part_2.py       Part 2 scenario runner — yours to edit and run, one preset per simulation
check.py                 Provided — the check runner: python check.py [--fast] [--part 2]

part_1/                  YOURS — Project Part 1
  config.py                every tunable Part 1 parameter (SimConfig, RefAxisConfig, thrusters, your own dataclasses)
  current.py               ocean current model
  wind.py                  wind load model
  controller.py            DP controller
  reference.py             set-point reference model
  thrust_allocation.py     thrust allocation

part_2/                  YOURS — Project Part 2 (the same five blocks, extended, plus two new ones)
  config.py                Part 2 parameters (Part2SimConfig, RefAxisConfig, observer/gust/allocation settings)
  current.py               your Part 1 current model, unchanged
  wind.py                  your Part 1 wind model + gust and slowly varying direction
  controller.py            your controller, now fed observer estimates
  reference.py             your reference model
  thrust_allocation.py     your allocation, now respecting the actuator limits
  observer.py              NEW — nonlinear passive observer and extended Kalman filter
  utilization.py           NEW — the two thrust-utilisation metrics of Part 2, Simulation 5

models/                  Provided — the physical models
  gunnerus_3dof.py         Gunnerus 3-DOF vessel wrapper (mcsimpy)
  thruster_dynamics.py     actuator model: maximum thrust, ramp rate, azimuth slew
  sensors.py               Part 2: the measurement layer between vessel and observer
  waves.py                 Part 2: irregular waves, force RAOs and second-order drift loads

simulation/              Provided — engines, checks and plots
  simulation_part_1.py     DPSimulator3DOF — the Part 1 closed loop
  simulation_part_2.py     DPSimulatorPart2 — the Part 2 closed loop
  plant.py                 GunnerusPlant3DOF — the open-loop plant API
  utils.py                 Rz(psi), angle wrapping, 3-DOF <-> 6-DOF helpers
  plotter.py               Part 1 and shared plots
  plotter_part_2.py        Part 2 plots: observer, wind field, waves, thruster set-points
  capability.py            Part 2: the Simulation 5 direction sweep and polar plots — it calls
                           your part_2/utilization.py for the metric itself
  checks.py                the Part 1 check definitions
  checks_part_2.py         the Part 2 check definitions

data/wind_coeff.csv        Gunnerus wind coefficient table C(alpha_rw)
data/gunnerus_vessel.json  Gunnerus wave-load database (why it exists: data/GUNNERUS_WAVE_DATA.md)
notebooks/                 Teaching and verification notebooks (see below)
tests/                     pytest wrappers around the check definitions, run by CI
assets/                    the vessel photo and the NTNU logo
```

---

## The provided infrastructure

The engines are deliberately short and worth reading once. They exist so that you write physics and control law, not plumbing. Each one guarantees the following, every time step.

**Part 1 — `simulation/simulation_part_1.py`, `DPSimulator3DOF`.** One time step is exactly the signal chain the assignment asks you to draw and explain in the *DP System Architecture* section:

```text
 setpoint eta_cmd (NED)
        │
        ▼
 ReferenceModel.step()         smooth eta_ref, nu_ref, acc_ref        (NED)      part_1/reference.py
        │
        ▼
 DPController.compute()        desired wrench tau_d                    (BODY)     part_1/controller.py
        │
        ▼
 ThrustAllocator.allocate()    per-thruster (u_cmd, alpha_cmd)                    part_1/thrust_allocation.py
        │
        ▼
 ThrusterSet.step()            applied wrench tau_thr = B(alpha) u     (BODY)     models/thruster_dynamics.py
        │                        (ideal in Part 1; rate limits + saturation in Part 2)
        ▼
 Wind.step()                   wind loads tau_w6                       (BODY)     part_1/wind.py
        │
        ▼
 Gunnerus3DOF.integrate()      tau_total + current (relative velocity)            models/gunnerus_3dof.py
        │                        Current.step() gives nu_c in NED                 part_1/current.py
        ▼
 eta (NED), nu (BODY)  ───────►  fed straight back to the controller (Part 1 has no observer)
```

**Part 2 — `simulation/simulation_part_2.py`, `DPSimulatorPart2`.** The same chain with three insertions, and it wires only the `part_2/` blocks:

- **Waves** are added to the load sum, from `models/waves.py`.
- **A measurement layer** (`models/sensors.py`) sits between the true state and your feedback. Regular simulations use ideal measurements; the extra-credit task switches noise on.
- **Your observer** receives the measured **pose only** and the desired wrench of the *previous* step, and returns the estimates that the controller then uses. Velocity is never measured on the vessel, so estimating it from position and heading is the observer's job.

What the engines do for you, in both parts: integrate the vessel with forward Euler, reduce 6-DOF vectors to the 3-DOF vessel at one single boundary (`simulation.utils.to_3dof` / `to_6dof`), apply the actuator model, sum the loads, call your anti-windup hook with the wrench that was *actually applied*, and log every signal as a 6-DOF time history.

Two engine facts worth knowing before you tune:

- **Time step.** Part 1 integrates at `dt = 0.05 s` by default. Part 2 runs at a fixed `dt = 0.1 s` (or `0.01 s`) and rejects anything else, so the Part 1 default does not carry over. `mcsimpy` supports no multi-stage integrator for this model, and the wrapper rejects any `method` other than `"Euler"`.
- **Reset.** `reset_state()` puts the vessel at the requested pose and resets your controller, reference model and observer. It resets the observer **at the true pose**, so an estimate that starts on top of the truth has nothing to converge from. The Part 2 runner therefore keeps a separate offset run for the convergence demonstration.

---

## Array and frame conventions

Every generalized vector — everywhere in the project — is **6-DOF**, ordered `[surge, sway, heave, roll, pitch, yaw]`:

| Quantity | Layout | Frame |
| --- | --- | --- |
| Position/attitude `eta` | `[N, E, z, phi, theta, psi]` | NED |
| Velocity `nu` | `[u, v, w, p, q, r]` | BODY |
| Loads (`tau_d`, `tau_thr`, wind, waves) | `[Fx, Fy, Fz, Mx, My, Mz]` in N and Nm | BODY |
| Current `nu_c` | `[V_N, V_E, V_D, 0, 0, 0]` in m/s | NED, direction **towards** |

The vessel model is 3-DOF, so only indices **`[0, 1, 5]`** are active — heading is `eta[5]`, yaw rate is `nu[5]`, yaw moment is `tau[5]` — and the other components are zero. Each template docstring tells you exactly which entries to read and which to fill.

Frames follow the assignment: NED `x` North, `y` East, `z` down; BODY `x` to the bow, `y` to starboard; heading `psi` clockwise from North. The rotation used throughout is `NED = Rz(psi) @ BODY` (`simulation.utils.Rz`), so `Rz(psi).T` takes NED vectors — position errors, current, wind — into the body frame. Wrap heading differences with `simulation.utils.wrap_angle_pi`.

**Direction conventions.** The engine and the current model use *towards* (the direction the water flows to). The wind template defaults to the meteorological *from* convention ("wind from south" blows northward). Both constructors take a `semantics` argument so you can state the convention explicitly — and you must state it in the report. A 180° mistake here is the single most common DP bug.

---

## Project Part 1 — what you implement

The five templates in `part_1/` contain placeholder implementations (zero current, zero wind, zero wrench, pass-through reference, zero thrust). Replace the placeholders; keep the interfaces. Each docstring documents its interface in full — this is only the summary:

| Template | The engine calls | Returns |
| --- | --- | --- |
| `current.py` — `Current` | `step(t, dt, eta, nu)` | `nu_c` (6,) NED velocity, *towards* |
| `wind.py` — `Wind` | `step(t, dt, eta, nu)` | `(tau_w6, info)` — (6,) BODY loads + optional log dict |
| `controller.py` — `DPController` | `compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref)` | `tau_d` (6,) BODY wrench |
| `reference.py` — `ReferenceModel` | `step(t, dt, eta_cmd)` | `(eta_ref, nu_ref, acc_ref)` — all (6,) NED |
| `thrust_allocation.py` — `ThrustAllocator` | `allocate(t, dt, tau_d, u_now, alpha_now)` | `(u_cmd, alpha_cmd)` per thruster |

Useful details:

- **Wind coefficients.** `load_wind_coefficients()` in `part_1/wind.py` returns the angle grid (deg) and the `(M, 6)` table `[Cx, Cy, Cz, Cphi, Ctheta, Cpsi]` from `data/wind_coeff.csv`. Interpolate it periodically in `alpha_rw` and use the *relative* wind `V_rw = V_wind − V_vessel`, as `notebooks/wind_coefficients.ipynb` demonstrates.
- **Thruster layout** (`part_1/config.py::default_thrusters_gunnerus3`): bow tunnel at `x = +12 m` (fixed 90°, ±32 kN) and two stern azimuths at `x = −13 m, y = ±3 m` (80 kN each, freely rotating). Angles are BODY-frame, clockwise, 0 toward the bow. Each `ThrusterConfig` carries `x, y, u_max, u_rate, rot_speed, alpha0`.
- **Optional controller hooks** the engine uses if you define them: `reset()`, `apply_external_aw(tau_applied, psi, dt)` for anti-windup on the *applied* wrench, and the attributes `last_pid_body`, `int_ned`, `int_psi`, which are then logged and plotted for you.
- **Reference velocities and accelerations** are forwarded to the controller — a smooth reference model here is what makes velocity and acceleration feedforward possible there.

### Part 1 mandatory simulations

| Assignment task | Scenario | How to set it up |
| --- | --- | --- |
| Part 1, Simulation 1a | station keeping, current 0.5 m/s from east | `Current(0.5, np.pi/2, semantics="from")`, `Wind()` |
| Part 1, Simulation 1b | station keeping, wind 15 m/s mean from east, with slow variation | `Current()`, `Wind(15.0, np.pi/2, semantics="from", sigma_slow=...)` |
| Part 1, Simulation 2 | current rotating linearly from north to from east over 300 s | `Current(0.5, 0.0, semantics="from", beta_end=np.pi/2, duration=300.0)` |
| Part 1, Simulation 3 | setpoint `[10, 10, 3π/2]` with and without the reference model | same gains, `use_reference=True` vs `False` |
| Part 1, Simulation 4 | four-corner test | `eta_cmd` as an `(N_steps, 6)` time series with your switching rule |

`notebooks/part_1_demo.ipynb` builds all of these exactly as specified, produces the report plots, and runs the corresponding checks — start there.

---

## Project Part 2 — what you implement

Part 2 keeps the same vessel, the same three thrusters and the same five blocks, and extends them. `part_2/` is a separate folder with its own configuration and its own engine, so your finished Part 1 keeps working untouched while you develop Part 2.

| Block | Part 1 | Part 2 |
|---|---|---|
| Current (`part_2/current.py`) | your model | unchanged — copy it across |
| Wind (`part_2/wind.py`) | mean + slowly varying speed | **+ a gust** component from a wind spectrum (`gust=True`, `gust_params`) and a **slowly varying direction** bounded to ±5° (`sigma_dir`, `tau_dir`, `dir_limit`; the speed keeps `sigma_slow`, `tau_slow`) |
| Waves (`models/waves.py`) | — | **provided complete**: `Waves(hs, tp, direction, seed=...)`, first-order force RAOs and second-order drift loads, where `direction` is where the waves come *from*. Loads come from the corrected Gunnerus hydrodynamic database `data/gunnerus_vessel.json`; `data/GUNNERUS_WAVE_DATA.md` explains what was wrong with the original and what the correction does and does not fix. Verified in `notebooks/part_2_waves_verification.ipynb` |
| Sensors (`models/sensors.py`) | — | provided: `VesselSensors` between the true state and the observer. Regular simulations run with ideal measurements; the extra-credit task switches noise on (`use_sensor_noise=True`). See `notebooks/part_2_sensors.ipynb` |
| Observer (`part_2/observer.py`) | — | **new**: `NonlinearPassiveObserver` and `KalmanFilterObserver` behind one interface, `step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)` (a plain `(eta, nu, bias)` tuple is accepted too). It gets the **measured pose only**; `tau_est` is the desired wrench of the *previous* step, zero at the first sample. Select with `Part2SimConfig(observer_type=..., observer_kwargs=...)` and enable with `use_observer=True`. The design model (M, D) is derived in `notebooks/plant_model.ipynb` |
| Utilisation (`part_2/utilization.py`) | — | **new**: `thrust_utilization` and `average_utilization`, the two metrics behind the Part 2, Simulation 5 polar plots. The assignment defines them; the docstrings repeat the definition |
| Controller, reference | your design | retune or extend if needed, and justify it — the controller is now fed the *estimates*, not the truth |
| Thrust allocation (`part_2/thrust_allocation.py`) | ideal actuators | **constraints enforced** (`thruster_dynamics=True`): maximum thrust, ramp rate and a 2 rpm azimuth slew. Your allocator must respect them in its steady-state solution and resolve the thrust/angle sign ambiguity itself |

### Part 2 mandatory simulations

`run_case_part_2.py` has one preset per simulation of the Part 2 assignment:

```bash
python run_case_part_2.py --list    # sim1 … sim7, bonus
python run_case_part_2.py sim2      # Part 2, Simulation 2
```

| Preset | Assignment task | Scenario |
|---|---|---|
| `sim1` | Part 2, Simulation 1 | Environmental loads: free drift with the DP system off (`use_controller=False`); plots the total wind speed, its mean-plus-slow part and the gust separately, the wind direction, and the current and wave loads |
| `sim2` | Part 2, Simulation 2 | Four-corner DP test with the reference model and the constrained Part 2 thruster dynamics; environment and observer disabled |
| `sim3` | Part 2, Simulation 3 | Four-corner DP test in the Part 2, Simulation 1 environment, reference model and constrained thrusters on, raw measurements fed back with no observer |
| `sim4` | Part 2, Simulation 4 | Observer comparison and selection, in three stages: a fixed wrench with and without waves for both observers; the same run started from `OBSERVER_OFFSET` so the estimate has an error to converge from; then closed-loop station keeping with raw measurements against your `SELECTED_OBSERVER` |
| `sim5` | Part 2, Simulation 5 | Average thrust-utilisation polar plot over the environmental direction. `simulation/capability.py` runs the sweep and calls your `part_2/utilization.py` for the metric. Three polar plots: all directions; the DP watch-circle view, masking a direction whose maximum **low-frequency** excursion (30 s moving average, second half of the run) exceeds 5 m / 5°; and the operability view, masking a direction whose maximum **total** motion, wave-frequency oscillation included, exceeds 3 m / 3°. Both deviations are printed per direction |
| `sim6` | Part 2, Simulation 6 | Observer robustness in a heavy sea, 1000 s |
| `sim7` | Part 2, Simulation 7 | Your own demonstration — design it, explain it, discuss it |
| `bonus` | Extra credit (3 points) | The Part 2, Simulation 4 comparison and the raw-vs-observer closed loop repeated with measurement noise on. All regular simulations run with noise off |

**The two choices the runner leaves to you**, both at the top of `run_case_part_2.py`:

- `SELECTED_OBSERVER` starts as `None`. Compare both observers in Part 2, Simulation 4, then set it to the one you chose and justify the choice in the report. Part 2, Simulations 5–7 and the raw-vs-observer comparison use it explicitly, and stop with a message until you have set it.
- `OBSERVER_OFFSET` is the initial observer error used by the convergence demonstration. The engine resets every observer at the true pose, so this is what gives the estimate something to converge from. The shipped value is an example: choose your own and say why it is a fair test.

The environmental numbers are constants at the top of the same file. `notebooks/part_2_demo.ipynb` runs every preset with the report plots and all Part 2 checks.

---

## Carrying your Part 1 work into Part 2

The `part_2/` templates are intentionally minimal: copy your *implementation logic* into them while keeping the Part 2 imports, constructor signatures and new parameters. Do not overwrite the Part 2 files wholesale — `part_2/reference.py` must keep `from part_2.config import RefAxisConfig`, and `part_2/wind.py` must keep the gust and direction parameters.

1. Copy your custom configuration dataclasses and tuned Part 1 defaults into `part_2/config.py`, pointing their imports at `part_2.config`.
2. Transfer the controller implementation (`part_2/controller.py`).
3. Transfer the current model unchanged (`part_2/current.py`).
4. Transfer the reference model, keeping the Part 2 config import (`part_2/reference.py`).
5. Transfer the thrust allocation, then extend it for the constraints (`part_2/thrust_allocation.py`).
6. Transfer the wind model into the expanded Part 2 interface, then add the gust and the slowly varying, bounded direction (`part_2/wind.py`).
7. Implement the two thrust-utilisation functions (`part_2/utilization.py`). The Part 2, Simulation 5 sweep needs them and will stop until they exist.
8. Run `python check.py --part 2`. Everything but the two observer checks should pass before you start on `part_2/observer.py`.

---

## Configuration — nothing is hardcoded

Nothing you are asked to tune lives inside `simulation/` or `models/`; the engines only wire your blocks together. Every tunable parameter lives in **`part_1/config.py`** and **`part_2/config.py`**:

- `SimConfig` / `Part2SimConfig` — time step `dt`, duration `T`, and the switches: `use_reference`, `use_controller`, `use_observer`, `observer_type`, `use_sensor_noise`, `thruster_dynamics`, `bypass_actuators` (a debug mode that applies `tau_d` directly).
- `RefAxisConfig` — natural frequency `wn`, damping `zeta` and an optional `rate_limit` per reference axis. **None of the shipped values is a tuned value**: choose all three and justify them. The Part 2 file says so on all three fields; the Part 1 file flags only `wn`, because it was released before this note existed, but the same applies there.
- `default_thrusters_gunnerus3()` / `default_thrusters_part_2()` — the thruster geometry and limits given in the assignment.
- **Your own dataclasses.** This is the place to add `PIDGains`, allocation weights, observer gains and environment parameters — anything you tune — so that every simulation can be reconfigured from this one file plus the runner, without touching your model code. Each config file has a `TODO` marking the spot.

This is also what makes your report reproducible: a reader opens the config file and the runner and sees every number behind every figure.

---

## Running simulations

`run_case_part1.py` and `run_case_part_2.py` are the scenario files. Each builds the loop in the same order as the diagrams above — configuration, controller, reference model, thrusters, simulator, setpoint, environment, run, plot — and is meant to be edited per scenario:

```python
cfg = SimConfig(dt=0.05, T=800.0, use_reference=True)
controller = DPController()
reference  = ReferenceModel(dt=cfg.dt)
thrusters  = default_thrusters_gunnerus3()
sim = DPSimulator3DOF(cfg, controller, thrusters, reference=reference)

eta_cmd = np.array([0, 0, 0, 0, 0, 0.0])                   # 6-DOF setpoint, or an (N_steps, 6) time series
current = Current(0.5, np.pi / 2, semantics="from")        # Part 1, Simulation 1a: 0.5 m/s from east
wind    = Wind()                                           # none

sim.reset_state()
logs = sim.run(eta_cmd, current=current, wind=wind)
plot_dashboard(logs); plot_time_histories(logs); plt.show()
```

`run()` returns a `Logs` object with every time history as a 6-DOF array (`eta`, `nu`, `sp`, `cmd`, `tau_d`, `tau_thr`, `tau_total`, `tau_w6`, `u`, `alpha`, current and wind diagnostics, PID components, reference velocities, and in Part 2 the estimates and wave loads). The plot functions take `logs` directly and already carry axis labels, units and legends.

Part 1 and shared plots, in `simulation/plotter.py`:

| Function | Shows |
| --- | --- |
| `plot_dashboard` | trajectory, position and heading errors, thrusts, controller wrench, velocities |
| `plot_time_histories` | N, E, psi vs. setpoint; u, v, r vs. reference |
| `plot_xy` | North–East trajectory with setpoints |
| `plot_thrusters` | actual thrust and azimuth angle per thruster |
| `plot_wrench` | commanded vs. applied vs. total BODY wrench |
| `plot_current`, `plot_wind` | environment inputs and the resulting loads |

Part 2 adds `simulation/plotter_part_2.py`, used the same way:

| Function | Shows |
| --- | --- |
| `plot_observer` | truth, measurement and estimate — the velocity "measurements" are validation signals only, the observer never sees them |
| `plot_observer_error` | estimate minus truth, and estimate minus low-frequency truth, where the wave filtering is visible |
| `plot_wind_field` | total / mean+slow / gust speed, direction, loads |
| `plot_waves` | wave loads on the hull |
| `plot_thruster_setpoints` | allocator set-point vs. actual thrust and angle, per thruster |
| `plot_wrench_residual` | desired minus applied wrench |

and `simulation.capability.plot_capability` draws the Part 2, Simulation 5 polar plots.

### Simulation numbering

The two parts number their simulations separately: Part 1 has Simulations 1–4, Part 2 has Simulations 1–7 plus the extra-credit task. Part 1, Simulation 3 and Part 2, Simulation 3 are different exercises, so this README always states the project part when referring to a simulation. Do the same in your report.

---

## Automated checks

The repository ships with checks for both parts. They are your fastest feedback loop: they tell you whether a block satisfies the conventions and explicit requirements of the assignment before you spend time on a closed-loop run.

```bash
python check.py --fast      # Part 1 subsystem checks (seconds)
python check.py             # Part 1 subsystem checks and Part 1, Simulations 1–4 (~1 min)
python check.py --part 2    # the eight Part 2 subsystem checks (seconds)
pytest                      # both parts as a test suite
```

Each check reports `PASS`, `FAIL`, `NOT IMPLEMENTED` (the template placeholder is still in place — reported as skipped by pytest, so a fresh clone starts green) or `ERROR` (your code raised, or does not follow the template interface; the message says what to fix). The same suite runs on GitHub on every push, so your group always sees the current status online.

**What is covered.** Part 1 covers the subsystem sign tests — sign conventions, heading wrapping, allocation rank — and the four mandatory closed-loop simulations. Part 2 covers eight blocks: current, wind, thrust allocation, thrust utilisation, controller, reference model and the two observers. The Part 2 wind check looks for a gust and for a direction bounded to ±5°; the allocation check verifies that the requested wrench is reproduced once the allocator has converged through ideal actuators and that every command stays inside the thruster limits; the observer checks verify a finite output, velocity estimated from pose alone, reduced wave-frequency content and correct heading wrapping.

**What is not covered: performance.** How well your observer filters and how accurately the closed loop holds position are assessed from your submitted results and discussion, not by these checks. Passing them is necessary, not sufficient.

Three consequences worth stating plainly:

- **The checks construct your classes with their default constructors.** Tuning that exists only in a runner never reaches them, so your final tuned values must be the defaults — controller gains as `DPController.__init__` defaults, reference parameters as `RefAxisConfig` field defaults, observer gains as the observer's own defaults. The templates repeat this where it matters. The constructors the checks use are:

  ```python
  DPController()                                       # no arguments
  ReferenceModel(dt)                                   # RefAxisConfig defaults from the config file
  ThrustAllocator(thrusters)
  Current(speed, beta, semantics=..., beta_end=..., duration=...)
  Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)
  select_observer(kind)                                # no tuning arguments
  ```

- **Where a check puts a number on something the assignment leaves to your judgement, that number is a sanity bound, not a design target.** The reference-model check, for instance, asks a 10 m step to have moved less than 1 m after one second, to overshoot by less than 10 % and to settle within 600 s. That is a wide band which many defensible tunings satisfy; it is not a specification of the right answer.

- **Your Part 1 blocks keep their Part 1 behaviour in Part 2.** The Part 2 current check still exercises the rotating-current profile of Part 1, Simulation 2, and the Part 2 wind check reads the ambient speed, the mean-plus-slow speed, the gust speed and the direction out of the `info` dictionary your wind model returns each step, so keep populating those fields. The templates document the expected names and the sign convention at the point where you implement them.

The observer measurement contract — pose only, previous desired wrench, estimates into the controller — is additionally locked by infrastructure tests in `tests/`. If a check blocks you and you cannot see why, report it in the appendix and say what you tried, rather than weakening the check.

---

## Recommended workflow

The assignments ask for *Model → Implement → Test → Tune → Validate → Integrate → Analyse*, block by block, and the repository is built for exactly that. A `NOT IMPLEMENTED` result on a block you have not reached yet is normal — that is the point of checking block by block.

**Part 1**

1. **Understand the plant first.** Run `notebooks/plant_model.ipynb` for the equations and the numerical Gunnerus matrices you will need for your control plant model, and `notebooks/plant_tests.ipynb` for open-loop maneuvers, direction conventions and a frame-bug demonstration at a non-zero heading.
2. **Current.** Implement `Current.step()`, then check that the vessel drifts the right way.
3. **Wind.** Explore `notebooks/wind_coefficients.ipynb`, implement `Wind.step()`, and verify the sign test: bow north, wind from north, negative surge force.
4. **Thrust allocation.** Implement `allocate()` and test it on pure surge, sway and yaw requests — with ideal actuators `B(alpha_d) u_d = tau_c` must hold exactly. Confirm `rank(B_e) = 3`.
5. **Controller and reference model.** Implement, then tune from a physically motivated starting point using step responses. Document each tuning step for the report.
6. **Integrate.** Run the mandatory simulations, run the full checks, and analyse.

**Part 2**

7. **Move your Part 1 blocks across** and get the six non-observer checks passing, as described above.
8. **Extend the wind field** with the gust and the bounded slowly varying direction, then look at it in Part 2, Simulation 1 before any control is involved.
9. **Look at the disturbances alone.** Part 2, Simulation 1 is free drift with the DP system off: it tells you the size and character of what your controller must reject.
10. **Constrain the allocation.** Part 2, Simulation 2 has no environment, so it isolates actuator behaviour and your allocator's constraint handling.
11. **See why an observer is needed.** Part 2, Simulation 3 feeds raw measurements back in a full sea state. The thruster activity in that run is the motivation for everything that follows.
12. **Build both observers**, tune them, compare them in Part 2, Simulation 4, choose one on the evidence, and record the choice.
13. **Characterise and stress the result** through Part 2, Simulations 5–7, then analyse.

---

## Notebooks

```text
notebooks/plant_model.ipynb                # the model equations + the numerical Gunnerus matrices (use in your report)
notebooks/plant_tests.ipynb                # what the plant consists of + open-loop maneuver and environment tests
notebooks/wind_coefficients.ipynb          # load, plot and interpolate the wind coefficient table
notebooks/part_1_demo.ipynb                # Part 1, Simulations 1–4 with report plots + the Part 1 checks
notebooks/part_2_sensors.ipynb             # Part 2: the measurement layer between vessel and observer
notebooks/part_2_waves_verification.ipynb  # Part 2: verification of the provided irregular-wave model
notebooks/part_2_demo.ipynb                # Part 2, Simulations 1–7 with report plots + the Part 2 checks
```

The first three are reference material for your report: they document the provided plant, so their stored output is part of the documentation. The demo notebooks start empty and are meant to be run.

To run a notebook in VS Code: install the **Jupyter** extension, open the notebook, click **Select Kernel** → **Python Environments** → this project's `.venv`, and run the cells top to bottom. From the terminal: `jupyter notebook notebooks/plant_model.ipynb`.

---

## Plant API for open-loop tests

For open-loop tests or your own control loop, use the step-based plant. It chains the thruster set and the Gunnerus equations and accepts current, wind and wave inputs at every sample:

```python
import numpy as np
from part_1.config import default_thrusters_gunnerus3
from simulation.plant import GunnerusPlant3DOF

plant = GunnerusPlant3DOF(default_thrusters_gunnerus3(), dt=0.05,
                          thruster_dynamics=False)                   # ideal actuators, as in Part 1
plant.reset()                                                        # optional 6-DOF eta0 / nu0

sample = plant.step(
    t=0.0,
    thrust_command=[20_000.0, 30_000.0, 30_000.0],                   # [Tunnel_Bow, Azimuth_1, Azimuth_2]
    azimuth_command=[np.pi / 2, 0.0, 0.0],
    current=[0.4, 0.2, 0.0, 0.0, 0.0, 0.0],                          # NED, towards
    wind=[200.0, -50.0, 0.0, 0.0, 0.0, 20.0],                        # BODY wrench
    waves=lambda t, eta, nu: [0.0, 100.0 * np.sin(t), 0.0, 0.0, 0.0, 0.0],
)

r = plant.run(120.0, [0, 40e3, 40e3], [0, 0, 0])                     # whole run → dict of arrays
```

`step()` returns a `PlantStep` with the actual thruster state, each load contribution, the total wrench, the current and the post-integration state — all 6-DOF. `run(T, thrust_command, azimuth_command, current=..., wind=..., waves=..., external=..., eta0=...)` returns stacked histories; commands and the current may be constants or callables of time, and loads may be constant vectors or callables `load(t, eta, nu)`.

**Part 1 uses ideal actuators** (`thruster_dynamics=False`, the `SimConfig` default; the plant API defaults to `True`, so pass it explicitly as above): commands are applied exactly as requested, with no rate limits and no saturation. The plant will not clip a 120 kN command on an 80 kN thruster — respecting `u_max` is your allocation's job. Switching the dynamics on already in Part 1 is optional but a good robustness test; if you do, say so in the report. `notebooks/plant_tests.ipynb` compares the two modes side by side.

---

## Updating a repository you have been working in

The Part 2 release is additive. It adds the Part 2 files and changes only three shared files — `README.md`, `check.py` and `pyproject.toml` — and touches no file you implement. Everything under `part_1/`, plus `run_case_part1.py` and the Part 1 notebooks, is left exactly as you have it.

Commit your work first, then pull:

```bash
git status                       # nothing of yours should be uncommitted
git add -A && git commit -m "Save Project Part 1 work"

git pull --no-rebase origin main
pip install -e .                 # part_2 is a new package
python check.py --fast           # your Part 1 subsystems still pass
python check.py --part 2         # every Part 2 subsystem reports NOT IMPLEMENTED
```

If you have uncommitted work you do not want to commit yet, `git stash` before the pull and `git stash pop` after it.

**If the pull reports a conflict.** It can only happen in one of the three shared files, because those are the only ones the release changes. Git will say `CONFLICT (content): Merge conflict in <file>` and stop mid-merge. None of the three is yours, so take the release version:

```bash
git checkout --theirs README.md check.py pyproject.toml   # only those listed as conflicted
git add README.md check.py pyproject.toml
git commit                       # completes the merge
```

If you had deliberately edited one of them, open it instead and resolve by hand: the conflict markers `<<<<<<<`, `=======` and `>>>>>>>` bracket your version and the release version. Keep both intentions, delete the markers, then `git add` the file and `git commit`.

To back out and start over at any point before the final commit:

```bash
git merge --abort                # returns you to exactly where you were
```

Afterwards, confirm the merge did not disturb your Part 1 work:

```bash
python check.py                  # Part 1 subsystems and Simulations 1-4
```

---

## Common pitfalls

- **Wrong direction of drift or force.** Almost always a *from/towards* mix-up or a missing `Rz(psi).T` rotation. Test every environment model at a non-zero heading — at `psi = 0` the NED and BODY frames coincide and the bug is invisible (`plant_tests.ipynb`, Section 5).
- **Heading jumps by 2π.** Always compute heading errors with `wrap_angle_pi` / `atan2`; test setpoints near ±π (Part 1, Simulation 3 uses `3π/2`).
- **Tuning that "works for me" but fails the checks.** The checks use the default constructors — see [Automated checks](#automated-checks).
- **Position deviations far above 1–5 m.** Typical for a well-tuned solution is 1–5 m; much larger usually means a frame or sign error, poor tuning, an allocation rank deficiency, or over-commanded thrusters — not a bad vessel model.
- **Mixed units.** Metres, seconds, radians, newtons, newton-metres throughout. Plots convert to degrees and kN for display only.
- **Setpoint arrays.** `eta_cmd` must be `(6,)` or `(N_steps, 6)` with `N_steps = round(T/dt) + 1`.
- **A Part 2 time step of 0.05 s.** The Part 1 default does not carry over; Part 2 accepts 0.1 s or 0.01 s only.

---

## Credits

| | Author | GitHub | e-mail |
| --- | --- | --- | --- |
| PhD Candidate | Enio Krizman | [@kr1zzo](https://github.com/kr1zzo) | <enio.krizman@ntnu.no> |
| PhD Candidate | Saber Sakhrieh | [@sabersak](https://github.com/sabersak) | <saber.sakhrieh@ntnu.no> |
