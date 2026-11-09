"""
The DP controller requests a BODY-frame wrench tau = [Fx, Fy, Mz]. Directly
using azimuth thrust magnitude u and angle alpha as decision variables makes
the wrench equality nonlinear. Instead, the azimuth thrusters are represented
by Cartesian force components:

    z = [u_T, Fx1, Fy1, Fx2, Fy2]

which gives the linear relation

    Be @ z = tau.

The azimuth magnitude limit then becomes a circular constraint in the
(Fx, Fy) plane. This constraint is nonlinear but convex, which keeps the
saturation problem much easier to solve than the direct (u, alpha) form.

Normal operation uses the minimum-norm solution. If that solution violates
static thrust limits, a constrained optimisation redistributes the available
thrust. Uniform de-rating is kept as a robust fallback if the optimiser does
not return a valid solution.

Part 2 retains the Part 1 static allocation, then computes one-step-feasible
thruster commands using the current thrust, azimuth and rate limits. The four
equivalent azimuth sign combinations are compared using the predicted applied
BODY wrench. The supplied actuator model still enforces the physical limits.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi


class ThrustAllocator:
    """Allocates the requested 3-DOF BODY wrench to the thrusters"""

    # ------------------------------------------------------------------
    # Initialization of thruster configuration
    # ------------------------------------------------------------------

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

        # One fixed tunnel thruster and two azimuths
        if (
            len(thrusters) != 3
            or thrusters[0].kind != "tunnel"
            or thrusters[1].kind != "azimuth"
            or thrusters[2].kind != "azimuth"
        ):
            raise ValueError(
                "Thruster configuration does not match the expected layout"
            )

        self.tunnel = thrusters[0]
        self.az1 = thrusters[1]
        self.az2 = thrusters[2]

        # Constant extended configuration matrix for
        # z = [u_T, Fx1, Fy1, Fx2, Fy2]
        self.Be = self._build_configuration_matrix()

        # Rank(Be)=3 is required to independently produce surge, sway and yaw
        if np.linalg.matrix_rank(self.Be) != 3:
            raise ValueError("Extended thrust configuration matrix Be must have rank 3")
        self._Be_pinv = np.linalg.solve(self.Be @ self.Be.T, self.Be).T

        # The simulator stores thrust limits in N, while this allocator works in
        # kN/kNm internally to give SLSQP better numerical scaling
        self.u_max_kn = np.array([th.u_max for th in thrusters], dtype=float) / 1000.0

        # Part 2 uses these limits to compute commands that are reachable
        # within the next simulation step.
        self.u_rate = np.array([th.u_rate for th in thrusters], dtype=float)
        self.rot_speed = np.array([th.rot_speed for th in thrusters], dtype=float)

        if np.any(~np.isfinite(self.u_max_kn)) or np.any(self.u_max_kn <= 0.0):
            raise ValueError("Thruster maximum thrust must be finite and positive")
        if np.any(~np.isfinite(self.u_rate)) or np.any(self.u_rate <= 0.0):
            raise ValueError("Thruster thrust rates must be finite and positive")
        if np.any(~np.isfinite(self.rot_speed[1:])) or np.any(self.rot_speed[1:] <= 0.0):
            raise ValueError("Azimuth rotation rates must be finite and positive")

    # ------------------------------------------------------------------
    # Extended configuration matrix
    # ------------------------------------------------------------------

    def _build_configuration_matrix(self) -> np.ndarray:
        """
        Build Be such that tau = Be @ z for

            z = [u_T, Fx1, Fy1, Fx2, Fy2].
        """

        # Standard wrench column for the fixed tunnel thruster:
        # [cos(alpha), sin(alpha), x sin(alpha) - y cos(alpha)]^T.
        alpha_t = self.tunnel.alpha0
        B_tunnel = np.array([
            np.cos(alpha_t),
            np.sin(alpha_t),
            self.tunnel.x * np.sin(alpha_t) - self.tunnel.y * np.cos(alpha_t),
        ])

        # Splitting each azimuth force into Cartesian components gives
        # constant columns Fx -> [1, 0, -y] and Fy -> [0, 1, x]
        B_fx1 = np.array([1.0, 0.0, -self.az1.y])
        B_fy1 = np.array([0.0, 1.0, self.az1.x])
        B_fx2 = np.array([1.0, 0.0, -self.az2.y])
        B_fy2 = np.array([0.0, 1.0, self.az2.x])

        return np.column_stack([B_tunnel, B_fx1, B_fy1, B_fx2, B_fy2])

    # ------------------------------------------------------------------
    # Normal allocation
    # ------------------------------------------------------------------

    def _minimum_norm_allocation(self, tau: np.ndarray) -> np.ndarray:
        """
        Return the minimum-norm solution of Be @ z = tau.

        Since Be is 3x5, several allocations can produce the same wrench. The
        pseudo-inverse solution chooses the z with minimum Euclidean norm. This
        minimises squared thrust magnitude
        """

        # z = Be.T @ (Be @ Be.T)^(-1) @ tau.
        # solve() avoids explicitly forming the matrix inverse.
        return self._Be_pinv @ tau

    # ------------------------------------------------------------------
    # Static thrust limits
    # ------------------------------------------------------------------

    def _is_feasible(self, z: np.ndarray, tol: float = 1e-6) -> bool:
        """Check the tunnel limit and the circular azimuth thrust limits."""

        # The tunnel limit is scalar. For each azimuth, the physical thrust
        # magnitude is the Euclidean norm of its Cartesian force components
        u_tunnel = abs(z[0])
        u_az1 = np.hypot(z[1], z[2])
        u_az2 = np.hypot(z[3], z[4])

        return (
            u_tunnel <= self.u_max_kn[0] + tol
            and u_az1 <= self.u_max_kn[1] + tol
            and u_az2 <= self.u_max_kn[2] + tol
        )

    def _uniform_derating(self, z: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Uniformly scale z until all static thrust limits are satisfied.

        Because Be is linear, uniform scaling preserves the wrench direction.
        The returned lambda is the achievable fraction of the original wrench.
        """

        # Relative utilisation of tunnel, azimuth 1 and azimuth 2
        utilisation = np.array([
            abs(z[0]) / self.u_max_kn[0],
            np.hypot(z[1], z[2]) / self.u_max_kn[1],
            np.hypot(z[3], z[4]) / self.u_max_kn[2],
        ])

        # If the largest utilisation is r > 1, z/r satisfies all limits
        scale = max(1.0, float(np.max(utilisation)))
        return z / scale, 1.0 / scale

    # ------------------------------------------------------------------
    # Constrained allocation used only when the normal solution saturates
    # ------------------------------------------------------------------

    def _constrained_allocation(
        self,
        tau: np.ndarray,
        z_unconstrained: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Redistribute thrust when the minimum-norm solution is infeasible.

        The optimisation maximises the achievable wrench fraction lambda:

            Be @ z = lambda * tau,        0 <= lambda <= 1

        subject to the tunnel limit and

            Fx_j^2 + Fy_j^2 <= u_max_j^2

        for both azimuth thrusters. The equality is linear and each azimuth limit
        is a convex disc in force space, so the saturation problem remains
        convex even though the disc constraints are nonlinear.

        lambda = 1 means that the complete requested wrench is achievable;
        lambda < 1 means the request exceeds the available actuator capacity.
        """

        # Start from the uniformly de-rated solution. It is already feasible
        # and satisfies the scaled wrench equality
        z0, lambda0 = self._uniform_derating(z_unconstrained)
        x0 = np.append(z0, lambda0)  # x = [z, lambda]

        # SLSQP minimises, hence -lambda corresponds to maximising lambda
        def objective(x: np.ndarray) -> float:
            return -float(x[5])

        # Keep the achieved wrench in the same direction as the requested one
        def wrench_constraint(x: np.ndarray) -> np.ndarray:
            return self.Be @ x[:5] - x[5] * tau

        # The azimuth limits are discs in the (Fx, Fy) plane. Writing them in
        # squared form avoids the square root in u = sqrt(Fx^2 + Fy^2)
        constraints = [
            {"type": "eq", "fun": wrench_constraint},
            {
                "type": "ineq",
                "fun": lambda x: self.u_max_kn[1] ** 2 - x[1] ** 2 - x[2] ** 2,
            },
            {
                "type": "ineq",
                "fun": lambda x: self.u_max_kn[2] ** 2 - x[3] ** 2 - x[4] ** 2,
            },
        ]

        # Tunnel thrust and lambda have simple box constraints
        bounds = [
            (-self.u_max_kn[0], self.u_max_kn[0]),  # u_T
            (None, None),                     # Fx1
            (None, None),                     # Fy1
            (None, None),                     # Fx2
            (None, None),                     # Fy2
            (0.0, 1.0),                       # lambda
        ]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 150, "disp": False},
        )

        z = np.asarray(result.x[:5], dtype=float)
        lam = float(result.x[5])

        # Verify that the returned point
        # is finite, respects the thrust limits and satisfies the equality
        residual = np.linalg.norm(self.Be @ z - lam * tau)
        valid = (
            np.all(np.isfinite(z))
            and np.isfinite(lam)
            and self._is_feasible(z, tol=1e-3)
            and -1e-6 <= lam <= 1.0 + 1e-6
            and residual <= 1e-3
            and lam >= lambda0 - 1e-7
        )

        if not valid:
            return None

        return z

    # ------------------------------------------------------------------
    # Converts Cartesian azimuth forces back to force magnitude and angle
    # ------------------------------------------------------------------

    def _to_thruster_commands(
        self,
        z: np.ndarray,
        alpha_now: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert z to signed thrust [N] and azimuth angle [rad] commands.

        For each azimuth, u = sqrt(Fx^2 + Fy^2) and alpha = atan2(Fy, Fx).
        The equivalent command (-u, alpha + pi) is chosen if it requires less
        rotation from the current azimuth angle.
        """

        u_cmd = np.zeros(3)
        alpha_cmd = np.zeros(3)

        # Unpack z explicitly so the code mirrors the allocation-vector
        # definition used in the report
        u_tunnel, Fx1, Fy1, Fx2, Fy2 = z

        # Tunnel direction is fixed; only its signed thrust changes
        u_cmd[0] = u_tunnel * 1000.0
        alpha_cmd[0] = self.tunnel.alpha0

        # Convert each azimuth from Cartesian force components to (u, alpha).
        azimuth_forces = ((1, Fx1, Fy1), (2, Fx2, Fy2))
        for idx, Fx, Fy in azimuth_forces:
            u = float(np.hypot(Fx, Fy))

            current_angle = None if alpha_now is None else float(alpha_now[idx])

            # At zero thrust the angle has no force effect. Keep the current
            # angle, if known, to avoid an unnecessary azimuth command
            if u < 1e-10:
                alpha = 0.0 if current_angle is None else current_angle
            else:
                alpha = float(np.arctan2(Fy, Fx))

                if current_angle is not None:
                    # (u, alpha) and (-u, alpha + pi) generate the same force.
                    # Choose the representation with the smaller rotation.
                    alpha_flipped = wrap_angle_pi(alpha + np.pi)
                    rotation_normal = abs(wrap_angle_pi(alpha - current_angle))
                    rotation_flipped = abs(wrap_angle_pi(alpha_flipped - current_angle))

                    if rotation_flipped < rotation_normal:
                        u = -u
                        alpha = alpha_flipped

            # Convert kN back to N for the simulator and wrap the angle to the
            # project convention (-pi, pi].
            u_cmd[idx] = u * 1000.0
            alpha_cmd[idx] = wrap_angle_pi(alpha)

        return u_cmd, alpha_cmd

    # ------------------------------------------------------------------
    # Part 2: one-step-feasible thrust and azimuth commands
    # ------------------------------------------------------------------

    def _one_step_allocation(
        self,
        u_target: np.ndarray,
        alpha_target: np.ndarray,
        u_now: np.ndarray,
        alpha_now: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Choose rate-feasible commands for the next actuator time step.

        First, clip each target thrust and angle to the changes permitted in
        one step. For each azimuth, (u, alpha) and (-u, alpha + pi) are
        equivalent at steady state, but produce different transient forces.
        Evaluate all four combinations, preferring shorter transitions to
        the static target and then smaller predicted one-step wrench error.
        """

        # Targets already obey the static maximum-thrust constraints.
        # Match the slew-rate rules used by ThrusterSet.step().
        du_max = self.u_rate * dt
        da_max = self.rot_speed * dt
        u_limit = self.u_max_kn * 1000.0

        # The tunnel angle is fixed. Its thrust is the same in every option.
        best_u = None
        best_alpha = None
        best_cost = np.inf
        best_steps = np.inf

        # The nominal static allocation is the target for the predicted wrench.
        # Convert yaw moment to a force equivalent with a characteristic arm.
        c = np.cos(alpha_target)
        s = np.sin(alpha_target)
        tau_target = np.array([
            np.dot(u_target, c),
            np.dot(u_target, s),
            sum(u_target[i] * (self.thrusters[i].x * s[i] - self.thrusters[i].y * c[i]) for i in range(3)),
        ])
        moment_arm = max(1.0, max(np.hypot(th.x, th.y) for th in self.thrusters))

        for flip1 in (False, True):
            for flip2 in (False, True):
                desired_u = np.array(u_target, copy=True)
                desired_a = np.array(alpha_target, copy=True)

                for idx, flipped in ((1, flip1), (2, flip2)):
                    if flipped:
                        desired_u[idx] = -desired_u[idx]
                        desired_a[idx] = wrap_angle_pi(desired_a[idx] + np.pi)

                # Estimate how many steps are needed to reach the steady-state
                # representation. Prefer the shortest overall transition.
                steps = 0
                for idx in (1, 2):
                    thrust_steps = int(np.ceil(abs(desired_u[idx] - u_now[idx]) / du_max[idx] - 1e-12))
                    angle_steps = int(np.ceil(abs(wrap_angle_pi(desired_a[idx] - alpha_now[idx])) / da_max[idx] - 1e-12))
                    steps += max(thrust_steps, angle_steps)

                # Commands are one-step reachable from the present actuator state.
                u_step = np.clip(desired_u, u_now - du_max, u_now + du_max)
                u_step = np.clip(u_step, -u_limit, u_limit)
                a_step = np.array(desired_a, copy=True)
                for idx in (1, 2):
                    da = wrap_angle_pi(desired_a[idx] - alpha_now[idx])
                    a_step[idx] = wrap_angle_pi(alpha_now[idx] + np.clip(da, -da_max[idx], da_max[idx]))

                # Predict actual wrench at the end of the next step.
                ca, sa = np.cos(a_step), np.sin(a_step)
                tau_step = np.array([
                    np.dot(u_step, ca),
                    np.dot(u_step, sa),
                    sum(u_step[i] * (self.thrusters[i].x * sa[i] - self.thrusters[i].y * ca[i]) for i in range(3)),
                ])
                residual = (tau_target - tau_step) / np.array([1.0, 1.0, moment_arm])
                cost = float(np.dot(residual, residual))

                # Use one-step wrench error to break ties in transition time.
                # This avoids a short-sighted choice that looks good after one
                # step but takes much longer to reach the final allocation.
                if (steps < best_steps or
                    (steps == best_steps and
                     (best_u is None or cost < best_cost - max(1e-6, best_cost * 1e-8)))):
                    best_steps = steps
                    best_cost = cost
                    best_u = u_step
                    best_alpha = a_step

        return best_u, best_alpha

    # ------------------------------------------------------------------
    # "Main" function that runs per time-step
    # ------------------------------------------------------------------

    def allocate(
        self,
        t: float,                                           # Not used
        dt: float,                                          # Simulation time step [s]
        tau_d: np.ndarray,                                  # Desired BODY wrench [N, Nm]
        u_now: Optional[np.ndarray] = None,                 # Current thrust [N]
        alpha_now: Optional[np.ndarray] = None,             # Current azimuth angles [rad]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates thruster force-magnitude and direction based on the desired 3-DOF controller wrench"""

        # Allocate surge, sway and yaw only. Use kN/kNm internally so the
        # constrained optimisation is numerically well scaled.
        tau_d = np.asarray(tau_d, dtype=float).reshape(6)
        if not np.all(np.isfinite(tau_d)):
            raise ValueError("Desired wrench contains a non-finite value")

        if u_now is not None:
            u_now = np.asarray(u_now, dtype=float).reshape(3)
            if not np.all(np.isfinite(u_now)):
                raise ValueError("Current thrust contains a non-finite value")
        if alpha_now is not None:
            alpha_now = np.asarray(alpha_now, dtype=float).reshape(3)
            if not np.all(np.isfinite(alpha_now)):
                raise ValueError("Current angles contain a non-finite value")

        tau = tau_d[[0, 1, 5]] / 1000.0

        # Use the minimum-norm solution during normal operation.
        z = self._minimum_norm_allocation(tau)

        # Check if the requested wrench is feasible. If a static limit is
        # exceeded, redistribute thrust while preserving the wrench direction.
        # If this is not possible, the achieved wrench magnitude is reduced.
        if not self._is_feasible(z):
            z_constrained = self._constrained_allocation(tau, z)

            if z_constrained is not None:
                z = z_constrained

        # Ensure that all static thrust limits are satisfied.
        # Also serves as fallback if constrained allocation fails.
        z, _ = self._uniform_derating(z)

        u_cmd, alpha_cmd = self._to_thruster_commands(z, alpha_now)

        if u_now is not None and alpha_now is not None:
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("Simulation time step must be finite and positive")
            u_cmd, alpha_cmd = self._one_step_allocation(
                u_cmd, alpha_cmd, u_now, alpha_now, dt
            )

        # Verify that commanded thrust remains within the static limits.
        if np.any(np.abs(u_cmd) > self.u_max_kn * 1000.0 + 1e-6):
            raise RuntimeError("Thrust allocation produced an invalid thrust command")

        return u_cmd, alpha_cmd
