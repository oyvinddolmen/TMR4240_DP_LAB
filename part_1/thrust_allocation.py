"""
Thrust allocation for TMR4240 Project Part 1.

The DP controller requests a BODY-frame wrench tau = [Fx, Fy, Mz]. Directly
using azimuth thrust magnitude u and angle alpha as decision variables makes
the wrench equality nonlinear. Instead, the azimuth thrusters are represented
by Cartesian force components:

    z = [u_T, Fx1, Fy1, Fx2, Fy2]

which gives the linear relation

    Be @ z = tau.

Normal operation uses the minimum-norm solution. If that solution violates a
static thrust limit, a constrained optimisation redistributes the available
thrust. Uniform de-rating is kept as a robust fallback if the optimiser does
not return a valid solution.

Part 1 uses ideal actuator dynamics, so thrust-rate and azimuth-rate limits are
not handled by this allocator.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi


class ThrustAllocator:
    """Allocate the requested 3-DOF BODY wrench to the Gunnerus thrusters."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

        # Part 1 uses one fixed tunnel thruster followed by two azimuths.
        # Keeping this assumption explicit makes the indexing easy to verify.
        if (
            len(thrusters) != 3
            or thrusters[0].kind != "tunnel"
            or thrusters[1].kind != "azimuth"
            or thrusters[2].kind != "azimuth"
        ):
            raise ValueError(
                "Expected the Part 1 Gunnerus layout: one tunnel and two azimuth thrusters."
            )

        self.tunnel = thrusters[0]
        self.az1 = thrusters[1]
        self.az2 = thrusters[2]

        # Constant extended configuration matrix for
        # z = [u_T, Fx1, Fy1, Fx2, Fy2].
        self.Be = self._build_configuration_matrix()

        # Rank 3 is required to independently produce surge, sway and yaw.
        if np.linalg.matrix_rank(self.Be) != 3:
            raise ValueError("Extended thrust configuration matrix Be must have rank 3.")

        # The simulator uses N, but kN/kNm give better numerical scaling in SLSQP.
        self.u_max = np.array([th.u_max for th in thrusters], dtype=float) / 1000.0

    # ------------------------------------------------------------------
    # Extended configuration matrix and normal allocation
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
        # constant columns Fx -> [1, 0, -y] and Fy -> [0, 1, x].
        B_fx1 = np.array([1.0, 0.0, -self.az1.y])
        B_fy1 = np.array([0.0, 1.0, self.az1.x])
        B_fx2 = np.array([1.0, 0.0, -self.az2.y])
        B_fy2 = np.array([0.0, 1.0, self.az2.x])

        return np.column_stack([B_tunnel, B_fx1, B_fy1, B_fx2, B_fy2])

    def _minimum_norm_allocation(self, tau: np.ndarray) -> np.ndarray:
        """
        Return the minimum-norm solution of Be @ z = tau.

        Since Be is 3x5, several allocations can produce the same wrench. The
        pseudo-inverse solution chooses the z with minimum Euclidean norm.
        """

        # z = Be.T @ (Be @ Be.T)^(-1) @ tau.
        # solve() avoids explicitly forming the matrix inverse.
        return self.Be.T @ np.linalg.solve(self.Be @ self.Be.T, tau)

    # ------------------------------------------------------------------
    # Static thrust limits
    # ------------------------------------------------------------------

    def _is_feasible(self, z: np.ndarray, tol: float = 1e-6) -> bool:
        """Check the tunnel limit and the circular azimuth thrust limits."""

        u_tunnel = abs(z[0])
        u_az1 = np.hypot(z[1], z[2])
        u_az2 = np.hypot(z[3], z[4])

        return (
            u_tunnel <= self.u_max[0] + tol
            and u_az1 <= self.u_max[1] + tol
            and u_az2 <= self.u_max[2] + tol
        )

    def _uniform_derating(self, z: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Uniformly scale z until all static thrust limits are satisfied.

        Because Be is linear, uniform scaling preserves the wrench direction.
        The returned lambda is the achievable fraction of the original wrench.
        """

        # Relative utilisation of tunnel, azimuth 1 and azimuth 2.
        utilisation = np.array([
            abs(z[0]) / self.u_max[0],
            np.hypot(z[1], z[2]) / self.u_max[1],
            np.hypot(z[3], z[4]) / self.u_max[2],
        ])

        # If the largest utilisation is r > 1, z/r satisfies all limits.
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

        for both azimuth thrusters. lambda = 1 means that the complete
        requested wrench is achievable; lambda < 1 means the request exceeds
        the available actuator capacity.
        """

        # Start from the uniformly de-rated solution. It is already feasible
        # and satisfies the scaled wrench equality.
        z0, lambda0 = self._uniform_derating(z_unconstrained)
        x0 = np.append(z0, lambda0)  # x = [z, lambda]

        # SLSQP minimises, hence -lambda corresponds to maximising lambda.
        def objective(x: np.ndarray) -> float:
            return -float(x[5])

        # Keep the achieved wrench in the same direction as the requested one.
        def wrench_constraint(x: np.ndarray) -> np.ndarray:
            return self.Be @ x[:5] - x[5] * tau

        # The azimuth limits are discs in the (Fx, Fy) plane. Writing them in
        # squared form avoids the square root in u = sqrt(Fx^2 + Fy^2).
        constraints = [
            {"type": "eq", "fun": wrench_constraint},
            {
                "type": "ineq",
                "fun": lambda x: self.u_max[1] ** 2 - x[1] ** 2 - x[2] ** 2,
            },
            {
                "type": "ineq",
                "fun": lambda x: self.u_max[2] ** 2 - x[3] ** 2 - x[4] ** 2,
            },
        ]

        # Tunnel thrust and lambda have simple box constraints; the azimuth
        # magnitudes are limited by the circular constraints above.
        bounds = [
            (-self.u_max[0], self.u_max[0]),  # u_T
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
            options={"ftol": 1e-9, "maxiter": 100, "disp": False},
        )

        z = np.asarray(result.x[:5], dtype=float)
        lam = float(result.x[5])

        # Do not rely on result.success alone. Verify that the returned point
        # is finite, respects the thrust limits and satisfies the equality.
        residual = np.linalg.norm(self.Be @ z - lam * tau)
        valid = (
            np.all(np.isfinite(z))
            and np.isfinite(lam)
            and self._is_feasible(z, tol=1e-3)
            and -1e-6 <= lam <= 1.0 + 1e-6
            and residual <= 1e-3
        )

        return z if valid else None

    # ------------------------------------------------------------------
    # Convert Cartesian azimuth forces back to physical commands
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

        # Tunnel direction is fixed; only its signed thrust changes.
        u_cmd[0] = z[0] * 1000.0
        alpha_cmd[0] = self.tunnel.alpha0

        # Convert each azimuth from Cartesian force components to (u, alpha).
        for j, idx in enumerate((1, 2)):
            Fx = z[1 + 2 * j]
            Fy = z[2 + 2 * j]
            u = float(np.hypot(Fx, Fy))

            current_angle = None if alpha_now is None else float(alpha_now[idx])

            # At zero thrust the angle has no force effect. Keep the current
            # angle, if known, to avoid an unnecessary azimuth command.
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
    # Public interface called by the simulator once per time step
    # ------------------------------------------------------------------

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate the desired 6-DOF controller wrench to the Part 1 thrusters."""

        # Part 1 allocates surge, sway and yaw only. Use kN/kNm internally so
        # the constrained optimisation is numerically well scaled.
        tau = np.asarray(tau_d[[0, 1, 5]], dtype=float) / 1000.0

        # Use the simple minimum-norm solution during normal operation.
        z = self._minimum_norm_allocation(tau)

        # If a static thrust limit is exceeded, try to redistribute the thrust
        # before reducing the requested wrench magnitude.
        if not self._is_feasible(z):
            z_constrained = self._constrained_allocation(tau, z)

            if z_constrained is not None:
                z = z_constrained
            else:
                # Robust fallback if SLSQP does not return a valid solution.
                z, _ = self._uniform_derating(z)

        # t, dt and u_now belong to the common allocator interface but are not
        # required by this static Part 1 method. alpha_now is used only to
        # choose the equivalent azimuth command with the smallest rotation.
        return self._to_thruster_commands(z, alpha_now)
