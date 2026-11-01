"""Verify the nominal continuous-time NPO passivity conditions.

Place in the TMR4240_DP_LAB project root and run:
    python3.11 verify_npo_kyp.py

Reference: Sørensen (2018), Marine Cybernetics, Sec. 7.2.4,
Eqs. (7.23) and (7.28)-(7.29).
This script does NOT prove stability of the full discretised DP simulator.
"""

import numpy as np
from scipy.optimize import minimize
from part_2.observer import NonlinearPassiveObserver


def smallest_eigenvalue(matrix):
    # A symmetric matrix is positive definite if all eigenvalues are positive.
    return float(np.linalg.eigvalsh((matrix + matrix.T) / 2)[0])


def rank_tests(A, B, C):
    # Check that the error system is controllable and observable.
    n = A.shape[0]
    reachability = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    observability = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    return np.linalg.matrix_rank(reachability), np.linalg.matrix_rank(observability)


def main():
    obs = NonlinearPassiveObserver()  # Read the gains from the actual NPO class.
    I = np.eye(3)
    Z = np.zeros((3, 3))

    # Error dynamics from Sørensen Eq. (7.23), G = 0 for the DP vessel.
    A = np.block([
        [obs.Aw - obs.K1 @ obs.Cw, -obs.K1, np.zeros((6, 3))],
        [-obs.K2 @ obs.Cw,       -obs.K2, Z],
        [-obs.K3 @ obs.Cw,       -obs.K3, -np.diag(1.0 / obs.T_bias)],
    ])
    B = np.vstack((np.zeros((6, 3)), I, Z))
    C = np.hstack((obs.K4 @ obs.Cw, obs.K4, -I))

    print('NPO parameters read from part_2/observer.py')
    print(f'  Wave frequency:     {obs.omega_w:.6f} rad/s')
    print(f'  Cutoff frequency:   {obs.K2[0, 0]:.6f} rad/s')
    print(f'  Wave damping:       {obs.zeta_w:.3f}')
    print(f'  Desired damping:    {obs.zeta_desired:.3f}')
    print(f'  Bias time constants: {obs.T_bias} s')
    print(f'  K4 diagonal:        {np.diag(obs.K4)}')
    print(f'  K3 diagonal:        {np.diag(obs.K3)}')
    print(f'  K1 top diagonal:    {np.diag(obs.K1[:3, :])}')
    print(f'  K1 bottom diagonal: {np.diag(obs.K1[3:, :])}')
    print(f'  K2 diagonal:        {np.diag(obs.K2)}')

    # The 12-state error model must separate into 3 independent 4-state axes.
    indices = [[i, i + 3, i + 6, i + 9] for i in range(3)]
    layout = np.zeros((12, 12), dtype=bool)
    for idx in indices:
        layout[np.ix_(idx, idx)] = True
    if np.max(np.abs(A[~layout])) > 1e-10:
        raise ValueError('A has cross-axis terms: the 4-state reduction is invalid.')
    if np.min(np.diag(obs.K4)) <= 0:
        raise ValueError('K4 must have positive diagonal entries.')
    if not np.allclose(obs.K4, np.diag(np.diag(obs.K4))):
        raise ValueError('This script requires diagonal K4.')

    # Strictly positive numerical margins (in the SCALED coordinates).
    eps_P = 0.02
    eps_Q = 0.002
    # Numerical start point, not a fixed P result. The optimiser finds P.
    start = np.array([0.18, 0.0045, 0.069, 1.30, -0.018, 13.5])

    # Build the full P from independently solved 4x4 matrices.
    P = np.zeros((12, 12))
    axis_ok = True

    for axis, idx in enumerate(indices):
        k4 = obs.K4[axis, axis]
        S = np.diag([1.0, 1.0, 1.0, 1.0 / k4])
        S_inv = np.diag([1.0, 1.0, 1.0, k4])
        As = S @ A[np.ix_(idx, idx)] @ S_inv
        Bs = S @ B[idx, axis]
        Cs = (C[axis, idx] @ S_inv) / k4

        # For this model Bs = [0, 0, 1, 0]. Thus the third row of
        # Ps is fixed by Bs.T @ Ps = Cs; only six values remain unknown.
        if not np.allclose(Bs, [0, 0, 1, 0], atol=1e-10):
            raise ValueError('Unexpected Bs: cannot use the reduced P parameterisation.')

        def make_Ps(x):
            a, b, c, d, e, f = x
            return np.array([
                [a, b, Cs[0], c],
                [b, d, Cs[1], e],
                [Cs[0], Cs[1], Cs[2], Cs[3]],
                [c, e, Cs[3], f],
            ])

        def make_Qs(x):
            Ps = make_Ps(x)
            return -(As.T @ Ps + Ps @ As)

        # Find Ps numerically subject to the two KYP positivity requirements.
        result = minimize(
            fun=lambda x: float(np.sum((x - start) ** 2)),
            x0=start.copy(),
            method='SLSQP',
            constraints=[
                {'type': 'ineq', 'fun': lambda x: smallest_eigenvalue(make_Ps(x)) - eps_P},
                {'type': 'ineq', 'fun': lambda x: smallest_eigenvalue(make_Qs(x)) - eps_Q},
            ],
            options={'maxiter': 1000, 'ftol': 1e-12},
        )
        if not result.success:
            print(f'  Axis {axis + 1}: FAIL - optimiser: {result.message}')
            axis_ok = False
            continue

        Ps = make_Ps(result.x)
        Qs = -(As.T @ Ps + Ps @ As)
        eq = float(np.max(np.abs(Bs @ Ps - Cs)))
        eigP = smallest_eigenvalue(Ps)
        eigQ = smallest_eigenvalue(Qs)
        max_real = float(np.max(np.linalg.eigvals(As).real))
        rank_c, rank_o = rank_tests(As, Bs[:, None], Cs[None, :])
        valid = (eigP >= eps_P - 1e-7 and eigQ >= eps_Q - 1e-7
                 and eq < 1e-8 and max_real < 0
                 and rank_c == 4 and rank_o == 4)
        axis_ok &= valid

        print(f'\nAxis {axis + 1} (North, East, Heading): {"PASS" if valid else "FAIL"}')
        print(f'  min eig(Ps)        = {eigP:.8g}')
        print(f'  min eig(Qs)        = {eigQ:.8g}')
        print(f'  max |Bs.T Ps - Cs| = {eq:.3e}')
        print(f'  max Re(eig(As))    = {max_real:.8g}')
        print(f'  ranks (ctrl/obs)   = {rank_c}/4, {rank_o}/4')
        if axis == 0:
            print('  Calculated Ps for North (scaled):')
            print(np.array2string(Ps, precision=6, suppress_small=True))
            print('  Calculated Qs for North (scaled):')
            print(np.array2string(Qs, precision=6, suppress_small=True))

        # Undo the scaling: original 12-state model uses physical bias units.
        P[np.ix_(idx, idx)] = k4 * (S.T @ Ps @ S)

    if not axis_ok:
        print('\nOVERALL: FAIL - KYP certificate not established.')
        return 1

    # Verify KYP again in the ORIGINAL 12-state coordinates.
    Q = -(A.T @ P + P @ A)
    eigP_full = smallest_eigenvalue(P)
    eigQ_full = smallest_eigenvalue(Q)
    error_full = float(np.max(np.abs(B.T @ P - C)))
    max_real_full = float(np.max(np.linalg.eigvals(A).real))
    rank_c_full, rank_o_full = rank_tests(A, B, C)
    mass_sym = np.allclose(obs.M, obs.M.T)
    mass_ok = mass_sym and smallest_eigenvalue(obs.M) > 0
    damping_sym = np.allclose(obs.D, obs.D.T)
    damping_ok = damping_sym and smallest_eigenvalue(obs.D) > 0

    passed = (eigP_full > 0 and eigQ_full > 0 and error_full < 1e-7
              and max_real_full < 0 and rank_c_full == 12 and rank_o_full == 12
              and mass_ok and damping_ok)

    print('\nFull 12-state nominal continuous-time model:')
    print(f'  min eig(P)         = {eigP_full:.8g}')
    print(f'  min eig(Q)         = {eigQ_full:.8g}')
    print(f'  max |B.T P - C|    = {error_full:.3e}')
    print(f'  max Re(eig(A))     = {max_real_full:.8g}')
    print(f'  controllability    = {rank_c_full}/12')
    print(f'  observability      = {rank_o_full}/12')
    print(f'  M symmetric, pos.  = {mass_ok}')
    print(f'  D symmetric, pos.  = {damping_ok}')
    print(f'\nOVERALL: {"PASS" if passed else "FAIL"}')
    print('Scope: nominal continuous-time model, not a full discrete-time stability proof.')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())