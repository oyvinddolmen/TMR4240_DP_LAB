import numpy as np
from part_1.wind import Wind

# --- Recommended sanity check from the assignment paper ---
# psi = 0 (bow north), wind FROM north (blows towards south)
# Expected: alpha_rw = 180 deg, negative surge force (Fx < 0), near-zero sway (Fy ~ 0)
##BETA IN RADIANS
wind = Wind(mean_speed=10.0, beta=np.pi, semantics="towards")  # 10 m/s wind from the North

eta = np.zeros(6)   # vessel at rest, psi = eta[5] = 0 (bow north)
nu = np.zeros(6)    # vessel not moving, so relative wind = ambient wind

tau_w6, info = wind.step(t=0.0, dt=0.05, eta=eta, nu=nu)

print(f"alpha_rw (deg) = {np.degrees(info['alpha_body']):.2f}   (expected: 180.00)")
print(f"Fx (surge)     = {tau_w6[0]:.3f}   (expected: negative)")
print(f"Fy (sway)      = {tau_w6[1]:.3f}   (expected: near zero)")
print(f"Mz (yaw)       = {tau_w6[5]:.3f}")