import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe

# Constants
MU0 = 4 * np.pi * 1e-7
_EPS_RHO = 1e-10
_EPS_DENOM = 1e-15
_N_ZP = 64  # integration grid size (tune speed vs accuracy)

def get_B_field_correct(x, y, z, M, R, L):
    """
    Исправленный расчет вектора B.
    Направление Bz: снизу вверх (-L/2 -> +L/2).
    Направление Brho: от оси на северном полюсе, к оси на южном.
    Vectorized integration for speed.
    """
    rho = np.sqrt(x*x + y*y)
    rho_eff = max(rho, _EPS_RHO)
    phi = np.arctan2(y, x)

    zp = np.linspace(-L/2, L/2, _N_ZP)
    z_minus_zp = z - zp
    dist_sq_plus = (R + rho_eff)**2 + z_minus_zp**2
    dist_sq_minus = (R - rho_eff)**2 + z_minus_zp**2
    m = np.where(dist_sq_plus > _EPS_DENOM, (4 * R * rho_eff) / dist_sq_plus, 0.0)
    K, E = ellipk(m), ellipe(m)

    safe_minus = np.maximum(dist_sq_minus, _EPS_DENOM)
    sqrt_plus = np.sqrt(dist_sq_plus)

    # Bz integrand (Eq. 52)
    num_z = R**2 - rho_eff**2 - z_minus_zp**2
    integrand_z = (1.0 / sqrt_plus) * (num_z / safe_minus * E + K)

    # Brho integrand (z - zp for correct pole direction)
    num_rho = R**2 + rho_eff**2 + z_minus_zp**2
    integrand_rho = (z_minus_zp / (rho_eff * sqrt_plus)) * (num_rho / safe_minus * E - K)

    bz_val = np.trapezoid(integrand_z, zp)
    brho_val = np.trapezoid(integrand_rho, zp)

    const = (MU0 * M * R) / (2 * np.pi)
    Bz = bz_val * const
    Brho = brho_val * const
    return np.array([Brho * np.cos(phi), Brho * np.sin(phi), Bz])

# Параметры магнита (м)
R, L, M = 0.01, 0.04, 1e6

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Отрисовка магнита (цилиндр)
zc = np.linspace(-L/2, L/2, 20)
th = np.linspace(0, 2*np.pi, 30)
Theta, Zc = np.meshgrid(th, zc)
ax.plot_surface(R*np.cos(Theta), R*np.sin(Theta), Zc, color='red', alpha=0.4)

# 2. Трассировка для ЗАМЫКАНИЯ линий
# Начинаем из плоскости z=0 (центр магнита)
seeds = []
for r_s in [R*0.4, R*0.7, R*1.2]: 
    for p_s in np.linspace(0, 2*np.pi, 8, endpoint=False):
        seeds.append([r_s * np.cos(p_s), r_s * np.sin(p_s), 0])

def _field_dir(t, pos, M, R, L):
    B = get_B_field_correct(pos[0], pos[1], pos[2], M, R, L)
    mag = np.linalg.norm(B)
    return B / mag if mag > 1e-18 else np.zeros(3)

# Single callable for solve_ivp (avoids redefining in loop)
def _make_field_dir(M, R, L):
    return lambda t, pos: _field_dir(t, pos, M, R, L)

field_dir = _make_field_dir(M, R, L)
limit = 1.2
t_span_fwd = np.linspace(0, limit, 300)
t_span_bwd = np.linspace(0, -limit, 300)

print("Идет расчет 3D петель... Пожалуйста, подождите.")
for p0 in seeds:
    sol_f = solve_ivp(field_dir, [0, limit], p0, t_eval=t_span_fwd, rtol=1e-5)
    sol_b = solve_ivp(field_dir, [0, -limit], p0, t_eval=t_span_bwd, rtol=1e-5)
    
    lx = np.concatenate([sol_b.y[0][::-1], sol_f.y[0]])
    ly = np.concatenate([sol_b.y[1][::-1], sol_f.y[1]])
    lz = np.concatenate([sol_b.y[2][::-1], sol_f.y[2]])
    
    # Рисуем только в разумных пределах, чтобы не перегружать график
    mask = (np.abs(lx) < 0.2) & (np.abs(ly) < 0.2) & (np.abs(lz) < 0.2)
    ax.plot3D(lx[mask], ly[mask], lz[mask], color='blue', lw=0.7, alpha=0.5)

ax.set_xlim(-0.1, 0.1); ax.set_ylim(-0.1, 0.1); ax.set_zlim(-0.1, 0.1)
ax.set_title('Исправленное 3D поле: Север (+Z), Юг (-Z)')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.show()
