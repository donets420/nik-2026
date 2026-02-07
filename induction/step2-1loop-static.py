import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk, ellipe

# Порог для знаменателей: избежание деления на ноль в k² и подынтегральном выражении
DENOM_MIN = 1e-18

def get_B_z_only(rho, z, M, R, L):
    """
    Выделенная часть вашего кода для расчета только компоненты Bz.
    Это ускоряет расчет потока, так как Brho нам не нужна.
    """
    mu0 = 4 * np.pi * 1e-7
    const = (mu0 * M) / (2 * np.pi)

    def get_k2(zp):
        denom = (R + rho)**2 + (z - zp)**2
        if denom <= DENOM_MIN:
            return 0
        m = (4 * R * rho) / denom
        return min(max(m, 0), 1)  # ellipk, ellipe определены для m in [0, 1]

    def integrand_z(zp):
        m = get_k2(zp)
        K, E = ellipk(m), ellipe(m)
        num = R**2 - rho**2 - (z - zp)**2
        den = (R - rho)**2 + (z - zp)**2
        if den < DENOM_MIN:
            return 0
        return (1 / np.sqrt((R + rho)**2 + (z - zp)**2)) * ((num / den) * E + K)

    bz_val, _ = quad(integrand_z, -L/2, L/2, limit=100, epsrel=1e-9)
    return bz_val * const

def get_flux(rk, zk, M, Rm, L):
    """
    Расчет потока через кольцо радиусом rk на высоте zk.
    Возвращает (поток в Вб, оценка погрешности quadrature).
    """
    def flux_integrand(r):
        return get_B_z_only(r, zk, M, Rm, L) * 2 * np.pi * r

    flux, error = quad(flux_integrand, 0, rk, limit=100, epsrel=1e-8)
    return flux, error

if __name__ == "__main__":
    # --- Параметры расчета ---
    M_val = 1e6      # Намагниченность (А/м)
    R_magnet = 0.01  # Радиус магнита (м)
    L_magnet = 0.04  # Длина магнита (м)
    R_ring = 0.015   # Радиус кольца (м) - больше магнита
    Z_ring = 0.03    # Высота кольца относительно центра магнита (м)

    flux_total, flux_error = get_flux(R_ring, Z_ring, M_val, R_magnet, L_magnet)

    print("Параметры системы:")
    print(f"  Магнит: R={R_magnet*1000}мм, L={L_magnet*1000}мм, M={M_val} А/м")
    print(f"  Кольцо: R={R_ring*1000}мм, Z={Z_ring*1000}мм")
    print("-" * 30)
    print(f"Магнитный поток через кольцо: {flux_total:.8e} Вб")
    print(f"Оценка погрешности quadrature: {flux_error:.4e} Вб")
