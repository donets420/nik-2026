import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk, ellipe

# Порог для знаменателей
DENOM_MIN = 1e-18

def get_B_z_only(rho, z, M, R, L):
    """
    Расчет Bz компоненты поля цилиндрического источника (магнита/соленоида).
    """
    mu0 = 4 * np.pi * 1e-7
    const = (mu0 * M) / (2 * np.pi)

    def get_k2(zp):
        denom = (R + rho)**2 + (z - zp)**2
        if denom <= DENOM_MIN:
            return 0
        m = (4 * R * rho) / denom
        return min(max(m, 0), 1)

    def integrand_z(zp):
        m = get_k2(zp)
        K, E = ellipk(m), ellipe(m)
        num = R**2 - rho**2 - (z - zp)**2
        den = (R - rho)**2 + (z - zp)**2
        if den < DENOM_MIN:
            return 0
        return (1 / np.sqrt((R + rho)**2 + (z - zp)**2)) * ((num / den) * E + K)

    # limit и epsrel влияют на скорость. Тройной интеграл очень требователен.
    bz_val, _ = quad(integrand_z, -L/2, L/2, limit=50, epsrel=1e-5)
    return bz_val * const

def get_flux(rk, zk, M, Rm, L):
    """
    Расчет потока Ф через один виток радиусом rk на высоте zk.
    """
    def flux_integrand(r):
        # Интегрируем Bz * 2*pi*r по радиусу от 0 до rk
        return get_B_z_only(r, zk, M, Rm, L) * 2 * np.pi * r

    flux, error = quad(flux_integrand, 0, rk, limit=50, epsrel=1e-5)
    return flux, error

def get_coil_flux_linkage(z_center, L_coil, R_coil, n_density, M, R_source, L_source):
    """
    Расчет полного потокосцепления (Flux Linkage) для соленоида.
    
    Аргументы:
      z_center: координата центра катушки-приемника (м)
      L_coil: длина катушки-приемника (м)
      R_coil: радиус катушки-приемника (м)
      n_density: плотность намотки (витков на метр)
      M, R_source, L_source: параметры источника поля
    
    Возвращает:
      (Потокосцепление [Вб], Оценка погрешности)
    """
    z_start = z_center - L_coil / 2
    z_end = z_center + L_coil / 2

    # Функция для интегрирования по длине катушки: Ф(z) * n
    def d_psi_dz(z):
        phi, _ = get_flux(R_coil, z, M, R_source, L_source)
        return phi * n_density

    # Интегрируем распределение потока по длине катушки
    # Используем epsrel побольше (1e-4), так как это внешний цикл тройного интеграла
    psi_total, error = quad(d_psi_dz, z_start, z_end, limit=50, epsrel=1e-4)
    
    return psi_total, error

if __name__ == "__main__":
    # --- Параметры источника (Магнит или первичная катушка) ---
    M_val = 1e6        # Намагниченность (А/м). Если источник катушка: M = I * n_source
    R_source = 0.01    # Радиус источника (м)
    L_source = 0.04    # Длина источника (м)

    # --- Параметры приемника (Катушка, для которой считаем поток) ---
    R_coil = 0.015     # Радиус намотки (м)
    L_coil = 0.02      # Длина намотки (м)
    Z_coil = 0.0      # Смещение центра катушки относительно центра источника (м)
    N_turns = 100      # Общее количество витков
    
    # Плотность намотки n = N / L
    n_density = N_turns / L_coil 

    print("Начат расчет (это может занять время из-за тройного интегрирования)...")
    
    psi_val, psi_err = get_coil_flux_linkage(
        Z_coil, L_coil, R_coil, n_density, 
        M_val, R_source, L_source
    )

    print("-" * 40)
    print(f"Источник: R={R_source*1000} мм, L={L_source*1000} мм, M={M_val:.1e} А/м")
    print(f"Приемник: R={R_coil*1000} мм, L={L_coil*1000} мм, Z_center={Z_coil*1000} мм")
    print(f"Обмотка:  N={N_turns} витков, Плотность n={n_density:.1f} вит/м")
    print("-" * 40)
    print(f"Полное потокосцепление (Psi): {psi_val:.8e} Вб")
    print(f"Средний поток на виток:       {psi_val/N_turns:.8e} Вб")
    print(f"Оценка погрешности расчёта:   {psi_err:.2e}")