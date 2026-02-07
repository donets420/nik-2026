import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import ellipk, ellipe
import time

# --- ФИЗИЧЕСКОЕ ЯДРО (ОПТИМИЗИРОВАНО) ---

DENOM_MIN = 1e-18

# Quadrature tuning (reproducibility and speed vs accuracy)
QUAD_LIMIT_BZ = 30
QUAD_EPSREL_BZ = 1e-4
QUAD_LIMIT_FLUX = 30
QUAD_EPSREL_FLUX = 1e-3
QUAD_LIMIT_PSI = 20
QUAD_EPSREL_PSI = 1e-3

def get_B_z_only(rho, z, M, R, L):
    """Рачет Bz компоненты поля."""
    mu0 = 4 * np.pi * 1e-7
    const = (mu0 * M) / (2 * np.pi)

    def get_k2(zp):
        denom = (R + rho)**2 + (z - zp)**2
        if denom <= DENOM_MIN: return 0
        m = (4 * R * rho) / denom
        return min(max(m, 0), 1)

    def integrand_z(zp):
        m = get_k2(zp)
        K, E = ellipk(m), ellipe(m)
        num = R**2 - rho**2 - (z - zp)**2
        den = (R - rho)**2 + (z - zp)**2
        if den < DENOM_MIN: return 0
        return (1 / np.sqrt((R + rho)**2 + (z - zp)**2)) * ((num / den) * E + K)

    bz_val, _ = quad(integrand_z, -L/2, L/2, limit=QUAD_LIMIT_BZ, epsrel=QUAD_EPSREL_BZ)
    return bz_val * const

def get_flux_through_loop(rk, zk, M, Rm, L):
    """Поток через один виток (интеграл по радиусу)."""
    def flux_integrand(r):
        return get_B_z_only(r, zk, M, Rm, L) * 2 * np.pi * r
    
    flux, _ = quad(flux_integrand, 0, rk, limit=QUAD_LIMIT_FLUX, epsrel=QUAD_EPSREL_FLUX)
    return flux

def get_coil_flux_linkage(z_center_rel, L_coil, R_coil, n_density, M, R_source, L_source):
    """
    Полное потокосцепление (Psi).
    z_center_rel: позиция центра катушки ОТНОСИТЕЛЬНО центра магнита.
    """
    z_start = z_center_rel - L_coil / 2
    z_end = z_center_rel + L_coil / 2

    def d_psi_dz(z):
        # Поток через виток на высоте z * плотность намотки
        return get_flux_through_loop(R_coil, z, M, R_source, L_source) * n_density

    # Интегрируем вдоль длины катушки
    psi_total, _ = quad(d_psi_dz, z_start, z_end, limit=QUAD_LIMIT_PSI, epsrel=QUAD_EPSREL_PSI)
    return psi_total

# --- ПАРАМЕТРЫ СИМУЛЯЦИИ ---

# 1. Магнит
M_val = 1e6
R_magnet = 0.01
L_magnet = 0.04

# 2. Катушка (Неподвижная в лабораторной системе)
R_coil = 0.015
L_coil = 0.02
Z_coil_lab = 0.0 
N_turns = 100
n_density = N_turns / L_coil

# 3. Движение магнита: z(t) = A * sin(omega * t)
Amplitude = 0.02    # Амплитуда колебаний 2 см
Frequency = 1.0     # Частота 1 Гц
Omega = 2 * np.pi * Frequency

# Временная шкала
T_period = 1 / Frequency
num_points = 25  # Количество точек для графика (чем больше, тем дольше расчет!)
t_values = np.linspace(0, T_period, num_points)
psi_values = []

print(f"Запуск симуляции для {num_points} точек времени...")
print(f"Частота: {Frequency} Гц, Амплитуда: {Amplitude*1000} мм")

start_time = time.time()

for i, t in enumerate(t_values):
    # Текущее положение магнита
    z_mag = Amplitude * np.sin(Omega * t)
    
    # Положение катушки ОТНОСИТЕЛЬНО магнита
    # Если магнит сдвинулся вверх (+z), катушка относительно него сдвинулась вниз
    z_rel = Z_coil_lab - z_mag
    
    psi = get_coil_flux_linkage(z_rel, L_coil, R_coil, n_density, M_val, R_magnet, L_magnet)
    psi_values.append(psi)
    
    # Прогресс бар
    if i % 5 == 0:
        print(f"Calculated {i}/{num_points} points... (t={t:.3f}s)")

total_time = time.time() - start_time
print(f"Расчет завершен за {total_time:.2f} сек.")

# --- ПОСТРОЕНИЕ ГРАФИКА ---

plt.figure(figsize=(10, 9))

# Верхний график: Движение магнита
plt.subplot(3, 1, 1)
plt.plot(t_values, Amplitude * np.sin(Omega * t_values) * 1000, 'r--')
plt.ylabel('Z магнита (мм)')
plt.title('Движение магнита')
plt.grid(True)

# Средний график: Потокосцепление
plt.subplot(3, 1, 2)
plt.plot(t_values, psi_values, 'b.-', linewidth=2)
plt.ylabel('Потокосцепление $\Psi$ (Вб)')
plt.title('Изменение потокосцепления со временем')
plt.grid(True)

plt.tight_layout()
plt.show()