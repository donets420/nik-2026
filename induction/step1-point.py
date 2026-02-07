import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk, ellipe

def get_B_vector(x, y, z, M, R, L):
    """
    Расчет вектора магнитной индукции B в точке (x, y, z).
    
    Параметры:
    x, y, z : float - координаты точки наблюдения (м)
    M       : float - намагниченность (А/м)
    R       : float - радиус магнита (м)
    L       : float - длина магнита (м)
    
    Возвращает:
    numpy.array([Bx, By, Bz]) в Теслах (Тл)
    """
    mu0 = 4 * np.pi * 1e-7
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Константа перед интегралом (уравнения 52-53)
    const = (mu0 * M) / (2 * np.pi)

    # Вспомогательная функция для k^2 (модуль эллиптических интегралов)
    def get_k2(zp):
        denom = (R + rho)**2 + (z - zp)**2
        return (4 * R * rho) / denom if denom > 1e-18 else 0

    # Осевая компонента Bz (уравнение 52)
    def integrand_z(zp):
        m = get_k2(zp)
        K, E = ellipk(m), ellipe(m)
        num = R**2 - rho**2 - (z - zp)**2
        den = (R - rho)**2 + (z - zp)**2
        if den < 1e-18: return 0
        return (1 / np.sqrt((R + rho)**2 + (z - zp)**2)) * ((num / den) * E + K)

    # Радиальная компонента Brho (исправленное уравнение 53 для верной полярности)
    def integrand_rho(zp):
        m = get_k2(zp)
        K, E = ellipk(m), ellipe(m)
        num = R**2 + rho**2 + (z - zp)**2
        den = (R - rho)**2 + (z - zp)**2
        if den < 1e-18: return 0
        # Член (z - zp) обеспечивает физически корректное направление (Север на +z)
        return (rho / R) * ((z - zp) / np.sqrt((R + rho)**2 + (z - zp)**2)) * ((num / den) * E - K)

    # Численное интегрирование по длине магнита от -L/2 до L/2
    bz_val, _ = quad(integrand_z, -L/2, L/2)
    
    if rho < 1e-12:
        brho_val = 0
    else:
        brho_val, _ = quad(integrand_rho, -L/2, L/2)
    
    Bz = bz_val * const
    Brho = brho_val * const
    
    # Перевод в декартовы координаты
    Bx = Brho * np.cos(phi)
    By = Brho * np.sin(phi)
    
    return np.array([Bx, By, Bz])

# --- Пример использования ---
# Параметры: R=1см, L=4см, M=10^6 А/м (N42 ~ 1.2T)
# Точка наблюдения: x=0.005, y=0, z=0.03 (над северным торцом)
B = get_B_vector(0.005, 0, 0.03, 1e6, 0.01, 0.04)

print(f"Вектор B (Тл): {B}")
print(f"Модуль |B| (Тл): {np.linalg.norm(B):.4f}")
