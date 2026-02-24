import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit

# ============================================================
# 1) CONSTANTS
# ============================================================
MU0 = 4.0 * np.pi * 1e-7  # [T·m/A]


# ============================================================
# 2) ELLIPTIC-INTEGRAL MAGNET FIELD (whole elliptic integrals method)
# ============================================================
def cel(kc, p, c, s):
    """Bulirsch generalized elliptic integral (via Carlson forms)."""
    kc = np.asarray(kc, dtype=float)
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)

    # kc=0 corresponds to k=1 singularity; avoid feeding into Carlson directly
    singular_mask = (kc == 0.0)
    kc_safe = np.where(singular_mask, 1.0, kc)

    kc2 = kc_safe ** 2
    rf = special.elliprf(0.0, kc2, 1.0)
    rj = special.elliprj(0.0, kc2, 1.0, p)

    result = c * rf + (s - p * c) * (rj / 3.0)

    if np.any(singular_mask):
        result = np.asarray(result)
        result[singular_mask] = np.nan

    return result


def B_cylindrical_permanent_magnet(rho, z, a, b, *, mu):
    """
    Field of a cylindrical permanent magnet using elliptic integrals.

    Geometry:
      - radius = a
      - half-length = b  (total length = 2b)
      - magnet centered at z=0, magnet spans z in [-b, +b]
      - mu = magnetic dipole moment [A·m^2] = M * Volume

    Returns:
      Brho, Bz in cylindrical coordinates [T]
    """
    if a <= 0 or b <= 0:
        raise ValueError("Magnet geometry must have a>0 and b>0.")

    # Magnetization M (A/m)
    nI = mu / (2.0 * b * np.pi * a ** 2)

    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)

    B0 = (MU0 / np.pi) * nI
    Brho = np.zeros_like(rho)
    Bz = np.zeros_like(rho)

    # on-axis special case
    on_axis = np.isclose(rho, 0.0, atol=1e-12)
    if np.any(on_axis):
        zp = z[on_axis] + b
        zm = z[on_axis] - b
        Bz[on_axis] = 0.5 * MU0 * nI * (
            zp / np.sqrt(zp ** 2 + a ** 2) - zm / np.sqrt(zm ** 2 + a ** 2)
        )

    # off-axis: elliptic integrals
    off = ~on_axis
    if np.any(off):
        rr = rho[off]
        zz = z[off]
        z_p, z_m = zz + b, zz - b

        d_p = np.sqrt(z_p ** 2 + (rr + a) ** 2)
        d_m = np.sqrt(z_m ** 2 + (rr + a) ** 2)

        gamma = (a - rr) / (a + rr)
        p = gamma ** 2

        k_p = np.sqrt((z_p ** 2 + (a - rr) ** 2) / (z_p ** 2 + (a + rr) ** 2))
        k_m = np.sqrt((z_m ** 2 + (a - rr) ** 2) / (z_m ** 2 + (a + rr) ** 2))

        Brho[off] = B0 * (
            (a / d_p) * cel(k_p, 1, 1, -1) - (a / d_m) * cel(k_m, 1, 1, -1)
        )

        term_p = (z_p / d_p) * cel(k_p, p, 1, gamma)
        term_m = (z_m / d_m) * cel(k_m, p, 1, gamma)
        Bz[off] = B0 * (term_p - term_m)

    return Brho, Bz


# ============================================================
# 3) EXPERIMENT GEOMETRY: distance from MAGNET CENTER along axis
# ============================================================
def Bz_from_center_axis(z_m, mu, a, b, z_shift=0.0, B_offset=0.0):
    """
    Excel provides z measured from the magnet center.
    Mapping: z_model = z_excel + z_shift
    """
    z_m = np.asarray(z_m, dtype=float)
    z = z_m + z_shift
    rho = np.zeros_like(z)
    _, Bz = B_cylindrical_permanent_magnet(rho, z, a=a, b=b, mu=mu)
    return Bz + B_offset


# ============================================================
# 4) OPTIONAL: PROBE ACTIVE-AREA AVERAGING (disk average)
# ============================================================
def Bz_probe_averaged_from_center(z_m, mu, a, b, r_sensor,
                                  z_shift=0.0, B_offset=0.0,
                                  Nr=120):
    """
    Disk-averaged Bz over sensor radius r_sensor at each z.

    Excel provides z measured from magnet center.
    """
    z_m = np.asarray(z_m, dtype=float)
    z = z_m + z_shift

    if r_sensor <= 0:
        return Bz_from_center_axis(z_m, mu, a, b, z_shift=z_shift, B_offset=B_offset)

    r = np.linspace(0.0, r_sensor, Nr)
    w = r
    denom = np.trapz(w, r)

    out = np.empty_like(z)
    for i, zi in enumerate(z):
        R = r
        Z = np.full_like(r, zi)
        _, Bz_r = B_cylindrical_permanent_magnet(R, Z, a=a, b=b, mu=mu)
        out[i] = np.trapz(Bz_r * w, r) / denom

    return out + B_offset


# ============================================================
# 5) LOAD DATA (Excel)
# ============================================================
def load_excel_distance_B(excel_path):
    """
    Expects:
      - first 2 rows are meta/header (skipped)
      - then two columns: z [mm] from center, B [mT]
    """
    df = pd.read_excel(excel_path, skiprows=2, header=None, names=["z_mm", "B_mT"])
    z_m = df["z_mm"].to_numpy(dtype=float) * 1e-3
    B_T = df["B_mT"].to_numpy(dtype=float) * 1e-3
    return z_m, B_T


# ============================================================
# 6) FIT
# ============================================================
def fit_mu_only(z_m, B_T, a, b, mu_guess=10.0):
    """Fit only mu (no z_shift, no B_offset)."""
    def f(z, mu):
        return Bz_from_center_axis(z, mu, a=a, b=b, z_shift=0.0, B_offset=0.0)

    popt, pcov = curve_fit(
        f, z_m, B_T,
        p0=[mu_guess],
        bounds=([0.0], [np.inf]),
        maxfev=20000
    )
    mu_fit = popt[0]
    mu_std = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else np.nan
    return mu_fit, mu_std


def fit_mu_zshift_offset(z_m, B_T, a, b,
                         mu_guess=10.0, z_shift_guess=0.0, B_offset_guess=0.0,
                         use_probe_average=False, r_sensor=0.0):
    """
    Fit mu + z_shift + B_offset
    """
    if not use_probe_average:
        def f(z, mu, z_shift, B_offset):
            return Bz_from_center_axis(z, mu, a=a, b=b, z_shift=z_shift, B_offset=B_offset)
    else:
        def f(z, mu, z_shift, B_offset):
            return Bz_probe_averaged_from_center(
                z, mu, a=a, b=b, r_sensor=r_sensor, z_shift=z_shift, B_offset=B_offset
            )

    popt, pcov = curve_fit(
        f, z_m, B_T,
        p0=[mu_guess, z_shift_guess, B_offset_guess],
        bounds=(
            [0.0,   -0.02,   -0.05],   # mu>=0, z_shift±2cm, offset±50mT
            [np.inf, +0.02,   +0.05]
        ),
        maxfev=80000
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


# ============================================================
# 7) MAIN
# ============================================================
if __name__ == "__main__":
    # --- magnet geometry ---
    magnet_length_m = 0.05
    magnet_diameter_m = 0.02
    a = magnet_diameter_m / 2.0
    b = magnet_length_m / 2.0

    # --- data file ---
    excel_path = r"C:\Users\petrn\Desktop\GYPT26\Experiments\Regio\Prelim\Dipole_Moment\M_D_5.xlsx"
    z_m, B_T = load_excel_distance_B(excel_path)

    # --- choose model ---
    use_probe_average = False
    r_sensor = 0.0015  # [m] (only used if use_probe_average=True)

    # --- fit ---
    mu_guess = 16.0
    (mu_fit, z_shift_fit, B_offset_fit), perr = fit_mu_zshift_offset(
        z_m, B_T, a=a, b=b,
        mu_guess=mu_guess,
        z_shift_guess=0.0,
        B_offset_guess=0.0,
        use_probe_average=use_probe_average,
        r_sensor=r_sensor
    )

    print("Fit result (elliptic-integrals magnet model):")
    print(f"  mu       = {mu_fit:.6g} A·m^2   (1σ ≈ {perr[0]:.3g})")
    print(f"  z_shift  = {z_shift_fit*1e3:.6g} mm     (1σ ≈ {perr[1]*1e3:.3g} mm)")
    print(f"  B_offset = {B_offset_fit*1e3:.6g} mT     (1σ ≈ {perr[2]*1e3:.3g} mT)")

    # --- evaluate fitted curve ---
    if not use_probe_average:
        B_fit = Bz_from_center_axis(
            z_m, mu_fit,
            a=a, b=b,
            z_shift=z_shift_fit,
            B_offset=B_offset_fit
        )
        model_label = "Fit (elliptic integrals, axis)"
    else:
        B_fit = Bz_probe_averaged_from_center(
            z_m, mu_fit,
            a=a, b=b,
            r_sensor=r_sensor,
            z_shift=z_shift_fit,
            B_offset=B_offset_fit
        )
        model_label = f"Fit (elliptic integrals + probe avg, r={r_sensor * 1e3:.2f} mm)"

    rmse_mT = np.sqrt(np.mean((B_fit - B_T) ** 2)) * 1e3
    print(f"  RMSE     = {rmse_mT:.4g} mT")

    # --- plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(z_m * 1e3, B_T * 1e3, "o", label="Measured")
    plt.plot(z_m * 1e3, B_fit * 1e3, "-", label=model_label)
    plt.xlabel("Distance from magnet center z [mm]")
    plt.ylabel("Bz [mT]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
