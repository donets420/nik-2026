import numpy as np
import matplotlib.pyplot as plt
from step1_1_M_FIeld_Precomputation import MagnetFieldTable

def validate_coil_in_grid(tbl: MagnetFieldTable,
                         z_center: float, L_coil: float, R_coil: float,
                         z_mag_range=None) -> None:
    """

    Validate that a coil lies within the field grid for the full magnet motion range.

    The field table is in the magnet's rest frame. When the magnet is at lab position z_mag,
    the field at lab point (rho, z) is Bz(rho, z - z_mag). Validation ensures that for all
    z_mag in z_mag_range, the coil extent maps to valid table coordinates.

    Parameters
    ----------
    tbl : Precomputed field table.
    z_center, L_coil, R_coil : Coil geometry [m].
    z_mag_range : tuple (z_mag_min, z_mag_max) or None
        Range of magnet lab position z_mag(t) over the full motion [m].
        If None, assumes fixed magnet at z_mag=0.
    """
    z_start = z_center - L_coil / 2.0
    z_end = z_center + L_coil / 2.0

    if R_coil > tbl.rho_max:
        raise ValueError(
            f"Coil radius R_coil={R_coil} exceeds field grid rho_max={tbl.rho_max}. "
            "Results would silently use fill_value=0 outside the grid."
        )

    if z_mag_range is None:
        z_mag_min, z_mag_max = 0.0, 0.0
    else:
        z_mag_min, z_mag_max = z_mag_range

    # When magnet at z_mag, we evaluate Bz(rho, z_lab - z_mag) for z_lab in [z_start, z_end].
    # Table z must be in [z_min, z_max], so z_min <= z_lab - z_mag <= z_max.
    # Worst case: z_start - z_mag_max (min table z) and z_end - z_mag_min (max table z).
    z_table_min = z_start - z_mag_max
    z_table_max = z_end - z_mag_min
    if z_table_min < tbl.z_min or z_table_max > tbl.z_max:
        raise ValueError(
            f"Coil axial extent [{z_start}, {z_end}] with magnet range z_mag in [{z_mag_min}, {z_mag_max}] "
            f"maps to table z in [{z_table_min}, {z_table_max}], which exceeds grid [z_min={tbl.z_min}, z_max={tbl.z_max}]. "
            "Results would silently use fill_value=0 outside the grid."
        )


def get_coil_flux_linkage(tbl: MagnetFieldTable,
                         z_center: float, L_coil: float, R_coil: float,
                         n_density: float,
                         z_mag: float = 0.0,
                         z_mag_range=None,
                         Nr: int = 100, Nz: int = 100):
    """
    Total flux linkage Ψ for an axially distributed coil (thin-walled, single radius R_coil)
    with uniform turn density n_density [turns/m]. Magnet at lab position z_mag.

        Ψ = n_density * ∫∫ Bz(r, z_lab - z_mag) 2πr dr dz

    Uses 2D grid + vectorized quadrature (single Bz evaluation, double trapezoidal rule).

    Parameters
    ----------
    tbl : Precomputed field table (magnet rest frame).
    z_center, L_coil, R_coil : Coil geometry [m].
    n_density : Turn density [turns/m].
    z_mag : Magnet lab position [m] at the instant of evaluation (default 0).
    z_mag_range : tuple (z_mag_min, z_mag_max) or None
        Full range of magnet motion for grid validation. If None, uses (z_mag, z_mag).
    Nr, Nz : Number of radial and axial grid points (default 50 each).

    Returns
    -------
    psi : Total flux linkage [Wb·turn].
    """
    z_mag_range = (z_mag, z_mag) if z_mag_range is None else z_mag_range
    validate_coil_in_grid(tbl, z_center, L_coil, R_coil, z_mag_range=z_mag_range)

    z_start = z_center - L_coil / 2.0
    z_end = z_center + L_coil / 2.0

    r_vals = np.linspace(0.0, R_coil, Nr)
    z_vals = np.linspace(z_start, z_end, Nz)
    R, Z = np.meshgrid(r_vals, z_vals)

    # Field in magnet frame: Bz(r, z_lab - z_mag)
    integrand = tbl.Bz_cyl(R, Z - z_mag) * 2.0 * np.pi * R

    psi = n_density * np.trapezoid(np.trapezoid(integrand, r_vals, axis=1), z_vals, axis=0)
    return psi

def flux_linkage_vs_time(tbl: MagnetFieldTable,
                        z_center: float, L_coil: float, R_coil: float,
                        n_density: float,
                        t_vals, z_mag_func,
                        z_mag_range=None,
                        Nr: int = 50, Nz: int = 50):
    """
    Compute flux linkage Ψ(t) for magnet moving along z with position z_mag(t).

    Parameters
    ----------
    tbl : Precomputed field table.
    z_center, L_coil, R_coil : Coil geometry [m].
    n_density : Turn density [turns/m].
    t_vals : Time points [s].
    z_mag_func : Magnet position [m] as function of time: z_mag(t).
    z_mag_range : tuple (z_mag_min, z_mag_max) or None
    If None, inferred from t_vals.
    Nr, Nz : Grid points for integration.

    Returns
    -------
    t_vals : Time points (same as input, as array).
    psi_vals : Flux linkage [Wb·turn] at each time.
    """
    t_vals = np.asarray(t_vals)
    z_mag_at_t = np.array([float(z_mag_func(t)) for t in t_vals])
    if z_mag_range is None:
        z_mag_range = (float(np.min(z_mag_at_t)), float(np.max(z_mag_at_t)))
    psi_vals = np.array([
        get_coil_flux_linkage(tbl, z_center, L_coil, R_coil, n_density,
                             z_mag=z_mag_at_t[i], z_mag_range=z_mag_range, Nr=Nr, Nz=Nz)
        for i in range(len(t_vals))
    ])
    return t_vals, psi_vals


# -------------------- Example usage --------------------

filename = "magnet_field_L10cm_D2cm_mu3.npz"
tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

# Coil parameters
z_center = 0.0      # coil's centered relative to the magnet
L_coil   = 0.1      # coil'S length
R_coil   = 0.015    # coil's radius
N_turns = 1000      # turn amount
n_density = N_turns / L_coil  # [turns/m]

# Magnet motion: z_mag(t) = position of magnet center along z axis [m]
def z_mag(t):
    # Example: sinusoidal motion with amplitude 0.05 m
    return -0.05 * np.sin(2 * np.pi * 1 * t)

# Full range of magnet motion for grid validation
z_mag_min, z_mag_max = -0.05, 0.05
z_mag_range = (z_mag_min, z_mag_max)

# Flux linkage at t=0 (magnet centered at z=0)
psi_0 = get_coil_flux_linkage(tbl, z_center, L_coil, R_coil, n_density,
                              z_mag=0.0, z_mag_range=z_mag_range)

print(f"Psi(t=0) = {psi_0:.6e} Wb·turn")

# Flux linkage vs time
t_vals = np.linspace(0, 1.0, 101)
t_arr, psi_arr = flux_linkage_vs_time(tbl, z_center, L_coil, R_coil, n_density,
                                      t_vals, z_mag, z_mag_range=z_mag_range)
# Plot
plt.plot(t_arr, psi_arr)
plt.show()
