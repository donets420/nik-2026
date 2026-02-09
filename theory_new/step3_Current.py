import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from step1_1_M_FIeld_Precomputation import MagnetFieldTable
from step2_Flux_Linkage import get_coil_flux_linkage, flux_linkage_vs_time

class FluxLinkageLookup:
    """
    Precompute Psi_ext(z_mag) on a 1D grid and build smooth interpolators
    for Psi(z) and dPsi/dz(z).

    This is much faster than doing the 2D (r,z) integration at every time step,
    and usually produces cleaner EMF/current.
    """
    def __init__(self, tbl, z_center: float, L_coil: float, R_coil: float, n_density: float,
                 z_mag_min: float, z_mag_max: float, Nz_mag: int = 401, Nr: int = 80, Nz: int = 80):
        self.z_grid = np.linspace(z_mag_min, z_mag_max, Nz_mag)

        psi_grid = []
        for z_mag in self.z_grid:
            psi = get_coil_flux_linkage(
                tbl, z_center, L_coil, R_coil, n_density,
                z_mag=z_mag, z_mag_range=(z_mag_min, z_mag_max),
                Nr=Nr, Nz=Nz
            )
            psi_grid.append(psi)
        self.psi_grid = np.asarray(psi_grid)

        # Derivative dPsi/dz on the same grid
        self.dpsi_dz_grid = np.gradient(self.psi_grid, self.z_grid)

        # Interpolators (linear is fine; cubic can overshoot near edges)
        self.psi_of_z = interp1d(self.z_grid, self.psi_grid,
                                 kind="linear", bounds_error=True)
        self.dpsi_dz_of_z = interp1d(self.z_grid, self.dpsi_dz_grid,
                                     kind="linear", bounds_error=True)

    def psi(self, z_mag):
        return self.psi_of_z(z_mag)

    def dpsi_dz(self, z_mag):
        return self.dpsi_dz_of_z(z_mag)

def induced_current_vs_time(tbl: MagnetFieldTable,
                            z_center: float, L_coil: float, R_coil: float,
                            n_density: float,
                            t_vals, z_mag_func,
                            R_total: float, L_self: float,
                            i0: float = 0.0,
                            # lookup controls
                            use_lookup: bool = True,
                            z_mag_range=None,
                            Nz_mag: int = 401,
                            Nr_flux: int = 80, Nz_flux: int = 80):
    """
    Compute Psi_ext(t), EMF(t) and induced current i(t) in an RL circuit:
        L_self di/dt + R_total i = - dPsi_ext/dt

    Returns:
        t, z, v, psi, emf, i
    """
    t = np.asarray(t_vals, dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_vals must be strictly increasing.")

    # Magnet trajectory
    z = np.array([float(z_mag_func(tt)) for tt in t])

    # Range for validation / lookup
    if z_mag_range is None:
        z_min, z_max = float(np.min(z)), float(np.max(z))
        # small margin helps avoid edge issues when differentiating
        margin = 0.02 * (z_max - z_min + 1e-12)
        z_mag_range = (z_min - margin, z_max + margin)

    z_min, z_max = z_mag_range

    # Velocity v(t) from z(t) (central differences)
    v = np.gradient(z, t)

    if use_lookup:
        lookup = FluxLinkageLookup(
            tbl, z_center, L_coil, R_coil, n_density,
            z_min, z_max,
            Nz_mag=Nz_mag, Nr=Nr_flux, Nz=Nz_flux
        )
        psi = lookup.psi(z)
        dpsi_dz = lookup.dpsi_dz(z)
        emf = -(dpsi_dz * v)  # V
    else:
        # Slower: compute Psi(t) directly and differentiate in time
        _, psi = flux_linkage_vs_time(
            tbl, z_center, L_coil, R_coil, n_density,
            t, z_mag_func, z_mag_range=z_mag_range,
            Nr=Nr_flux, Nz=Nz_flux
        )
        emf = -np.gradient(psi, t)

    if L_self <= 0:
        # Quasi-static limit (no inductance): i = emf / R
        if R_total <= 0:
            raise ValueError("Need R_total > 0 if L_self <= 0.")
        i = emf / R_total
        return t, z, v, psi, emf, i

    if R_total <= 0:
        raise ValueError("R_total must be > 0 for a physical RL circuit.")

    # Build continuous EMF(t) for ODE solver
    emf_of_t = interp1d(t, emf, kind="linear", bounds_error=False, fill_value=(emf[0], emf[-1]))

    def rhs(tt, ii):
        # L di/dt + R i = emf(t)  ->  di/dt = (emf(t) - R i)/L
        return (float(emf_of_t(tt)) - R_total * ii) / L_self

    sol = solve_ivp(rhs, (t[0], t[-1]), y0=[i0], t_eval=t, method="RK45")
    if not sol.success:
        raise RuntimeError("ODE solver failed: " + str(sol.message))
    i = sol.y[0]

    return t, i

# -------------------- Example usage --------------------

filename = "magnet_field_L10cm_D2cm_mu3.npz"
tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

# Coil geometry
z_center = 0.2      # coil's centered relative to the magnet
L_coil   = 0.1      # coil'S length
R_coil   = 0.015    # coil's radius
N_turns = 1000      # turn amount
n_density = N_turns / L_coil  # [turns/m]

# Circuit
R_total = 5.0      # Ohm (coil + load)
L_self  = 20e-3    # 20 mH

# Magnet motion: sinusoidal
A = 0.03            # 3 cm amplitude
f = 2.0
w = 2*np.pi*f
z0 = 0.0
z_mag_func = lambda t: z0 + A*np.cos(w*t)

t_vals = np.linspace(0, 2.0, 2000)

t, i = induced_current_vs_time(
    tbl,
    z_center, L_coil, R_coil, n_density,
    t_vals, z_mag_func,
    R_total=R_total, L_self=L_self,
    i0=0.0,
    use_lookup=True,
    Nr_flux=80, Nz_flux=80
)


plt.plot(t, i)
plt.xlabel("t [s]")
plt.ylabel("Current [A]")
plt.grid(True)

plt.show()


