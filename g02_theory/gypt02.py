import numpy as np
from scipy import special
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

MU0 = 4.0 * np.pi * 1e-7  # [T·m/A]
mu = 3.0            # dipole moment

# Magnit parameters
length = 0.10
diameter = 0.02

# Grid parameters
rho_max = 0.1     
Nrho = 1000        # step size in rho
zmax = 0.3         # grid size in z
Nz = 1000          # step size in z

# Coil parameters
z_center = 0.0      # coil's centered relative to the magnet
L_coil   = 0.1      # coil'S length
R_coil   = 0.015    # coil's radius
N_turns = 1000      # turn amount
n_density = N_turns / L_coil  # [turns/m]

# Create magent, grid, coil dicts
magnet = {
    "length": length,
    "diameter": diameter,
    "radius": diameter / 2,    # it was a
    "half_length": length / 2,  # it was b
    "filename": f"magnet_field_L{str(length*100)}cm_D{str(diameter*100)}cm_mu3.npz"
}

grid = {
    "rho_max": rho_max,
    "Nrho": Nrho,
    "zmax": zmax,
    "Nz": Nz
}

coil = {
    "z_center": z_center,
    "L_coil": L_coil,
    "R_coil": R_coil,
    "N_turns": N_turns,
    "n_density": n_density,
    "filename": f"coil_kernel.npz" # Do we need it for 4_1?
}

# ---------- 1) Robust Elliptic Integral (Crash-Proof) ----------
def cel(kc, p, c, s):
    """
    Generalized complete elliptic integral with singularity handling.
    """
    kc = np.asarray(kc, dtype=float)
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)

    # Detect singularities (kc=0 implies k=1)
    singular_mask = (kc == 0.0)                                     

    # Use dummy values for calculation to suppress SciPy warnings
    kc_safe = np.where(singular_mask, 1.0, kc)                      

    kc2 = kc_safe ** 2
    rf = special.elliprf(0.0, kc2, 1.0)
    rj = special.elliprj(0.0, kc2, 1.0, p)

    result = c * rf + (s - p * c) * (rj / 3.0)

    # Overwrite singularities with NaN (or 0) to allow grid computation to continue
    if np.any(singular_mask):
        result = np.asarray(result)  # ensure writable
        result[singular_mask] = np.nan
    return result

# ---------- 2) Physics Model (Unchanged) ----------
def B_cylindrical_permanent_magnet(rho, z, a, b, *, M=None, nI=None, mu=None, axis_tol=1e-12):
    """
    Calculate the magnetic field of a cylindrical permanent magnet.
    :param rho: Radial distance from the axis of the magnet [m]
    :param z: Axial distance from the center of the magnet [m]
    :param a: Radius of the magnet [m]
    :param b: Half-length of the magnet [m]
    :param M: Magnetisation [A/m]
    :param nI: Current density [A/m^2]
    """
    provided = [M is not None, nI is not None, mu is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly ONE of: M, nI, mu")
    if nI is None:
        nI = M if M is not None else mu / (2.0 * b * np.pi * a ** 2)

    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)

    B0 = (MU0 / np.pi) * nI
    Brho = np.zeros_like(rho)                           
    Bz = np.zeros_like(rho)                             

    on_axis = np.isclose(rho, 0.0, atol=axis_tol)    
    if np.any(on_axis):                                 
        zp = z[on_axis] + b
        zm = z[on_axis] - b
        Bz[on_axis] = 0.5 * MU0 * nI * (zp / np.sqrt(zp ** 2 + a ** 2) - zm / np.sqrt(zm ** 2 + a ** 2))

    off = ~on_axis                                         
    if np.any(off):
        rr = rho[off]
        zz = z[off]
        z_p, z_m = zz + b, zz - b
        d_p = np.sqrt(z_p ** 2 + (rr + a) ** 2)
        d_m = np.sqrt(z_m ** 2 + (rr + a) ** 2)

        # Calculate parameters (gamma, p, k, etc.)
        gamma = (a - rr) / (a + rr)
        p = gamma ** 2
        k_p = np.sqrt((z_p ** 2 + (a - rr) ** 2) / (z_p ** 2 + (a + rr) ** 2))
        k_m = np.sqrt((z_m ** 2 + (a - rr) ** 2) / (z_m ** 2 + (a + rr) ** 2))

        # Brho
        Brho[off] = B0 * ((a / d_p) * cel(k_p, 1, 1, -1) - (a / d_m) * cel(k_m, 1, 1, -1))

        # Bz
        term_p = (z_p / d_p) * cel(k_p, p, 1, gamma)
        term_m = (z_m / d_m) * cel(k_m, p, 1, gamma)
        Bz[off] = B0 * (a / (a + rr)) * (term_p - term_m)

    return Brho, Bz

# ---------- 3) Safer Grid Generation ----------
def precompute_field_table_uniform(filename, a, b, *, mu=None, M=None, nI=None,
                                   rho_max=0.10, Nrho=1000,
                                   zmax=0.30, Nz=1000):
    """
    :param filename: Tabel's file name
    :param a: что это
    :param b: что это
    :param mu: Dipole moment
    :param M: Magnetisation
    :param nI: Current density "K"
    :param rho_max: Borders of the grid in Rho
    :param Nrho: Step size in Rho
    :param zmax: Borders of the gris in Z
    :param Nz: Step size in Z
    :return: File with full B-Field
    """
    # Shift grid slightly to avoid hitting 0.0 or a/b exactly
    # This is a "Numerical Jitter" technique
    epsilon = 1e-9

    rho = np.linspace(0.0, rho_max, Nrho)                   
    # If a grid point is exactly at 'a', shift it slightly
    rho[np.isclose(rho, a)] += epsilon

    z = np.linspace(-zmax, zmax, Nz)
    z[np.isclose(np.abs(z), b)] += epsilon

    print(f"Generating uniform grid: {Nz}x{Nrho} ({Nz * Nrho / 1e6:.1f} M points)")

    RHO, Z = np.meshgrid(rho, z, indexing="xy")
    Br, Bz = B_cylindrical_permanent_magnet(RHO, Z, a, b, mu=mu, M=M, nI=nI)

    np.savez_compressed(filename, rho=rho, z=z, Br=Br, Bz=Bz,
                        a=a, b=b, mu=mu if mu is not None else np.nan)
    print(f"Saved to {filename}.")

# ---------- 4) Robust 3D Lookup ----------
class MagnetFieldTable:
    def __init__(self, npz_file, method="linear", bounds_error=False, fill_value=0.0):      
        # NOTE: bounds_error=False is much safer for simulations
        d = np.load(npz_file)
        self.rho = d["rho"]
        self.z = d["z"]
        self.Br = d["Br"]
        self.Bz = d["Bz"]

        # Grid bounds for validation (avoid silent extrapolation)
        self.rho_min = float(np.min(self.rho))
        self.rho_max = float(np.max(self.rho))
        self.z_min = float(np.min(self.z))
        self.z_max = float(np.max(self.z))

        self._Br = RegularGridInterpolator((self.z, self.rho), self.Br,
                                           method=method, bounds_error=bounds_error, fill_value=fill_value)
        self._Bz = RegularGridInterpolator((self.z, self.rho), self.Bz,
                                           method=method, bounds_error=bounds_error, fill_value=fill_value)

    def B_rz(self, rho, z):
        rho = np.asarray(rho, dtype=float)
        z = np.asarray(z, dtype=float)

        out_shape = np.broadcast(rho, z).shape          
        rho, z = np.broadcast_arrays(rho, z)

        pts = np.stack([z.ravel(), rho.ravel()], axis=1)  

        Br = self._Br(pts).reshape(out_shape)
        Bz = self._Bz(pts).reshape(out_shape)

        # If input was scalar, return scalars
        if out_shape == ():                                        
            return Br.item(), Bz.item()
        return Br, Bz

    def B_xyz(self, x, y, z):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        out_shape = np.broadcast(x, y, z).shape
        x, y, z = np.broadcast_arrays(x, y, z)

        rho = np.sqrt(x ** 2 + y ** 2)

        cos_phi = np.zeros_like(rho)
        sin_phi = np.zeros_like(rho)

        # Avoid division by zero at rho=0
        mask = rho > 1e-12
        cos_phi[mask] = x[mask] / rho[mask]
        sin_phi[mask] = y[mask] / rho[mask]

        Br, Bz = self.B_rz(rho, z)

        Bx = Br * cos_phi
        By = Br * sin_phi

        if out_shape == ():
            return float(Bx), float(By), float(Bz)
        return Bx, By, Bz

    def Bz_cyl(self, rho, z):
        """Convenience: return only axial component Bz(rho, z)."""
        return self.B_rz(rho, z)[1]

#----------- 5) 2D plot --------------------------------

def plot_streamlines_xz(tbl, xlim=0.10, zlim=0.30, nx=1000, nz=1000, density=2.0):
    x = np.linspace(-xlim, xlim, nx)
    z = np.linspace(-zlim, zlim, nz)
    X, Z = np.meshgrid(x, z, indexing="xy")

    RHO = np.abs(X)
    Br, Bz = tbl.B_rz(RHO, Z)
    Bx = Br * np.sign(X)  # convert Brho -> Bx in x-z plane

    Bmag = np.sqrt(Bx*Bx + Bz*Bz)

    plt.figure()
    plt.streamplot(x, z, Bx, Bz, density=density, linewidth=1)
    plt.contour(X, Z, Bmag, levels=12)  # magnitude contours (optional, no explicit colors)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Magnetic field lines in x–z plane")
    plt.tight_layout()
    plt.show()

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

def BrBz_loop_carlson(rho, z, a, I, axis_tol=1e-12):
    """
    Filamentary circular loop of radius a in plane z=0 carrying current I.
    Returns (Br, Bz) in cylindrical coordinates using Carlson RF/RD.

    Valid everywhere except the filament singularity (rho=a, z=0).
    """
    rho = np.asarray(rho, dtype=float)
    z   = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)

    Br = np.empty_like(rho, dtype=float)
    Bz = np.empty_like(rho, dtype=float)

    on_axis = np.isclose(rho, 0.0, atol=axis_tol, rtol=0.0)
    if np.any(on_axis):
        zz = z[on_axis]
        # On-axis loop field:
        Bz[on_axis] = MU0 * I * a*a / (2.0 * (a*a + zz*zz)**1.5)
        Br[on_axis] = 0.0

    off = ~on_axis
    if np.any(off):
        rr = rho[off]
        zz = z[off]

        Q = (a + rr)**2 + zz**2
        P = (a - rr)**2 + zz**2

        m   = (4.0 * a * rr) / Q           # m = k^2
        kc2 = P / Q                        # complementary modulus squared

        RF = special.elliprf(0.0, kc2, 1.0)   # = K(m)
        RD = special.elliprd(0.0, kc2, 1.0)
        K  = RF
        E  = RF - (m / 3.0) * RD

        sqrtQ = np.sqrt(Q)

        # Standard loop formulas in terms of K,E:
        # Bz = (μ0 I)/(2π √Q) [ K + (a^2 - ρ^2 - z^2)/P * E ]
        # Br = (μ0 I z)/(2π ρ √Q) [ -K + (a^2 + ρ^2 + z^2)/P * E ]
        with np.errstate(divide="ignore", invalid="ignore"):
            Bz_off = (MU0 * I) / (2.0 * np.pi * sqrtQ) * (
                K + ((a*a - rr*rr - zz*zz) / P) * E
            )
            Br_off = (MU0 * I * zz) / (2.0 * np.pi * rr * sqrtQ) * (
                -K + ((a*a + rr*rr + zz*zz) / P) * E
            )

        Br[off] = Br_off
        Bz[off] = Bz_off

    return Br, Bz


def precompute_coil_kernel_npz(
    out_npz: str,
    *,
    a: float,
    z_center: float,
    L_coil: float,
    n_density: float,
    rho_grid: np.ndarray,
    z_grid: np.ndarray,
    Nz_coil: int = 600,
    axis_tol: float = 1e-12,
    wire_eps: float = 0.0,     # optionally mask near filament singularity
):
    """
    Precompute coil field kernel (Br_unit, Bz_unit) for unit current (I=1 A):
        Br(rho,z) = I * Br_unit(rho,z)
        Bz(rho,z) = I * Bz_unit(rho,z)

    Coil model:
      - single radius a
      - finite length L_coil, centered at z_center
      - uniform turn density n_density [turns/m]
      - treated as continuous stack of loops (filamentary)

    Exports:
      rho, z, Br_unit, Bz_unit (shapes: (len(z), len(rho)))
      plus coil geometry metadata.
    """
    rho_grid = np.asarray(rho_grid, dtype=float)
    z_grid   = np.asarray(z_grid, dtype=float)

    # Mesh shaped (Nz, Nr) = (len(z_grid), len(rho_grid))
    RHO, Z = np.meshgrid(rho_grid, z_grid, indexing="xy")

    z1 = z_center - 0.5 * L_coil
    z2 = z_center + 0.5 * L_coil
    zprime = np.linspace(z1, z2, Nz_coil)

    # Streaming trapezoid integration along z' (memory-safe)
    Br_int = np.zeros_like(RHO, dtype=float)
    Bz_int = np.zeros_like(RHO, dtype=float)

    # First slice
    z_rel_prev = Z - zprime[0]
    Br_prev, Bz_prev = BrBz_loop_carlson(RHO, z_rel_prev, a, I=1.0, axis_tol=axis_tol)

    # Optional singularity masking near the filament:
    if wire_eps > 0.0:
        mask_prev = (np.abs(RHO - a) < wire_eps) & (np.abs(z_rel_prev) < wire_eps)
        Br_prev = np.where(mask_prev, np.nan, Br_prev)
        Bz_prev = np.where(mask_prev, np.nan, Bz_prev)

    for k in range(1, Nz_coil):
        z_rel = Z - zprime[k]
        Br_cur, Bz_cur = BrBz_loop_carlson(RHO, z_rel, a, I=1.0, axis_tol=axis_tol)

        if wire_eps > 0.0:
            mask = (np.abs(RHO - a) < wire_eps) & (np.abs(z_rel) < wire_eps)
            Br_cur = np.where(mask, np.nan, Br_cur)
            Bz_cur = np.where(mask, np.nan, Bz_cur)

        dz = zprime[k] - zprime[k - 1]

        # trapezoid segment
        Br_int += 0.5 * (Br_prev + Br_cur) * dz
        Bz_int += 0.5 * (Bz_prev + Bz_cur) * dz

        Br_prev, Bz_prev = Br_cur, Bz_cur

    # Multiply by turn density
    Br_unit = n_density * Br_int
    Bz_unit = n_density * Bz_int

    # If you used nan-masking: export as-is, or sanitize:
    # Br_unit = np.nan_to_num(Br_unit, nan=0.0, posinf=np.inf, neginf=-np.inf)
    # Bz_unit = np.nan_to_num(Bz_unit, nan=0.0, posinf=np.inf, neginf=-np.inf)

    np.savez_compressed(
        out_npz,
        rho=rho_grid,
        z=z_grid,
        Br_unit=Br_unit,
        Bz_unit=Bz_unit,
        # metadata
        a=float(a),
        z_center=float(z_center),
        L_coil=float(L_coil),
        n_density=float(n_density),
        Nz_coil=int(Nz_coil),
        axis_tol=float(axis_tol),
        wire_eps=float(wire_eps),
        model=np.array("finite-length single-radius coil; Carlson RF/RD; unit-current kernel", dtype=object),
    )

class CoilKernelField:
    """
    Axisymmetric coil field from a precomputed unit-current kernel:
      Br(rho,z) = I * Br_unit(rho,z)
      Bz(rho,z) = I * Bz_unit(rho,z)

    The kernel is stored as arrays shaped (len(z), len(rho)).
    """
    def __init__(self, kernel_npz, method="linear", bounds_error=True, fill_value=0.0):
        d = np.load(kernel_npz, allow_pickle=True)
        self.rho = d["rho"]
        self.z   = d["z"]
        self.Br_unit = d["Br_unit"]
        self.Bz_unit = d["Bz_unit"]

        # Geometry metadata (optional but handy)
        self.a         = float(d["a"])
        self.z_center  = float(d["z_center"])
        self.L_coil    = float(d["L_coil"])
        self.n_density = float(d["n_density"])

        # Interpolators expect points as (z, rho)
        self._Br = RegularGridInterpolator(
            (self.z, self.rho), self.Br_unit,
            method=method, bounds_error=bounds_error, fill_value=fill_value
        )
        self._Bz = RegularGridInterpolator(
            (self.z, self.rho), self.Bz_unit,
            method=method, bounds_error=bounds_error, fill_value=fill_value
        )

    def B_rz(self, rho, z, I):
        """
        Return (Br,Bz) at (rho,z) for current I (scalar or array).
        Supports broadcasting over rho/z/I.
        """
        rho = np.asarray(rho, dtype=float)
        z   = np.asarray(z, dtype=float)
        I   = np.asarray(I, dtype=float)

        out_shape = np.broadcast(rho, z, I).shape
        rho, z, I = np.broadcast_arrays(rho, z, I)

        pts = np.stack([z.ravel(), rho.ravel()], axis=1)
        Br = self._Br(pts).reshape(out_shape) * I
        Bz = self._Bz(pts).reshape(out_shape) * I

        if out_shape == ():
            return float(Br), float(Bz)
        return Br, Bz

    def Bz(self, rho, z, I):
        return self.B_rz(rho, z, I)[1]
