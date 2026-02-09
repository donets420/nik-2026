import numpy as np
from scipy import special
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

MU0 = 4.0 * np.pi * 1e-7  # [T·m/A]

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
    singular_mask = (kc == 0.0)                                     #  почему

    # Use dummy values for calculation to suppress SciPy warnings
    kc_safe = np.where(singular_mask, 1.0, kc)                      #  почему

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
    provided = [M is not None, nI is not None, mu is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly ONE of: M, nI, mu")
    if nI is None:
        nI = M if M is not None else mu / (2.0 * b * np.pi * a ** 2)

    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)

    B0 = (MU0 / np.pi) * nI
    Brho = np.zeros_like(rho)                           # что это
    Bz = np.zeros_like(rho)                             # что это

    on_axis = np.isclose(rho, 0.0, atol=axis_tol)    # что это
    if np.any(on_axis):                                 # что это
        zp = z[on_axis] + b
        zm = z[on_axis] - b
        Bz[on_axis] = 0.5 * MU0 * nI * (zp / np.sqrt(zp ** 2 + a ** 2) - zm / np.sqrt(zm ** 2 + a ** 2))

    off = ~on_axis                                         # что это
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

    rho = np.linspace(0.0, rho_max, Nrho)                   # разве не от -rho_max rho_max?
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
    def __init__(self, npz_file, method="linear", bounds_error=False, fill_value=0.0):      # что это
        # NOTE: bounds_error=False is much safer for simulations
        d = np.load(npz_file)
        self.rho = d["rho"]
        self.z = d["z"]
        self.Br = d["Br"]
        self.Bz = d["Bz"]

        self._Br = RegularGridInterpolator((self.z, self.rho), self.Br,
                                           method=method, bounds_error=bounds_error, fill_value=fill_value)
        self._Bz = RegularGridInterpolator((self.z, self.rho), self.Bz,
                                           method=method, bounds_error=bounds_error, fill_value=fill_value)

    def B_rz(self, rho, z):
        rho = np.asarray(rho, dtype=float)
        z = np.asarray(z, dtype=float)

        out_shape = np.broadcast(rho, z).shape          # что это
        rho, z = np.broadcast_arrays(rho, z)

        pts = np.stack([z.ravel(), rho.ravel()], axis=1)  # что это

        Br = self._Br(pts).reshape(out_shape)
        Bz = self._Bz(pts).reshape(out_shape)

        # If input was scalar, return scalars
        if out_shape == ():                                        # что это
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

# ---------- 6) Example usage for your magnet ----------

if __name__ == "__main__":
    # Magnet parameters
    length = 0.10       # length
    diameter = 0.02     # diameter
    a = diameter / 2.0  # (radius)
    b = length / 2.0    # (half-length)

    mu = 3.0            # dipole moment

    # Grid parameters
    rho_max = 0.1       # grid size in rho
    Nrho = 1000         # step size in rho

    zmax = 0.3          # grid size in z
    Nz = 1000           # step size in z

    # Precompute field table over your region (rho<=10 cm, |z|<=30 cm)
    filename = "magnet_field_L10cm_D2cm_mu3.npz"
    precompute_field_table_uniform(
        filename,
        a=a, b=b, mu=mu,
        rho_max=rho_max, Nrho=Nrho,
        zmax=zmax, Nz=Nz
    )

    # Load the file and plot
    tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

    plot_streamlines_xz(tbl)

