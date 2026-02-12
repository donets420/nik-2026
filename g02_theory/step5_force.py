import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scipy.integrate import quad

MU0 = 4e-7 * np.pi

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


def calculate_magnet_force(M, R_mag, L_mag, z_defection, current_coil):
    """
    M: Magnetization (A/m)
    R_mag: Radius of the magnet (m)
    L_mag: Length of the magnet (m)
    z_defection: Position of the magnet at the time t (m)
    current_coil: Current passed to your coil function
    """

    z_start = z_defection - L_mag / 2
    z_end = z_defection + L_mag / 2

    # Pre-scale constant to avoid repeated multiplication
    force_const = -2 * np.pi * R_mag * M * current_coil

    # Access the interpolator directly to skip the B_rz overhead
    def integrand(z):
        # pts must be [[z, rho]]
        return force_const * coil._Br([[z, R_mag]])[0]

    force_z, _ = quad(integrand, z_start, z_end)
    return force_z

filename_kernel = "coil_kernel.npz"
coil = CoilKernelField(filename_kernel, method="linear", bounds_error=True)

