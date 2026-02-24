import numpy as np
import os
from scipy import special
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. MATHEMATICAL HELPERS
# ==========================================
MU0 = 4.0 * np.pi * 1e-7


def cel(kc, p, c, s):
    """Bulirsch generalized elliptic integral."""
    kc = np.asarray(kc, dtype=float)
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
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


# ==========================================
# 2. MAGNET PHYSICS
# ==========================================
def B_cylindrical_permanent_magnet(rho, z, a, b, *, mu=None):
    """Field of a cylindrical magnet using elliptic integrals."""
    nI = mu / (2.0 * b * np.pi * a ** 2)
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)

    B0 = (MU0 / np.pi) * nI
    Brho = np.zeros_like(rho)
    Bz = np.zeros_like(rho)

    on_axis = np.isclose(rho, 0.0, atol=1e-12)
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

        gamma = (a - rr) / (a + rr)
        p = gamma ** 2
        k_p = np.sqrt((z_p ** 2 + (a - rr) ** 2) / (z_p ** 2 + (a + rr) ** 2))
        k_m = np.sqrt((z_m ** 2 + (a - rr) ** 2) / (z_m ** 2 + (a + rr) ** 2))

        Brho[off] = B0 * ((a / d_p) * cel(k_p, 1, 1, -1) - (a / d_m) * cel(k_m, 1, 1, -1))
        term_p = (z_p / d_p) * cel(k_p, p, 1, gamma)
        term_m = (z_m / d_m) * cel(k_m, p, 1, gamma)
        Bz[off] = B0 * (term_p - term_m)

    return Brho, Bz


class MagnetFieldTable:
    def __init__(self, npz_file):
        d = np.load(npz_file)
        self.rho = d["rho"]
        self.z = d["z"]
        self.Br = d["Br"]
        self.Bz = d["Bz"]
        self._Br = RegularGridInterpolator((self.z, self.rho), self.Br, bounds_error=False, fill_value=0.0)
        self._Bz = RegularGridInterpolator((self.z, self.rho), self.Bz, bounds_error=False, fill_value=0.0)

    def B_rz(self, rho, z):
        rho, z = np.broadcast_arrays(rho, z)
        pts = np.stack([z.ravel(), rho.ravel()], axis=-1)
        Br = self._Br(pts).reshape(rho.shape)
        Bz = self._Bz(pts).reshape(rho.shape)
        return Br, Bz


def precompute_field_table(filename, a, b, mu, rho_max=0.05, z_max=0.30, Nr=200, Nz=600):
    print(f"Generating new magnet table: {filename}...")
    rho = np.linspace(0.0, rho_max, Nr)
    z = np.linspace(-z_max, z_max, Nz)
    RHO, Z = np.meshgrid(rho, z, indexing="xy")
    Br, Bz = B_cylindrical_permanent_magnet(RHO, Z, a, b, mu=mu)
    np.savez(filename, rho=rho, z=z, Br=Br, Bz=Bz)


# ==========================================
# 3. FLUX LINKAGE
# ==========================================
def get_coil_flux_linkage(tbl, z_center, L_coil, R_coil, N_turns, z_mag, Nr=100, Nz=100):

    z1, z2 = z_center - 0.5 * L_coil, z_center + 0.5 * L_coil
    r = np.linspace(0.0, R_coil, Nr)
    Z = np.linspace(z1, z2, Nz)
    R_grid = r[np.newaxis, :]
    Z_grid = (Z[:, np.newaxis] - z_mag)
    _, Bz = tbl.B_rz(R_grid, Z_grid)
    integrand = Bz * (2.0 * np.pi * R_grid)
    phi_per_turn = trapezoid(integrand, r, axis=1)
    total_flux_int = trapezoid(phi_per_turn, Z)
    return (N_turns / L_coil) * total_flux_int


class FluxLinkageLookup:
    def __init__(self, tbl, z_center, L_coil, R_coil, N_turns, z_range=(-0.3, 0.3), Nz_mag=600):
        self.z_center = z_center
        self.z_rel_grid = np.linspace(z_range[0], z_range[1], Nz_mag)
        psi_vals = []

        # Added tqdm progress bar for precomputation
        for z_rel in tqdm(self.z_rel_grid, desc="Building Flux Lookup Table", unit="pt"):
            z_abs = z_center + z_rel
            psi = get_coil_flux_linkage(tbl, z_center, L_coil, R_coil, N_turns, z_mag=z_abs)
            psi_vals.append(psi)

        self.psi_grid = np.array(psi_vals)
        self.dpsi_grid = np.gradient(self.psi_grid, self.z_rel_grid)
        self.psi_interp = interp1d(self.z_rel_grid, self.psi_grid, kind='cubic', bounds_error=False, fill_value=0.0)
        self.dpsi_interp = interp1d(self.z_rel_grid, self.dpsi_grid, kind='cubic', bounds_error=False, fill_value=0.0)

    def get_data(self, z_rel):
        return self.psi_interp(z_rel), self.dpsi_interp(z_rel)


# ==========================================
# 4. SIMULATION LOOP (MODIFIED: NO INDUCTANCE)
# ==========================================
def solve_coupled_dynamics(t_span, y0, flux_lookup, mech, elec):
    m, k, c = mech['m'], mech['k'], mech['c']
    R_res = elec['R']  # Removed L_ind
    z_top, z_bot, g = mech['z_top'], mech['z_bot'], mech['g']
    z_eq = 0.5 * (z_top + z_bot)
            # - (m * g) / (2 * k))

    # Setup progress bar for ODE solver
    pbar = tqdm(total=100, desc="Solving ODE Dynamics", unit="%")
    last_p = [0]

    def system_ode(t, y):
        # Update progress bar based on simulation time t
        progress = int(100 * t / t_span[1])
        if progress > last_p[0]:
            pbar.update(progress - last_p[0])
            last_p[0] = progress

        # State vector y now only has 2 elements: [Position, Velocity]
        z_disp, v = y

        z_rel = (z_eq + z_disp) - flux_lookup.z_center
        _, dpsi_dz = flux_lookup.get_data(z_rel)

        # 1. Calculate EMF (Faraday's Law)
        emf = -dpsi_dz * v

        # 2. Calculate Current (Ohm's Law, neglecting Inductance L*di/dt)
        i = emf / R_res

        # 3. Calculate Forces
        F_mag = i * dpsi_dz  # Lorentz Force
        F_spring = -2.0 * k * z_disp
        F_damp = -c * v

        # 4. Acceleration (Newton's 2nd Law)
        a = (F_spring + F_damp + F_mag) / m

        return [v, a]

    sol = solve_ivp(system_ode, t_span, y0, method='RK45', rtol=1e-6, atol=1e-8)
    pbar.close()
    return sol, z_eq


if __name__ == "__main__":
    magnet_conf = {
        'length': 0.05,
        'diameter': 0.02,
        'mu': 7.5,
        'fname': 'magnet_field_v3.npz'
    }
    coil_conf = {
        'z_center': 0.01,
        'L_coil': 0.1,
        'R_coil': 0.0182,
        'N_turns': 1167
    }
    mech_conf = {
        'm': 5 * 0.0237,
        'k': 5.15,
        'c': 0.00690676,
        'g': 9.81,
        'z_top': 0.2,
        'z_bot': -0.2
    }
    elec_conf = {
        'R': 47,
        # 'L' is removed completely
    }

    if not os.path.exists(magnet_conf['fname']):
        precompute_field_table(magnet_conf['fname'], a=magnet_conf['diameter'] / 2, b=magnet_conf['length'] / 2,
                               mu=magnet_conf['mu'])

    tbl = MagnetFieldTable(magnet_conf['fname'])
    lookup = FluxLinkageLookup(tbl, **coil_conf)

    # Initial Conditions: [Position, Velocity] (Current is removed)
    y0 = [0.05, 0.0]

    sol, z_eq = solve_coupled_dynamics((0, 30.0), y0, lookup, mech_conf, elec_conf)

    # Post-processing to recover Current and Force for plotting
    t = sol.t
    z_disp, v = sol.y

    # Re-calculate electrical values algebraically
    i = []
    f_mag = []

    for val_z, val_v in zip(z_disp, v):
        # Get coupling factor dPhi/dz at this position
        z_rel = (z_eq + val_z) - coil_conf['z_center']
        _, dpsi_dz = lookup.get_data(z_rel)

        # Calculate instantaneous current (I = EMF / R)
        emf = -dpsi_dz * val_v
        val_i = emf / elec_conf['R']

        # Calculate force
        val_f = val_i * dpsi_dz

        i.append(val_i)
        f_mag.append(val_f)

    i = np.array(i)
    f_mag = np.array(f_mag)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax1.plot(t, z_disp * 100)
    ax1.set_ylabel("Pos [cm]")
    ax1.set_title("Magnet Motion (No Inductance)")
    ax1.grid(True)

    ax2.plot(t, i, 'r')
    ax2.set_ylabel("Current [A]")
    ax2.grid(True)

    ax3.plot(t, f_mag, 'g')
    ax3.set_ylabel("Force [N]")
    ax3.set_xlabel("Time [s]")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
