import numpy as np
import os
from scipy import special, integrate
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, quad, trapezoid
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTS & MATHEMATICAL HELPERS
# ==========================================

MU0 = 4.0 * np.pi * 1e-7  # [T·m/A]

def cel(kc, p, c, s):
    """
    Generalized complete elliptic integral with singularity handling (Bulirsch).
    """
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
# 2. PERMANENT MAGNET PHYSICS
# ==========================================

def B_cylindrical_permanent_magnet(rho, z, a, b, *, M=None, nI=None, mu=None, axis_tol=1e-12):
    """
    Calculate field of a cylindrical magnet (radius a, half-length b).
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

        gamma = (a - rr) / (a + rr)
        p = gamma ** 2
        k_p = np.sqrt((z_p ** 2 + (a - rr) ** 2) / (z_p ** 2 + (a + rr) ** 2))
        k_m = np.sqrt((z_m ** 2 + (a - rr) ** 2) / (z_m ** 2 + (a + rr) ** 2))

        Brho[off] = B0 * ((a / d_p) * cel(k_p, 1, 1, -1) - (a / d_m) * cel(k_m, 1, 1, -1))

        term_p = (z_p / d_p) * cel(k_p, p, 1, gamma)
        term_m = (z_m / d_m) * cel(k_m, p, 1, gamma)
        Bz[off] = B0 * (a / (a + rr)) * (term_p - term_m)

    return Brho, Bz

def precompute_field_table_uniform(filename, a, b, *, mu=None, M=None, nI=None,
                                   rho_max=0.10, Nrho=500,
                                   zmax=0.30, Nz=500):
    print(f"Generating Magnet Field Table: {filename} ...")
    epsilon = 1e-9
    rho = np.linspace(0.0, rho_max, Nrho)
    rho[np.isclose(rho, a)] += epsilon
    z = np.linspace(-zmax, zmax, Nz)
    z[np.isclose(np.abs(z), b)] += epsilon

    RHO, Z = np.meshgrid(rho, z, indexing="xy")
    Br, Bz = B_cylindrical_permanent_magnet(RHO, Z, a, b, mu=mu, M=M, nI=nI)

    np.savez_compressed(filename, rho=rho, z=z, Br=Br, Bz=Bz, a=a, b=b, mu=mu)
    print(f"Saved {filename}.")

class MagnetFieldTable:
    def __init__(self, npz_file, method="linear", bounds_error=False, fill_value=0.0):
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"File {npz_file} not found. Please generate it first.")
        d = np.load(npz_file)
        self.rho = d["rho"]
        self.z = d["z"]
        self.Br = d["Br"]
        self.Bz = d["Bz"]
        
        # Metadata
        self.a = float(d["a"]) if "a" in d else 0.0
        self.b = float(d["b"]) if "b" in d else 0.0

        self.rho_max = float(np.max(self.rho))
        self.z_min = float(np.min(self.z))
        self.z_max = float(np.max(self.z))

        self._Br = RegularGridInterpolator((self.z, self.rho), self.Br,
                                           method=method, bounds_error=bounds_error, fill_value=fill_value)
        self._Bz = RegularGridInterpolator((self.z, self.rho), self.Bz,
                                           method=method, bounds_error=bounds_error, fill_value=fill_value)

    def Bz_cyl(self, rho, z):
        """Return axial component Bz(rho, z)."""
        # Quick check for scalar vs array to avoid overhead
        rho = np.asarray(rho)
        z = np.asarray(z)
        pts = np.stack([z.ravel(), rho.ravel()], axis=1)
        return self._Bz(pts).reshape(rho.shape)

# ==========================================
# 3. COIL PHYSICS (FLUX & INDUCTANCE)
# ==========================================

def get_coil_flux_linkage(tbl: MagnetFieldTable,
                         z_center: float, L_coil: float, R_coil: float,
                         n_density: float,
                         z_mag: float = 0.0,
                         Nr: int = 50, Nz: int = 50):
    """
    Calculate Flux Linkage (Psi) by integrating Bz over the coil volume.
    Magnet is at lab position z_mag. Coil is fixed at z_center.
    """
    z_start = z_center - L_coil / 2.0
    z_end = z_center + L_coil / 2.0
    
    # Grid for integration
    r_vals = np.linspace(0.0, R_coil, Nr)
    z_vals = np.linspace(z_start, z_end, Nz)
    R, Z = np.meshgrid(r_vals, z_vals)

    # Field in magnet frame: Bz(r, z_lab - z_mag)
    # This maps the coil points (Z) back to the magnet's rest frame
    Bz_val = tbl.Bz_cyl(R, Z - z_mag) 
    
    integrand = Bz_val * 2.0 * np.pi * R

    # Double integration: first over r, then over z
    # Using scipy.integrate.trapezoid or np.trapz (older numpy)
    integral_r = trapezoid(integrand, r_vals, axis=1)
    flux_total = trapezoid(integral_r, z_vals, axis=0)
    
    return n_density * flux_total

class FluxLinkageLookup:
    """
    Precomputes Psi(z) and dPsi/dz(z) for fast lookup during simulation.
    """
    def __init__(self, tbl, z_center, L_coil, R_coil, n_density,
                 z_range, Nz_mag=500):
        
        self.z_grid = np.linspace(z_range[0], z_range[1], Nz_mag)
        psi_list = []
        
        # Precompute flux at each magnet position
        for z_m in self.z_grid:
            psi = get_coil_flux_linkage(tbl, z_center, L_coil, R_coil, n_density, z_mag=z_m)
            psi_list.append(psi)
            
        self.psi_grid = np.array(psi_list)
        
        # Compute gradient (Coupling Factor)
        self.dpsi_dz_grid = np.gradient(self.psi_grid, self.z_grid)
        
        # Create Interpolators
        self.psi_interp = interp1d(self.z_grid, self.psi_grid, kind='cubic', fill_value="extrapolate")
        self.dpsi_interp = interp1d(self.z_grid, self.dpsi_dz_grid, kind='cubic', fill_value="extrapolate")
        
    def get_data(self, z):
        """Returns (Psi, dPsi/dz) at position z."""
        return self.psi_interp(z), self.dpsi_interp(z)

# ==========================================
# 4. COUPLED DYNAMICS SOLVER
# ==========================================

def solve_coupled_dynamics(t_span, y0, flux_lookup, mech_params, elec_params):
    """
    Solves coupled ODEs:
      1. m*a = -k*z - c*v + F_mag + F_ext
      2. L*di/dt + R*i = EMF
    Using Energy Method: F_mag = i * dPsi/dz
    """
    m, k, c = mech_params['m'], mech_params['k'], mech_params['c']
    z_eq = mech_params.get('z_eq', 0.0)
    R_res, L_ind = elec_params['R'], elec_params['L']
    
    def system_ode(t, y):
        z, v, i = y
        
        # 1. Lookup Electromagnetic Coupling
        # psi, dpsi_dz = flux_lookup.get_data(z) # psi not needed for force
        _, dpsi_dz = flux_lookup.get_data(z)
        
        # 2. Electrical Equation (Kirchhoff)
        # EMF = - dPsi/dt = - (dPsi/dz) * v
        emf = -dpsi_dz * v
        di_dt = (emf - R_res * i) / L_ind
        
        # 3. Mechanical Equation (Newton)
        # Force from Magnet on Coil = - Force from Coil on Magnet
        # F_mag (on magnet) = i * dPsi/dz (Lenz Law direction handled by sign of dPsi/dz)
        F_mag = i * dpsi_dz
        F_spring = -k * (z - z_eq)
        F_damp = -c * v
        
        a = (F_spring + F_damp + F_mag) / m
        
        return [v, a, di_dt]

    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    
    sol = solve_ivp(system_ode, t_span, y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-6, atol=1e-8)
    
    return sol

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # --- A. Setup Parameters ---
    
    # Magnet Geometry (Neodymium N42 approx)
    magnet_params = {
        'length': 0.05,   # 10 cm
        'diameter': 0.02, # 2 cm
        'mu': 15.0,        # Dipole moment approx
        'filename': 'magnet_field.npz'
    }
    
    # Coil Geometry
    coil_params = {
        'z_center': 0.0,
        'L_coil': 0.10,    # 5 cm length
        'diameter_coil': 0.03, # 3 cm diameter
        'R_coil': 0.015,   # 1.5 cm radius
        'N_turns': 1160    # 1000 turns
    }
    coil_params['n_density'] = coil_params['N_turns'] / coil_params['L_coil']

    # Simulation Parameters
    mech_conf = {'m': 0.2, 'k': 5.0, 'c': 0.005, 'z_eq': 0.0} # m=0.2kg, k=50N/m
    elec_conf = {'R': 2.0, 'L': 0.01} # 5 Ohms, 10mH
    
    # --- B. Generate Field Table (if needed) ---
    if not os.path.exists(magnet_params['filename']):
        precompute_field_table_uniform(
            magnet_params['filename'], 
            a=magnet_params['diameter']/2, 
            b=magnet_params['length']/2, 
            mu=magnet_params['mu']
        )
    
    # Load Table
    print("Loading Field Table...")
    tbl = MagnetFieldTable(magnet_params['filename'])
    
    # --- C. Precompute Flux Lookup ---
    print("Building Flux Lookup (Fast)...")
    # Scan range slightly larger than expected motion
    lookup = FluxLinkageLookup(
        tbl, 
        coil_params['z_center'], 
        coil_params['L_coil'], 
        coil_params['R_coil'], 
        coil_params['n_density'],
        z_range=(-0.2, 0.2) # -20cm to +20cm
    )
    
    # --- D. Run Simulation ---
    print("Running Coupled Dynamics...")
    # Initial: Magnet dropped from z=15cm, v=0, i=0
    y0 = [0.05, 0.0, 0.0] 
    t_span = (0, 60.0) 
    
    sol = solve_coupled_dynamics(t_span, y0, lookup, mech_conf, elec_conf)
    
    # --- E. Plotting ---
    t = sol.t
    z = sol.y[0]
    v = sol.y[1]
    i = sol.y[2]
    
    # Calculate derived quantities for plotting
    _, dPsi_dz = lookup.get_data(z)
    force_mag = i * dPsi_dz
    emf = -dPsi_dz * v
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(t, z * 100, 'b', label='Position')
    axs[0].set_ylabel('Position [cm]')
    axs[0].set_title('Magnet Position')
    axs[0].grid(True)
    
    axs[1].plot(t, i, 'r', label='Current')
    axs[1].set_ylabel('Current [A]')
    axs[1].set_title('Induced Current')
    axs[1].grid(True)
    
    axs[2].plot(t, force_mag, 'g', label='Magnetic Force')
    axs[2].set_ylabel('Force [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_title('Magnetic Damping Force')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    print("Done.")
