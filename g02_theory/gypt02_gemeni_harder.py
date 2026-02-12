import numpy as np
import os
from scipy import special, integrate
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, quad, trapezoid
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTS & MATH HELPERS
# ==========================================

MU0 = 4.0 * np.pi * 1e-7  # [T·m/A]

def cel(kc, p, c, s):
    """Generalized complete elliptic integral (Bulirsch)."""
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
# 2. MAGNET MODEL (For Flux/EMF)
# ==========================================

def B_cylindrical_permanent_magnet(rho, z, a, b, *, M=None, nI=None, mu=None, axis_tol=1e-12):
    """Standard cylindrical magnet field calculation."""
    provided = [M is not None, nI is not None, mu is not None]
    if sum(provided) != 1: raise ValueError("Provide exactly ONE of: M, nI, mu")
    if nI is None: nI = M if M is not None else mu / (2.0 * b * np.pi * a ** 2)

    rho, z = np.broadcast_arrays(np.asarray(rho, float), np.asarray(z, float))
    B0 = (MU0 / np.pi) * nI
    Brho = np.zeros_like(rho)
    Bz = np.zeros_like(rho)

    on_axis = np.abs(rho) < axis_tol
    if np.any(on_axis):
        zp, zm = z[on_axis] + b, z[on_axis] - b
        Bz[on_axis] = 0.5 * MU0 * nI * (zp / np.sqrt(zp**2 + a**2) - zm / np.sqrt(zm**2 + a**2))

    off = ~on_axis
    if np.any(off):
        rr, zz = rho[off], z[off]
        z_p, z_m = zz + b, zz - b
        d_p, d_m = np.sqrt(z_p**2 + (rr + a)**2), np.sqrt(z_m**2 + (rr + a)**2)
        gamma = (a - rr) / (a + rr)
        k_p = np.sqrt((z_p**2 + (a - rr)**2) / (z_p**2 + (a + rr)**2))
        k_m = np.sqrt((z_m**2 + (a - rr)**2) / (z_m**2 + (a + rr)**2))
        
        # Elliptic Integrals (Bulirsch)
        Brho[off] = B0 * ((a / d_p) * cel(k_p, 1, 1, -1) - (a / d_m) * cel(k_m, 1, 1, -1))
        term_p = (z_p / d_p) * cel(k_p, gamma**2, 1, gamma)
        term_m = (z_m / d_m) * cel(k_m, gamma**2, 1, gamma)
        Bz[off] = B0 * (a / (a + rr)) * (term_p - term_m)
    return Brho, Bz

def precompute_magnet_table(filename, a, b, M, N=500):
    """Generate field table for the Magnet."""
    print(f"Generating Magnet Table: {filename}...")
    rho = np.linspace(0.0, 0.1, N)
    z = np.linspace(-0.3, 0.3, N)
    R, Z = np.meshgrid(rho, z, indexing='xy')
    Br, Bz = B_cylindrical_permanent_magnet(R, Z, a, b, M=M)
    np.savez_compressed(filename, rho=rho, z=z, Br=Br, Bz=Bz, a=a, b=b, M=M)

class MagnetFieldTable:
    """Interpolates B_magnet for Flux calculation."""
    def __init__(self, filename):
        d = np.load(filename)
        self.Bz = RegularGridInterpolator((d['z'], d['rho']), d['Bz'], bounds_error=False, fill_value=0.0)
    
    def get_Bz(self, rho, z):
        pts = np.stack([np.asarray(z).ravel(), np.asarray(rho).ravel()], axis=1)
        return self.Bz(pts).reshape(np.asarray(rho).shape)

# ==========================================
# 3. COIL MODEL (For Force Calculation)
# ==========================================

def BrBz_loop_carlson(rho, z, a, I=1.0, axis_tol=1e-12):
    """
    Filamentary loop field (Carlson method).
    FIXED: Handles rho=0 explicitly to avoid divide-by-zero.
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)
    
    Br = np.zeros_like(rho)
    Bz = np.zeros_like(rho)
    
    # 1. Handle On-Axis (rho ~ 0) - Avoids singularity
    on_axis = np.abs(rho) < axis_tol
    if np.any(on_axis):
        zz = z[on_axis]
        # Standard on-axis formula for a loop
        Bz[on_axis] = (MU0 * I * a**2) / (2.0 * (a**2 + zz**2)**1.5)
        Br[on_axis] = 0.0

    # 2. Handle Off-Axis
    off_axis = ~on_axis
    if np.any(off_axis):
        rr = rho[off_axis]
        zz = z[off_axis]
        
        Q = (a + rr)**2 + zz**2
        P = (a - rr)**2 + zz**2
        
        # Mask out wire singularity (P=0) if it occurs
        # If P is exactly 0, we are on the wire. 
        # We clamp P to a tiny value to prevent NaN, though physically it's singular.
        P = np.maximum(P, 1e-15)

        m = 4 * a * rr / Q
        # Carlson conversion: K(m) = RF(0, 1-m, 1)
        kc2 = 1.0 - m
        
        RF = special.elliprf(0.0, kc2, 1.0)
        RD = special.elliprd(0.0, kc2, 1.0)
        K = RF
        E = RF - (m / 3.0) * RD
        
        sqrtQ = np.sqrt(Q)
        
        # Calculate Fields
        Bz[off_axis] = (MU0 * I) / (2*np.pi*sqrtQ) * (K + (a**2 - rr**2 - zz**2)/P * E)
        Br[off_axis] = (MU0 * I * zz) / (2*np.pi*rr*sqrtQ) * (-K + (a**2 + rr**2 + zz**2)/P * E)
        
    return Br, Bz

def precompute_coil_kernel(filename, a, L, z_center, N=400):
    """Generate unit-current field table for the Coil."""
    print(f"Generating Coil Kernel: {filename}...")
    # Ensure rho includes 0.0 for completeness
    rho = np.linspace(0.0, 0.05, N)
    z = np.linspace(-0.2, 0.2, N)
    R, Z = np.meshgrid(rho, z, indexing='xy')
    
    # Integrate loops along length
    z_prime = np.linspace(z_center - L/2, z_center + L/2, 200)
    dz = z_prime[1] - z_prime[0]
    Br_acc, Bz_acc = np.zeros_like(R), np.zeros_like(R)
    
    for zp in z_prime:
        br, bz = BrBz_loop_carlson(R, Z - zp, a)
        Br_acc += br
        Bz_acc += bz
        
    # Scale by turn density later. This is for 1 turn/meter effective.
    Br_unit = Br_acc * dz
    Bz_unit = Bz_acc * dz
    
    np.savez_compressed(filename, rho=rho, z=z, Br=Br_unit, Bz=Bz_unit, a=a, L=L, z_center=z_center)

class CoilKernelField:
    """Interpolates Coil Field for Force calculation."""
    def __init__(self, filename, n_density):
        d = np.load(filename)
        self.n_density = n_density
        self.Br_unit = RegularGridInterpolator((d['z'], d['rho']), d['Br'], bounds_error=False, fill_value=0.0)
    
    def get_Br(self, rho, z, I_coil):
        """Returns Radial field at (rho, z) for current I."""
        pts = np.stack([np.asarray(z).ravel(), np.asarray(rho).ravel()], axis=1)
        # Field = Unit_Field * n_density * Current
        return self.Br_unit(pts).reshape(np.asarray(rho).shape) * self.n_density * I_coil

# ==========================================
# 4. FORCE & FLUX CALCULATIONS
# ==========================================

def get_flux_linkage(magnet_table, coil_geom, z_mag):
    """Calculates Flux through coil via Magnet Table (Method B compatible)."""
    z_c, L, R, n = coil_geom['z_center'], coil_geom['L_coil'], coil_geom['R_coil'], coil_geom['n_density']
    
    # Grid the coil volume
    r_vals = np.linspace(0.0, R, 40)
    z_vals = np.linspace(z_c - L/2, z_c + L/2, 40)
    RR, ZZ = np.meshgrid(r_vals, z_vals, indexing='xy')
    
    # Magnet B at Coil position (shifted by magnet position z_mag)
    Bz = magnet_table.get_Bz(RR, ZZ - z_mag)
    
    # Integrate
    flux = trapezoid(trapezoid(Bz * 2 * np.pi * RR, r_vals, axis=1), z_vals, axis=0)
    return flux * n

def calculate_force_integral(coil_kernel, magnet_geom, z_mag, I_coil):
    """
    Calculates Force on Magnet via Coil Kernel (Method A).
    F_z = - integral( M * Br_coil * dA )
    """
    if abs(I_coil) < 1e-9: return 0.0
    
    R_mag, L_mag, M = magnet_geom['radius'], magnet_geom['length'], magnet_geom['M']
    
    # Force constant (Current is passed to get_Br, not here)
    # F = - integral ( K x B ) dA.  K = M.  dA = 2*pi*R*dz
    const = -2 * np.pi * R_mag * M 
    
    # Integration range: Magnet's length in Lab Frame
    z1 = z_mag - L_mag/2
    z2 = z_mag + L_mag/2
    
    # Define integrand function
    def force_integrand(z_lab):
        # We need Coil's Br at the Magnet's surface (R_mag) at position z_lab
        br_val = coil_kernel.get_Br(R_mag, z_lab, I_coil)
        return const * br_val

    # Perform integration (quad is accurate but slower)
    force, _ = quad(force_integrand, z1, z2, limit=50)
    return force

class FluxLookup:
    """Pre-computes Flux for fast EMF calculation."""
    def __init__(self, magnet_tbl, coil_geom, z_range):
        z = np.linspace(z_range[0], z_range[1], 200)
        psi = [get_flux_linkage(magnet_tbl, coil_geom, zi) for zi in z]
        self.dpsi_dz = interp1d(z, np.gradient(psi, z), kind='cubic', fill_value="extrapolate")
    
    def get_dpsi(self, z):
        return self.dpsi_dz(z)

# ==========================================
# 5. MAIN SOLVER
# ==========================================

def solve_full_force(t_span, y0, magnet_tbl, coil_kernel, flux_lookup, params):
    
    m, k, c = params['m'], params['k'], params['c']
    R, L = params['R'], params['L']
    mag_geom = params['mag_geom']
    
    def ode(t, y):
        z, v, i = y
        
        # 1. Electrical (Using Flux Gradient for EMF)
        dPsi_dz = flux_lookup.get_dpsi(z)
        emf = -dPsi_dz * v
        di_dt = (emf - R*i) / L
        
        # 2. Mechanical (Using FULL INTEGRAL for Force)
        # Note: This is the computationally heavy step!
        F_mag = calculate_force_integral(coil_kernel, mag_geom, z, i)
        
        F_net = -k*z - c*v + F_mag
        a = F_net / m
        
        return [v, a, di_dt]

    return solve_ivp(ode, t_span, y0, t_eval=np.linspace(*t_span, 1000), rtol=1e-5)

# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":
    # Parameters
    M_val = 1.0e6 # Magnetization approx 1 MA/m (N42)
    mag_conf = {'radius': 0.01, 'length': 0.05, 'M': M_val}
    coil_conf = {'z_center': 0.0, 'L_coil': 0.03, 'R_coil': 0.015, 'n_density': 1160}
    
    # NOTE: Delete old .npz files if you change geometry parameters!
    files = {'mag': 'mag_field.npz', 'coil': 'coil_kernel.npz'}
    
    # 1. Generate Tables
    if not os.path.exists(files['mag']):
        precompute_magnet_table(files['mag'], mag_conf['radius'], mag_conf['length'], mag_conf['M'])
        
    if not os.path.exists(files['coil']):
        precompute_coil_kernel(files['coil'], coil_conf['R_coil'], coil_conf['L_coil'], coil_conf['z_center'])

    # 2. Load Objects
    print("Loading tables...")
    mag_tbl = MagnetFieldTable(files['mag'])
    coil_kern = CoilKernelField(files['coil'], coil_conf['n_density'])
    
    # 3. Lookup for EMF (needed for voltage calculation)
    print("Precomputing Flux Gradient...")
    flux_lkp = FluxLookup(mag_tbl, coil_conf, (-0.2, 0.2))
    
    # 4. Solve
    print("Solving with Full Integral Force...")
    sys_params = {
        'm': 0.2, 'k': 5.0, 'c': 0.005, 
        'R': 5.0, 'L': 0.01, 
        'mag_geom': mag_conf
    }
    
    sol = solve_full_force((0, 5.0), [0.05, 0.0, 0.0], mag_tbl, coil_kern, flux_lkp, sys_params)
    
    # 5. Plot
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
    ax[0].plot(sol.t, sol.y[0]*100); ax[0].set_ylabel('Pos [cm]')
    ax[0].grid(True)
    
    ax[1].plot(sol.t, sol.y[2], 'r'); ax[1].set_ylabel('Current [A]')
    ax[1].grid(True)
    
    # Recalculate force for plotting
    f_vals = [calculate_force_integral(coil_kern, mag_conf, z, i) for z, i in zip(sol.y[0], sol.y[2])]
    ax[2].plot(sol.t, f_vals, 'g'); ax[2].set_ylabel('Force [N]')
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.show()
