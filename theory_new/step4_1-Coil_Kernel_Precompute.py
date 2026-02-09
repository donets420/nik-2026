import numpy as np
from scipy import special

MU0 = 4.0 * np.pi * 1e-7  # [T·m/A]

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



rho_grid = np.linspace(0, 0.025, 400)
z_grid   = np.linspace(-0.15, 0.15, 400)

precompute_coil_kernel_npz(
     "coil_kernel.npz",
     a=0.015, z_center=0.0, L_coil=0.1, n_density=4000,
     rho_grid=rho_grid, z_grid=z_grid,
     Nz_coil=700, wire_eps=1e-9
)
