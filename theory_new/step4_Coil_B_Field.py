import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from pathlib import Path

from step1_1_M_FIeld_Precomputation import MagnetFieldTable
from step3_Current import induced_current_vs_time

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

# -------------------- Example usage --------------------

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# -------------------- Load data --------------------
filename_mag = "magnet_field_L10cm_D2cm_mu3.npz"
filename_kernel = "coil_kernel.npz"

assert Path(filename_mag).exists(), f"Missing: {filename_mag}"
assert Path(filename_kernel).exists(), f"Missing: {filename_kernel}"

tbl  = MagnetFieldTable(filename_mag, method="linear", bounds_error=False, fill_value=0.0)
coil = CoilKernelField(filename_kernel, method="linear", bounds_error=True)  # keep strict; Fix A prevents OOB

print("Kernel rho range:", float(coil.rho.min()), " .. ", float(coil.rho.max()))
print("Kernel z   range:", float(coil.z.min()),   " .. ", float(coil.z.max()))

# Coil geometry (must match kernel precompute)
z_center  = coil.z_center
L_coil    = coil.L_coil
R_coil    = coil.a
n_density = coil.n_density

# Circuit
R_total = 5.0       # [Ohm]
L_self  = 20e-3     # [H]

# Magnet geometry (for drawing only)
R_m = 0.01          # [m] magnet radius  (2 cm diameter -> 1 cm radius)
L_m = 0.10          # [m] magnet length  (10 cm)

# -------------------- Harmonic magnet motion --------------------
A = 0.02            # [m] amplitude (choose so validation passes)
f = 2.0             # [Hz]
w = 2.0 * np.pi * f
z0 = 0.0

z_mag_func = lambda t: z0 + A * np.cos(w * t)

t_vals = np.linspace(0.0, 2.0, 2000)

# -------------------- Induced current i(t) --------------------
t, i = induced_current_vs_time(
    tbl, z_center, L_coil, R_coil, n_density,
    t_vals, z_mag_func,
    R_total=R_total, L_self=L_self,
    i0=0.0,
    use_lookup=True,
    Nr_flux=80, Nz_flux=80
)

z_mag = np.array([z_mag_func(tt) for tt in t])

# Example calculation output: Bz of the coil at the magnet center over time
Bz_center = coil.Bz(0.0, z_mag, i)

print("\n---- Example results ----")
print(f"peak |i(t)|: {np.max(np.abs(i)):.6g} A")
print(f"RMS i(t)  : {np.sqrt(np.mean(i*i)):.6g} A")
print(f"peak |Bz_coil(center)|: {np.max(np.abs(Bz_center)):.6g} T")

# Optional quick plots (can comment out)
plt.figure()
plt.plot(t, i)
plt.xlabel("t [s]")
plt.ylabel("i(t) [A]")
plt.grid(True)

plt.figure()
plt.plot(t, Bz_center)
plt.xlabel("t [s]")
plt.ylabel("Bz_coil(center) [T]")
plt.grid(True)
plt.show()

# ==================== Animation (Fix A applied) ====================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d

# -------------------- Inputs (use your existing values) --------------------
# coil kernel geometry (from file)
z_center  = coil.z_center
L_coil    = coil.L_coil
R_coil    = coil.a

# magnet geometry (drawing only)
R_m = 0.01
L_m = 0.10

# harmonic motion (same you used for z_mag_func)
# A = 0.02
# f = 2.0
w = 2.0 * np.pi * f

# -------------------- Build smooth animation time base --------------------
fps = 60                 # smoother than 30
Nframes = 720            # more frames -> smoother reversal
t_anim = np.linspace(t[0], t[-1], Nframes)

# smooth current interpolation (cubic looks nice; linear also ok)
i_of_t = interp1d(t, i, kind="cubic", bounds_error=False, fill_value=(i[0], i[-1]))
i_anim = i_of_t(t_anim)

# harmonic magnet position on same animation timeline
z_mag_anim = z0 + A * np.cos(w * t_anim)

# -------------------- Build plotting grid using the KERNEL GRID (no OOB, very fast) --------------------
rho_k = coil.rho
z_k   = coil.z

# Desired view window (will be clamped to kernel coverage)
xlim_desired = max(2.5 * R_coil, 2.5 * R_m)
zlim_desired = max(1.5 * L_m, 1.5 * L_coil + A)

# Clamp x (rho) range
rho_max = float(rho_k.max())
xlim = min(xlim_desired, rho_max)
rho_sub = rho_k[rho_k <= xlim]                 # includes 0
Nr = len(rho_sub)

# Clamp z range
z_min_k = float(z_k.min())
z_max_k = float(z_k.max())
z_low  = max(-zlim_desired, z_min_k)
z_high = min( zlim_desired, z_max_k)
z_sub = z_k[(z_k >= z_low) & (z_k <= z_high)]
Nz = len(z_sub)

# Extract unit-current fields on the subgrid (already tabulated, no interpolation needed)
# Br_unit, Bz_unit arrays are shaped (len(z_k), len(rho_k))
# Find indices for subsetting:
rho_idx = np.where(rho_k <= xlim)[0]
z_idx   = np.where((z_k >= z_low) & (z_k <= z_high))[0]

Br_u = coil.Br_unit[np.ix_(z_idx, rho_idx)]    # (Nz, Nr)
Bz_u = coil.Bz_unit[np.ix_(z_idx, rho_idx)]    # (Nz, Nr)

# Build symmetric x grid from rho_sub: x = [-rho...0...+rho]
x_vals = np.concatenate((-rho_sub[:0:-1], rho_sub))  # negative (exclude 0) + nonnegative
# Mirror fields into x-z plane:
# Negative x side uses same |rho| but Bx flips sign; Bz stays same
Br_neg = Br_u[:, 1:][:, ::-1]    # omit rho=0 then reverse
Bz_neg = Bz_u[:, 1:][:, ::-1]
Bx_u_map = np.concatenate((-Br_neg, Br_u), axis=1)   # (Nz, 2*Nr-1)
Bz_u_map = np.concatenate(( Bz_neg, Bz_u), axis=1)

# Mesh for imshow and quiver
X, Z = np.meshgrid(x_vals, z_sub, indexing="xy")

# Coarser grid for quiver arrows (speed)
step_x = max(1, (2*Nr-1) // 40)   # ~40 arrows across
step_z = max(1, Nz // 50)         # ~50 arrows vertically
Xq = X[::step_z, ::step_x]
Zq = Z[::step_z, ::step_x]
Bx_u_q = Bx_u_map[::step_z, ::step_x]
Bz_u_q = Bz_u_map[::step_z, ::step_x]

# -------------------- Arrow scaling (magnitude-based => smooth reversal) --------------------
# Choose arrow max length as ~8% of the smaller axis range
axis_span = min(float(x_vals.max() - x_vals.min()), float(z_sub.max() - z_sub.min()))
arrow_max_len = 0.08 * axis_span

# Reference field magnitude for scaling arrows (based on max current and max unit field)
B_unit_max = float(np.nanmax(np.sqrt(Bx_u_map*Bx_u_map + Bz_u_map*Bz_u_map)))
I_max = float(np.max(np.abs(i_anim)))
B_max = max(1e-30, I_max * B_unit_max)

# Convert Tesla vectors -> plot-length vectors
vec_scale = arrow_max_len / B_max   # [m/T] so U,V are in meters (data units)

# -------------------- Initial frame --------------------
I0 = float(i_anim[0])
Bx0 = Bx_u_map * I0
Bz0 = Bz_u_map * I0
Bmag0 = np.sqrt(Bx0*Bx0 + Bz0*Bz0)

fig, ax = plt.subplots(figsize=(7.2, 7.2))

# background: magnitude (helps see activity while arrows show direction)
im = ax.imshow(
    Bmag0, origin="lower",
    extent=[x_vals.min(), x_vals.max(), z_sub.min(), z_sub.max()],
    vmin=0.0, vmax=B_max,
    aspect="auto",
)
cb = fig.colorbar(im, ax=ax)
cb.set_label("|B_coil| [T]")

# quiver: magnitude-scaled (smoothly shrinks to zero near i(t)=0)
U0 = (Bx_u_q * I0) * vec_scale
V0 = (Bz_u_q * I0) * vec_scale
quiv = ax.quiver(
    Xq, Zq, U0, V0,
    angles="xy", scale_units="xy", scale=1.0,   # scale=1 uses U,V in data units
    width=0.003
)

# coil outline
coil_rect = Rectangle(
    (-R_coil, z_center - L_coil/2),
    2*R_coil, L_coil,
    fill=False, linewidth=2
)
ax.add_patch(coil_rect)

# magnet outline (moving)
mag_rect = Rectangle(
    (-R_m, z_mag_anim[0] - L_m/2),
    2*R_m, L_m,
    fill=False, linewidth=2
)
ax.add_patch(mag_rect)

title = ax.set_title("")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")

def update(k):
    Ik = float(i_anim[k])

    # update background |B|
    Bx = Bx_u_map * Ik
    Bz = Bz_u_map * Ik
    im.set_data(np.sqrt(Bx*Bx + Bz*Bz))

    # update quiver vectors (smooth reversal)
    U = (Bx_u_q * Ik) * vec_scale
    V = (Bz_u_q * Ik) * vec_scale
    quiv.set_UVC(U, V)

    # update magnet position (clamp drawing to visible range is optional; not required)
    mag_rect.set_xy((-R_m, float(z_mag_anim[k]) - L_m/2))

    title.set_text(
        f"t={t_anim[k]:.3f} s,  z_mag={z_mag_anim[k]:+.3f} m,  i={Ik:+.4f} A"
    )
    return im, quiv, mag_rect, title

ani = FuncAnimation(fig, update, frames=len(t_anim), interval=1000/fps, blit=False)

out_gif = "coil_vector_field_smooth.gif"
ani.save(out_gif, writer=PillowWriter(fps=fps))
plt.close(fig)

print(f"Saved: {out_gif}")

