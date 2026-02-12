import numpy as np
import matplotlib.pyplot as plt
from gypt02 import MagnetFieldTable, get_coil_flux_linkage, flux_linkage_vs_time
from gypt02 import magnet, coil

# Define file name with M.F predefined based on magnet's size
filename = magnet["filename"]

# Read the precomputed field table from file
tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

# Magnet motion: z_mag(t) = position of magnet center along z axis [m]
A = 0.05  # [m]
f = 1.0   # [Hz]

def z_mag(t):
    return -A * np.sin(2 * np.pi * f * t)

z_mag_range = (-A, A)

# Flux linkage at t=0 (magnet centered at z=0)
psi_0 = get_coil_flux_linkage(tbl, coil["z_center"], coil["L_coil"], coil["R_coil"], coil["n_density"],
                              z_mag=0.0, z_mag_range=z_mag_range)

print(f"Psi(t=0) = {psi_0:.6e} Wb·turn")

# Flux linkage vs time
t_start, t_end, n_points = 0.0, 1.0, 101
t_vals = np.linspace(t_start, t_end, n_points)

t_arr, psi_arr = flux_linkage_vs_time(tbl, coil["z_center"], coil["L_coil"], coil["R_coil"], coil["n_density"],
                                      t_vals, z_mag, z_mag_range=z_mag_range)
# Plot

plt.plot(t_arr, psi_arr)
plt.show()