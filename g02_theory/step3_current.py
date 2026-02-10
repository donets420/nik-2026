import numpy as np
import matplotlib.pyplot as plt

from gypt02 import MagnetFieldTable, induced_current_vs_time
from gypt02 import magnet, coil

filename = magnet["filename"]
tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

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
    coil["z_center"], coil["L_coil"], coil["R_coil"], coil["n_density"],
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