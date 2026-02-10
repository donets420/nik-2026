import numpy as np
from gypt02 import precompute_coil_kernel_npz

from gypt02 import MU0

coil_file = "coil_kernel.npz" # Could we use here coil["filename"] instead?

rho_grid = np.linspace(0, 0.025, 400)
z_grid   = np.linspace(-0.15, 0.15, 400)

precompute_coil_kernel_npz(
     coil_file,
     a=0.015, z_center=0.0, L_coil=0.1, n_density=4000,  # Could we use parameters from coil?
     rho_grid=rho_grid, z_grid=z_grid,
     Nz_coil=700, wire_eps=1e-9
)