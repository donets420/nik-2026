import numpy as np
from gypt02 import precompute_coil_kernel_npz

from gypt02 import MU0, coil, grid_coil

precompute_coil_kernel_npz(
     coil["filename"],
     a=coil["R_coil"], z_center=coil["z_center"], L_coil=coil["L_coil"], n_density=coil["n_density"],  # Could we use parameters from coil?
     rho_grid=grid_coil["rho_grid"], z_grid=grid_coil["z_grid"],
     Nz_coil=grid_coil["Nz_coil"], wire_eps=grid_coil["wire_eps"]
)
