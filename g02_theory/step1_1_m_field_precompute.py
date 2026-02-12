from gypt02 import precompute_field_table_uniform, MagnetFieldTable, plot_streamlines_xz
from gypt02 import magnet, mu, grid_magnet

# Precompute field table over your region (rho<=10 cm, |z|<=30 cm)

# Define file name based on magnet parameters
filename = magnet["filename"]

precompute_field_table_uniform(
    filename,
    a=magnet["radius"], b=magnet["half_length"], mu=mu,
    rho_max=grid_magnet["rho_max"], Nrho=grid_magnet["Nrho"],
    zmax=grid_magnet["zmax"], Nz=grid_magnet["Nz"]
)

# Load the file
tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

# Plot
plot_streamlines_xz(tbl)