from gypt02 import precompute_field_table_uniform, MagnetFieldTable, plot_streamlines_xz
from gypt02 import magnet, mu, grid

# Precompute field table over your region (rho<=10 cm, |z|<=30 cm)

# Define file name based on magnet parameters
filename = magnet["filename"]

precompute_field_table_uniform(
    filename,
    a=magnet["radius"], b=magnet["half_length"], mu=mu,
    rho_max=grid["rho_max"], Nrho=grid["Nrho"],
    zmax=grid["zmax"], Nz=grid["Nz"]
)

# Load the file
tbl = MagnetFieldTable(filename, method="linear", bounds_error=False, fill_value=0.0)

# Plot
plot_streamlines_xz(tbl)