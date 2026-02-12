import numpy as np
from scipy.integrate import solve_ivp

def eom_rhs(t, y, m, c, k, z_eq, force_z, force_args):
    """
    t : Current simulation time [s].

    y : array_like, shape (2,)
        State vector of the system:
        y[0] = z  -> position [m]
        y[1] = v  -> velocity dz/dt [m/s]

    m : Mass of the oscillator [kg].

    c : Damping coefficient [N·s/m].
        Represents viscous damping force: F_damp = -c * v.

    k : Spring constant [N/m].

    z_eq : Equilibrium position of the spring [m].

    force_z : callable
        External force function.
        Must have the signature:
            force_z(t, z, v, **force_args)
        and return the force in z-direction [N].

    force_args : dict
        Dictionary containing additional parameters required
        by force_z (e.g. coil geometry, magnetic moment,
        current, lookup tables, etc.).
    """

    z, v = y

    # External force (your custom function)
    F = force_z(t, z, v, **force_args)

    dzdt = v
    dvdt = (F - c * v - k * (z - z_eq)) / m
    return [dzdt, dvdt]


def solve_oscillator(
    t_span,
    z0,
    v0,
    m,
    c,
    k,
    z_eq=0.0,
    force_z=None,
    force_args=None,
    t_eval=None,
    rtol=1e-8,
    atol=1e-10,
    method="RK45",
):

    if force_args is None:
        force_args = {}

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)

    y0 = [z0, v0]

    sol = solve_ivp(
        fun=lambda t, y: eom_rhs(t, y, m, c, k, z_eq, force_z, force_args),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    # sol.y[0] = z(t), sol.y[1] = v(t)
    return sol
