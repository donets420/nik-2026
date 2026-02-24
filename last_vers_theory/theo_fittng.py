import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# Your simulation module (must be in same folder / PYTHONPATH)
import gypt02_V3_final as sim  # :contentReference[oaicite:0]{index=0}


# ============================================================
# 1) Experimental data loader (centers EXP around 0)
# ============================================================

def load_experiment_ml_txt_centered(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads ML_60mm_A1.txt-like files:
      line 1: "mass A,"
      line 2: "t,y,"
      then rows: t,y,...

    Returns:
      t [s], y_centered [m] where y_centered = y - median(y)
    """
    # This file has 2 header rows in the example. :contentReference[oaicite:1]{index=1}
    data = np.loadtxt(path, delimiter=",", skiprows=2)

    t = data[:, 0].astype(float)
    y = data[:, 1].astype(float)

    # clean
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    # sort by time
    order = np.argsort(t)
    t, y = t[order], y[order]

    # drop duplicate times
    if len(t) >= 2:
        keep = np.ones_like(t, dtype=bool)
        keep[1:] = t[1:] > t[:-1]
        t, y = t[keep], y[keep]

    if len(t) < 5:
        raise ValueError("Experimental file seems too short after cleaning (<5 points).")

    # center around 0 (DC offset removal)
    y_centered = y - float(np.median(y))
    return t, y_centered


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def best_scale_only(sim_y: np.ndarray, exp_y: np.ndarray) -> float:
    """
    Find alpha minimizing || alpha*sim_y - exp_y ||_2 (no offset term).
    Use this because exp was explicitly centered around 0.
    """
    denom = float(np.dot(sim_y, sim_y))
    if denom <= 1e-30:
        return 0.0
    return float(np.dot(sim_y, exp_y) / denom)


# ============================================================
# 2) Parameter accessors: set "mech.c" etc.
# ============================================================

def get_nested(conf: Dict[str, Any], dotted: str) -> Any:
    keys = dotted.split(".")
    obj = conf
    for k in keys:
        obj = obj[k]
    return obj


def set_nested(conf: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    obj = conf
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value


# ============================================================
# 3) Simulation wrapper with caching
# ============================================================

@dataclass
class BaseConfig:
    magnet: Dict[str, Any]
    coil: Dict[str, Any]
    mech: Dict[str, Any]
    elec: Dict[str, Any]
    y0: Tuple[float, float]          # [z_disp0, v0]
    t_span: Tuple[float, float]      # (t0, t1)


class SimulationRunner:
    """
    Wraps gypt02_V3_final.py simulation with caching of FluxLinkageLookup.
    """

    def __init__(self, base: BaseConfig):
        self.base = base
        self._lookup_cache = {}  # key -> (tbl, lookup)

    def _ensure_magnet_table(self, magnet_conf: Dict[str, Any]) -> None:
        fname = magnet_conf["fname"]
        if not os.path.exists(fname):
            sim.precompute_field_table(
                fname,
                a=magnet_conf["diameter"] / 2.0,
                b=magnet_conf["length"] / 2.0,
                mu=magnet_conf["mu"]
            )

    def _lookup_cache_key(self, magnet_conf: Dict[str, Any], coil_conf: Dict[str, Any]) -> Tuple:
        # If you fit parameters that change magnet/coil geometry or mu, this key changes,
        # forcing a rebuild of the lookup.
        return (
            magnet_conf["fname"],
            magnet_conf["length"],
            magnet_conf["diameter"],
            magnet_conf["mu"],
            coil_conf["z_center"],
            coil_conf["L_coil"],
            coil_conf["R_coil"],
            coil_conf["N_turns"],
        )

    def _get_lookup(self, magnet_conf: Dict[str, Any], coil_conf: Dict[str, Any]):
        self._ensure_magnet_table(magnet_conf)
        key = self._lookup_cache_key(magnet_conf, coil_conf)

        if key in self._lookup_cache:
            return self._lookup_cache[key]

        tbl = sim.MagnetFieldTable(magnet_conf["fname"])
        lookup = sim.FluxLinkageLookup(tbl, **coil_conf)
        self._lookup_cache[key] = (tbl, lookup)
        return tbl, lookup

    def run(self, conf: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (t, z_disp(t)) from the simulation.
        NOTE: Per your request, we do NOT center the simulation here.
        """
        magnet_conf = conf["magnet"]
        coil_conf = conf["coil"]
        mech_conf = conf["mech"]
        elec_conf = conf["elec"]

        _, lookup = self._get_lookup(magnet_conf, coil_conf)

        sol, _z_eq = sim.solve_coupled_dynamics(
            t_span=self.base.t_span,
            y0=list(self.base.y0),
            flux_lookup=lookup,
            mech=mech_conf,
            elec=elec_conf
        )
        t = sol.t
        z_disp = sol.y[0]
        return t, z_disp


# ============================================================
# 4) Fit routine (fit one parameter to centered EXP)
# ============================================================

def fit_one_parameter(
    runner: SimulationRunner,
    exp_t: np.ndarray,
    exp_y_centered: np.ndarray,
    base_conf: Dict[str, Any],
    fit_param: str,
    bounds: Tuple[float, float],
    *,
    exp_time_window: Optional[Tuple[float, float]] = None,
    sim_time_padding: float = 0.0,
) -> Dict[str, Any]:
    """
    Fits a single parameter by minimizing RMSE between exp_y_centered and alpha*sim(t) (no offset).
    """

    # Restrict experimental window if requested
    if exp_time_window is not None:
        tmin, tmax = exp_time_window
        m = (exp_t >= tmin) & (exp_t <= tmax)
        exp_t_use = exp_t[m]
        exp_y_use = exp_y_centered[m]
        if len(exp_t_use) < 5:
            raise ValueError("Too few experimental points left after applying exp_time_window.")
    else:
        exp_t_use = exp_t
        exp_y_use = exp_y_centered

    conf0 = copy.deepcopy(base_conf)

    # Ensure simulation span covers the exp time range
    t0 = float(min(runner.base.t_span[0], exp_t_use[0]))
    t1 = float(max(runner.base.t_span[1], exp_t_use[-1] + sim_time_padding))
    runner.base.t_span = (t0, t1)

    pbar = tqdm(total=1, desc=f"Fitting {fit_param}", unit="opt", leave=True)
    eval_count = {"n": 0}

    def objective(x: float) -> float:
        eval_count["n"] += 1

        conf = copy.deepcopy(conf0)
        set_nested(conf, fit_param, float(x))

        sim_t, sim_z = runner.run(conf)

        # Interpolate simulation onto experimental time grid
        sim_interp = interp1d(sim_t, sim_z, kind="linear", bounds_error=False, fill_value="extrapolate")
        sim_on_exp = sim_interp(exp_t_use)

        # scale-only fit (no offset) because experiment is centered around 0
        alpha = best_scale_only(sim_on_exp, exp_y_use)
        pred = alpha * sim_on_exp

        val = rmse(pred, exp_y_use)

        pbar.set_postfix({
            "eval": eval_count["n"],
            "x": f"{x:.6g}",
            "rmse": f"{val:.6g}",
            "alpha": f"{alpha:.3g}",
        })
        return val

    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-6})
    pbar.close()

    best_x = float(res.x)
    best_conf = copy.deepcopy(conf0)
    set_nested(best_conf, fit_param, best_x)

    best_t, best_z = runner.run(best_conf)

    best_interp = interp1d(best_t, best_z, kind="linear", bounds_error=False, fill_value="extrapolate")
    best_sim_on_exp = best_interp(exp_t_use)

    best_alpha = best_scale_only(best_sim_on_exp, exp_y_use)
    best_pred = best_alpha * best_sim_on_exp

    return {
        "fit_param": fit_param,
        "best_x": best_x,
        "rmse": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
        "alpha": float(best_alpha),
        "exp_t": exp_t_use,
        "exp_y_centered": exp_y_use,
        "sim_t": best_t,
        "sim_z": best_z,
        "sim_on_exp": best_sim_on_exp,
        "pred_on_exp": best_pred,
        "best_conf": best_conf,
    }


# ============================================================
# 5) Example usage
# ============================================================

if __name__ == "__main__":
    # ------------------------
    # A) Baseline configs
    # ------------------------
    base = BaseConfig(
        magnet={
            "length": 0.05,
            "diameter": 0.02,
            "mu": 14.5713,
            "fname": "magnet_field_v3.npz",
        },
        coil={
            "z_center": 0.0055,
            "L_coil": 0.1,
            "R_coil": 0.0356/2,
            "N_turns": 1167,
        },
        mech={
            "m": 5 * 0.0237,
            "k": 4.51573827,
            "c": 0.01327113799,
            "g": 9.81,
            "z_top": 0.2,
            "z_bot": -0.2,
        },
        elec={
            "R": 3.234,
        },
        y0=(-0.053, 0.0),
        t_span=(0.0, 30.0),
    )

    runner = SimulationRunner(base)

    base_conf = {
        "magnet": base.magnet,
        "coil": base.coil,
        "mech": base.mech,
        "elec": base.elec,
    }

    # ------------------------
    # B) Load experiment (CENTERED around 0)
    # ------------------------
    exp_path = r"C:\Users\petrn\Desktop\GYPT26\Experiments\Bundes\Magnet_lenght\ML_60mm\Amplitude\ML_60mm_A1.txt"
    exp_t, exp_y_centered = load_experiment_ml_txt_centered(exp_path)

    # ------------------------
    # C) Choose fit parameter
    # ------------------------
    fit_param = "elec.R"
    bounds = (3.234, 3.234)  # choose sensible physical limits

    # Optional: fit only part of the time series
    exp_time_window = None  # e.g. (0.0, 15.0)

    result = fit_one_parameter(
        runner=runner,
        exp_t=exp_t,
        exp_y_centered=exp_y_centered,
        base_conf=base_conf,
        fit_param=fit_param,
        bounds=bounds,
        exp_time_window=exp_time_window,
        sim_time_padding=0.0,
    )

    print("\n=== Fit result ===")
    print(f"Parameter: {result['fit_param']}")
    print(f"Best value: {result['best_x']:.10g}")
    print(f"RMSE: {result['rmse']:.10g}")
    print(f"Scale alpha (exp ≈ alpha * sim): {result['alpha']:.10g}")
    print(f"Success: {result['success']}  |  {result['message']}")

    # ------------------------
    # D) Plot experiment vs best fit
    # ------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(result["exp_t"], result["exp_y_centered"], label="Experiment (centered)")
    plt.plot(result["exp_t"], result["pred_on_exp"], label="Best-fit simulation (scaled)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [m] (centered)")
    plt.title(f"Fit of {fit_param} = {result['best_x']:.4g}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
