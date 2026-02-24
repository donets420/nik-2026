"""
Fit frequency separately by adjusting ONE simulation parameter (typically mech.k or mech.m).

What it does:
1) Loads experimental file ML_60mm_A1.txt (t,y in meters).
2) Centers experimental y around 0 (median subtraction).
3) Estimates experimental frequency from peak-to-peak times.
4) Sweeps/fits ONE parameter (e.g. mech.k) so that simulated frequency matches experimental frequency.
5) Plots experiment vs best-fit simulation and prints results.

Dependencies:
  pip install numpy scipy matplotlib tqdm

Put this file next to:
  - gypt02_V3_final.py
  - ML_60mm_A1.txt

Run:
  python fit_frequency_only.py
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from tqdm import tqdm

import gypt02_V3_final as sim


# ============================================================
# 1) Load experimental data (center around 0)
# ============================================================

def load_experiment_ml_txt_centered(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads ML_60mm_A1.txt-like files:
      line 1: "mass A,"
      line 2: "t,y,"
      then rows: t,y,...

    Returns:
      t [s], y_centered [m]
    """
    data = np.loadtxt(path, delimiter=",", skiprows=2)  # :contentReference[oaicite:1]{index=1}
    t = data[:, 0].astype(float)
    y = data[:, 1].astype(float)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    order = np.argsort(t)
    t, y = t[order], y[order]

    # drop duplicate times
    if len(t) >= 2:
        keep = np.ones_like(t, dtype=bool)
        keep[1:] = t[1:] > t[:-1]
        t, y = t[keep], y[keep]

    if len(t) < 5:
        raise ValueError("Experimental file seems too short after cleaning (<5 points).")

    y_centered = y - float(np.median(y))
    return t, y_centered


# ============================================================
# 2) Peak-based frequency estimator
# ============================================================

def estimate_frequency_from_peaks(
    t: np.ndarray,
    y: np.ndarray,
    *,
    min_prominence: Optional[float] = None,
    min_distance_s: Optional[float] = None
) -> float:
    """
    Estimate dominant oscillation frequency [Hz] using peak-to-peak spacing.

    - min_prominence: minimum peak prominence (in units of y). If None -> 10% of max|y|.
    - min_distance_s: minimum time between peaks, used to avoid false peaks.
                      If None -> not enforced.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if len(t) < 10:
        raise ValueError("Not enough samples to estimate frequency.")

    if min_prominence is None:
        min_prominence = 0.10 * float(np.max(np.abs(y)))

    if min_distance_s is not None:
        dt = float(np.median(np.diff(t)))
        min_distance_pts = max(1, int(min_distance_s / dt))
    else:
        min_distance_pts = 1

    peaks, _ = find_peaks(y, prominence=min_prominence, distance=min_distance_pts)

    # if positive peaks not enough, try negative peaks
    if len(peaks) < 3:
        peaks, _ = find_peaks(-y, prominence=min_prominence, distance=min_distance_pts)

    if len(peaks) < 3:
        raise ValueError("Not enough peaks to estimate frequency (need >= 3).")

    periods = np.diff(t[peaks])
    T_med = float(np.median(periods))
    if T_med <= 0:
        raise ValueError("Invalid period estimate.")
    return 1.0 / T_med


# ============================================================
# 3) Nested config set/get (mech.k etc.)
# ============================================================

def set_nested(conf: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    obj = conf
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value


# ============================================================
# 4) Simulation wrapper (caches flux lookup)
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
    def __init__(self, base: BaseConfig):
        self.base = base
        self._lookup_cache = {}

    def _ensure_magnet_table(self, magnet_conf: Dict[str, Any]) -> None:
        fname = magnet_conf["fname"]
        if not os.path.exists(fname):
            sim.precompute_field_table(
                fname,
                a=magnet_conf["diameter"] / 2.0,
                b=magnet_conf["length"] / 2.0,
                mu=magnet_conf["mu"]
            )

    def _lookup_key(self, magnet_conf: Dict[str, Any], coil_conf: Dict[str, Any]) -> Tuple:
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
        key = self._lookup_key(magnet_conf, coil_conf)
        if key in self._lookup_cache:
            return self._lookup_cache[key]

        tbl = sim.MagnetFieldTable(magnet_conf["fname"])
        lookup = sim.FluxLinkageLookup(tbl, **coil_conf)
        self._lookup_cache[key] = (tbl, lookup)
        return tbl, lookup

    def run(self, conf: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
        return sol.t, sol.y[0]


# ============================================================
# 5) Frequency-only fit
# ============================================================

def fit_frequency_only(
    runner: SimulationRunner,
    exp_t: np.ndarray,
    exp_y_centered: np.ndarray,
    base_conf: Dict[str, Any],
    fit_param: str,
    bounds: Tuple[float, float],
    *,
    time_window: Optional[Tuple[float, float]] = None,
    peak_prominence: Optional[float] = None,
    min_peak_distance_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fit ONE parameter (e.g. "mech.k") to match frequency only.

    Returns dict with best_x, f_exp, f_sim(best), etc.
    """
    # time window on experiment (recommended: early segment with large amplitude)
    if time_window is not None:
        tmin, tmax = time_window
        m = (exp_t >= tmin) & (exp_t <= tmax)
        exp_t_use = exp_t[m]
        exp_y_use = exp_y_centered[m]
    else:
        exp_t_use, exp_y_use = exp_t, exp_y_centered

    f_exp = estimate_frequency_from_peaks(
        exp_t_use, exp_y_use,
        min_prominence=peak_prominence,
        min_distance_s=min_peak_distance_s
    )

    conf0 = copy.deepcopy(base_conf)

    # Ensure sim time covers the window
    if time_window is not None:
        runner.base.t_span = (min(runner.base.t_span[0], time_window[0]),
                              max(runner.base.t_span[1], time_window[1]))

    pbar = tqdm(desc=f"Freq-fit {fit_param}", total=1, unit="opt", leave=True)
    eval_n = {"n": 0}

    def objective(x: float) -> float:
        eval_n["n"] += 1
        conf = copy.deepcopy(conf0)
        set_nested(conf, fit_param, float(x))
        sim_t, sim_z = runner.run(conf)

        if time_window is not None:
            m2 = (sim_t >= time_window[0]) & (sim_t <= time_window[1])
            sim_t2, sim_z2 = sim_t[m2], sim_z[m2]
        else:
            sim_t2, sim_z2 = sim_t, sim_z

        f_sim = estimate_frequency_from_peaks(
            sim_t2, sim_z2,
            min_prominence=peak_prominence,
            min_distance_s=min_peak_distance_s
        )

        val = (f_sim - f_exp) ** 2
        pbar.set_postfix({
            "eval": eval_n["n"],
            "x": f"{x:.6g}",
            "f_exp": f"{f_exp:.4g}",
            "f_sim": f"{f_sim:.4g}",
        })
        return val

    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-6})
    pbar.close()

    best_x = float(res.x)
    best_conf = copy.deepcopy(conf0)
    set_nested(best_conf, fit_param, best_x)

    best_t, best_z = runner.run(best_conf)
    if time_window is not None:
        m3 = (best_t >= time_window[0]) & (best_t <= time_window[1])
        best_t2, best_z2 = best_t[m3], best_z[m3]
    else:
        best_t2, best_z2 = best_t, best_z

    f_sim_best = estimate_frequency_from_peaks(
        best_t2, best_z2,
        min_prominence=peak_prominence,
        min_distance_s=min_peak_distance_s
    )

    return {
        "fit_param": fit_param,
        "best_x": best_x,
        "f_exp": float(f_exp),
        "f_sim_best": float(f_sim_best),
        "success": bool(res.success),
        "message": str(res.message),
        "best_conf": best_conf,
        "best_t": best_t,
        "best_z": best_z,
        "exp_t_use": exp_t_use,
        "exp_y_use": exp_y_use,
        "time_window": time_window,
    }


# ============================================================
# 6) Main (edit parameters here)
# ============================================================

if __name__ == "__main__":
    # ---- baseline configuration (edit to your setup) ----
    base = BaseConfig(
        magnet={
            "length": 0.05,
            "diameter": 0.02,
            "mu": 19.7556,
            "fname": "magnet_field_v3.npz",
        },
        coil={
            "z_center": 0.01,
            "L_coil": 0.1,
            "R_coil": 0.0182,
            "N_turns": 1167,
        },
        mech={
            "m": 5 * 0.0237,
            "k": 5.15,
            "c": 0.00690676,
            "g": 9.81,
            "z_top": 0.2,
            "z_bot": -0.2,
        },
        elec={
            "R": 47.0,
        },
        y0=(0.05, 0.0),
        t_span=(0.0, 10.0),  # keep it short for frequency fit
    )

    runner = SimulationRunner(base)

    base_conf = {
        "magnet": base.magnet,
        "coil": base.coil,
        "mech": base.mech,
        "elec": base.elec,
    }

    # ---- load experiment ----
    exp_path = r"C:\Users\petrn\Desktop\GYPT26\Experiments\Bundes\Magnet_lenght\ML_60mm\Amplitude\ML_60mm_A1.txt"
    exp_t, exp_y_centered = load_experiment_ml_txt_centered(exp_path)

    # ---- choose what parameter sets the frequency ----
    # Most typical: "mech.k" (spring constant) or "mech.m" (effective mass)
    fit_param = "mech.k"
    bounds = (0.5, 20.0)  # N/m example bounds, adjust!

    # ---- choose an early time window where peaks are clear ----
    time_window = (0.0, 4.0)  # seconds; adjust or set to None

    # Optional tuning: peak detection
    # If you know approx frequency f~2 Hz, you can set min_peak_distance_s~0.3..0.4
    peak_prominence = None
    min_peak_distance_s = None

    result = fit_frequency_only(
        runner=runner,
        exp_t=exp_t,
        exp_y_centered=exp_y_centered,
        base_conf=base_conf,
        fit_param=fit_param,
        bounds=bounds,
        time_window=time_window,
        peak_prominence=peak_prominence,
        min_peak_distance_s=min_peak_distance_s,
    )

    print("\n=== Frequency fit result ===")
    print(f"Fit parameter: {result['fit_param']}")
    print(f"Best value:   {result['best_x']:.10g}")
    print(f"f_exp:        {result['f_exp']:.10g} Hz")
    print(f"f_sim(best):  {result['f_sim_best']:.10g} Hz")
    print(f"Success:      {result['success']}  |  {result['message']}")

    # ---- plots ----
    plt.figure(figsize=(10, 5))
    plt.plot(exp_t, exp_y_centered, label="Experiment (centered)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [m] (centered)")
    plt.title("Experimental signal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot simulation best-fit (raw displacement) vs experiment window
    plt.figure(figsize=(10, 5))
    if result["time_window"] is not None:
        tmin, tmax = result["time_window"]
        mexp = (exp_t >= tmin) & (exp_t <= tmax)
        msim = (result["best_t"] >= tmin) & (result["best_t"] <= tmax)
        plt.plot(exp_t[mexp], exp_y_centered[mexp], label="Experiment window")
        plt.plot(result["best_t"][msim], result["best_z"][msim], label="Simulation (best) window")
        plt.title(f"Windowed comparison | f_exp={result['f_exp']:.3g} Hz, f_sim={result['f_sim_best']:.3g} Hz")
    else:
        plt.plot(exp_t, exp_y_centered, label="Experiment")
        plt.plot(result["best_t"], result["best_z"], label="Simulation (best)")
        plt.title(f"Full comparison | f_exp={result['f_exp']:.3g} Hz, f_sim={result['f_sim_best']:.3g} Hz")

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [m] / sim units")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()