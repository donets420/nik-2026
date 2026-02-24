"""
run_halfdecay_sweep.py

- Edit PARAMS in-code (no CLI).
- Runs your existing simulation from gypt02_V3_final.py
- Computes half-decay time from peak amplitudes of |z(t)| (includes t=0 endpoint peak)
- Sweeps ONE parameter over a range, shows a loading bar, and plots T_half vs parameter

Put this file in the same folder as: gypt02_V3_final.py
"""

from __future__ import annotations

import sys
import os
import copy
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Progress bar (optional)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Your existing simulation module
import gypt02_V3_final as sim


# ============================================================
# 1) EDIT PARAMETERS HERE (base config)
# ============================================================

PARAMS: Dict = {
    # Simulation time + initial conditions
    "t_end": 20.0,
    "z0": -0.053,
    "v0": 0.0,

    # Magnet
    "magnet": {
        "length": 0.05,
        "diameter": 0.02,
        "mu": 14.5713,
        "field_table_file": "magnet_field_v3.npz",
    },

    # Coil
    "coil": {
        "z_center": 0.0,
        "L_coil": 0.10,
        "R_coil": 0.0178,
        "N_turns": 1167,
    },

    # Mechanics
    "mech": {
        "m": 5 * 0.0237,
        "k": 5.15,
        "c": 0.00690676,
        "g": 9.81,
        "z_top": 0.2,
        "z_bot": -0.2,
    },

    # Electrical
    "elec": {
        "R": 200,
    },

    # Half-decay peak detection tuning
    "peaks": {
        # If you see too many spurious peaks, set this e.g. to 0.0002 (0.2 mm)
        "min_prominence": 0.0,      # meters on |z|
        # If you want to avoid peaks too close in time (noise), set e.g. 0.2
        "min_distance_s": None,     # seconds
        # For accepting endpoint peaks (t=0 etc.)
        "endpoint_eps": 1e-15,
    },

    # Plotting
    "plot_single_run": True,   # plot one representative run with peaks + T_half
    "plot_sweep": True,        # plot sweep curve
}


# ============================================================
# 2) Parameter path helpers
# ============================================================

def set_param_by_path(params: dict, path: Tuple[str, ...], value) -> None:
    """
    Example:
      set_param_by_path(PARAMS, ("mech","c"), 0.01)
      set_param_by_path(PARAMS, ("coil","z_center"), -0.02)
    """
    d = params
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def get_param_by_path(params: dict, path: Tuple[str, ...]):
    d = params
    for key in path:
        d = d[key]
    return d


# ============================================================
# 3) Peaks + Half-decay time (includes endpoint peak support)
# ============================================================

def _find_peaks_abs_with_endpoints(
    t: np.ndarray,
    z: np.ndarray,
    *,
    min_prominence: float = 0.0,
    min_distance_s: Optional[float] = None,
    endpoint_eps: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in |z(t)| and ALSO include endpoints if they look like peaks.
    This fixes the "first peak isn't tracked" issue when the max amplitude is at t=0.

    Returns:
      peak_times (1D)
      peak_amps  (1D)  = |z| at those times
    """
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    y = np.abs(z)

    if t.size < 3:
        return np.array([]), np.array([])

    # Convert distance in seconds to samples (approx.)
    distance_samples = None
    if min_distance_s is not None and min_distance_s > 0:
        dt = np.median(np.diff(t))
        if dt <= 0:
            raise ValueError("t must be strictly increasing.")
        distance_samples = max(1, int(round(min_distance_s / dt)))

    peaks_mid, _props = find_peaks(
        y,
        prominence=min_prominence,
        distance=distance_samples
    )

    peak_idx = list(peaks_mid.tolist())

    # Endpoint check: include first point if it looks like a peak
    # (peak-like if y[0] >= y[1] within eps and above min_prominence threshold)
    if (y[0] >= y[1] - endpoint_eps) and (y[0] >= min_prominence):
        peak_idx.append(0)

    # Endpoint check: include last point if it looks like a peak
    if (y[-1] >= y[-2] - endpoint_eps) and (y[-1] >= min_prominence):
        peak_idx.append(t.size - 1)

    peak_idx = np.array(sorted(set(peak_idx)), dtype=int)

    peak_times = t[peak_idx]
    peak_amps = y[peak_idx]

    # Re-apply min_distance_s in time-domain including endpoints (simple greedy filter)
    if min_distance_s is not None and min_distance_s > 0 and peak_times.size >= 2:
        keep = [0]
        for i in range(1, peak_times.size):
            if (peak_times[i] - peak_times[keep[-1]]) >= min_distance_s:
                keep.append(i)
        peak_times = peak_times[keep]
        peak_amps = peak_amps[keep]

    return peak_times, peak_amps


def half_decay_time_from_peak_sequence(
    peak_times: np.ndarray,
    peak_amps: np.ndarray
) -> Dict[str, object]:
    """
    Operational half-decay time:
      A0 = first peak amplitude (|z|)
      A_half = 0.5 * A0
      Find first pair of consecutive peaks where it crosses below A_half
      Interpolate linearly between those two peaks to estimate T_half.
    """
    peak_times = np.asarray(peak_times, float)
    peak_amps = np.asarray(peak_amps, float)

    if peak_times.size < 2:
        return {
            "T_half": np.nan,
            "A0": np.nan,
            "A_half": np.nan,
            "peak_times": peak_times,
            "peak_amps": peak_amps,
            "crossing_bracket": None,
            "reason": "Not enough peaks found."
        }

    A0 = float(peak_amps[0])
    A_half = 0.5 * A0

    bracket = None
    for k in range(len(peak_amps) - 1):
        if peak_amps[k] >= A_half and peak_amps[k + 1] < A_half:
            bracket = (k, k + 1)
            break

    if bracket is None:
        return {
            "T_half": np.nan,
            "A0": A0,
            "A_half": A_half,
            "peak_times": peak_times,
            "peak_amps": peak_amps,
            "crossing_bracket": None,
            "reason": "Amplitude did not fall below half within the simulated time."
        }

    k0, k1 = bracket
    t0, t1 = float(peak_times[k0]), float(peak_times[k1])
    A_t0, A_t1 = float(peak_amps[k0]), float(peak_amps[k1])

    if np.isclose(t1, t0) or np.isclose(A_t1, A_t0):
        Thalf = np.nan
    else:
        frac = (A_half - A_t0) / (A_t1 - A_t0)
        Thalf = t0 + frac * (t1 - t0)

    return {
        "T_half": float(Thalf),
        "A0": A0,
        "A_half": A_half,
        "peak_times": peak_times,
        "peak_amps": peak_amps,
        "crossing_bracket": bracket,
        "reason": None
    }


# ============================================================
# 4) Simulation wrapper
# ============================================================

def run_simulation(params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = params["magnet"]
    coil = params["coil"]
    mech = params["mech"]
    elec = params["elec"]

    field_table = mag["field_table_file"]
    if not os.path.exists(field_table):
        sim.precompute_field_table(
            field_table,
            a=mag["diameter"] / 2,
            b=mag["length"] / 2,
            mu=mag["mu"]
        )

    tbl = sim.MagnetFieldTable(field_table)
    lookup = sim.FluxLinkageLookup(
        tbl,
        z_center=coil["z_center"],
        L_coil=coil["L_coil"],
        R_coil=coil["R_coil"],
        N_turns=coil["N_turns"]
    )

    y0 = [params["z0"], params["v0"]]
    sol, _z_eq = sim.solve_coupled_dynamics(
        (0.0, float(params["t_end"])),
        y0,
        lookup,
        mech,
        elec
    )

    t = sol.t
    z = sol.y[0]
    v = sol.y[1]
    return t, z, v


# ============================================================
# 5) Plot helpers
# ============================================================

def plot_single_run_with_peaks_and_halfdecay(t: np.ndarray, z: np.ndarray, hd: Dict[str, object]) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(t, z, label="z(t)")
    plt.plot(t, np.abs(z), alpha=0.5, label="|z(t)|")

    pt = hd["peak_times"]
    pa = hd["peak_amps"]
    if pt.size:
        plt.scatter(pt, pa, marker="x", label="peaks of |z|")

    if np.isfinite(hd["T_half"]):
        plt.axvline(hd["T_half"], linestyle="--", label=f"T_half ≈ {hd['T_half']:.3f} s")
        plt.axhline(hd["A_half"], linestyle=":", label=f"A_half = {hd['A_half']:.4g} m")

    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("z [m]")
    plt.title("Theoretical oscillation + peaks + half-decay time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sweep(x: np.ndarray, y: np.ndarray, *, xlabel: str, title: str = "Half-decay time sweep") -> None:
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x, y, marker="o")
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel("T_half [s]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Sweep with loading bar
# ============================================================

def sweep_one_parameter(
    base_params: dict,
    path: Tuple[str, ...],
    values: np.ndarray,
    *,
    peak_settings: Optional[dict] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep one parameter over 'values'. Returns (values, T_half_array).

    Uses ONE persistent tqdm bar (no "new bar per update" spam in IDE consoles).
    """
    values = np.asarray(values, float)
    Th = np.full(values.shape, np.nan, dtype=float)

    # --- create ONE progress bar ---
    bar = None
    if show_progress and tqdm is not None:
        bar = tqdm(
            total=values.size,
            desc=f"Sweep {'.'.join(path)}",
            unit="run",
            position=0,          # keep single bar
            leave=True,          # keep it after finish
            dynamic_ncols=True,  # adapt width
            file=sys.stdout,     # helps in some IDEs
        )
    elif show_progress and tqdm is None:
        print(f"Sweep {'.'.join(path)}: {values.size} runs...")

    for i, val in enumerate(values):
        p = copy.deepcopy(base_params)
        set_param_by_path(p, path, float(val))
        if peak_settings is not None:
            p["peaks"].update(peak_settings)

        t, z, _v = run_simulation(p)

        pk = p["peaks"]
        peak_times, peak_amps = _find_peaks_abs_with_endpoints(
            t, z,
            min_prominence=pk["min_prominence"],
            min_distance_s=pk["min_distance_s"],
            endpoint_eps=pk["endpoint_eps"]
        )
        hd = half_decay_time_from_peak_sequence(peak_times, peak_amps)
        Th[i] = hd["T_half"]

        # update progress bar
        if bar is not None:
            bar.set_postfix_str(f"{path[-1]}={val:.4g}, T_half={Th[i]:.4g}")
            bar.update(1)
        elif show_progress and tqdm is None:
            # fallback coarse prints (~10 updates)
            step = max(1, values.size // 10)
            if (i + 1) % step == 0 or (i + 1) == values.size:
                print(f"  {i + 1}/{values.size} done")

    if bar is not None:
        bar.close()

    return values, Th


# ============================================================
# 7) MAIN: choose what to sweep
# ============================================================

def main() -> None:
    # ------------------------------------------------------------
    # Choose ONE parameter to sweep: path = ("mech","c") etc.
    # Examples:
    #   ("mech","c")          damping
    #   ("mech","k")          spring constant
    #   ("elec","R")          load resistance
    #   ("coil","z_center")   coil position
    # ------------------------------------------------------------
    sweep_path = ("coil", "R_coil")

    # Choose sweep values
    sweep_values = np.linspace(0.0356/2, 0.0606/2, 50)

    # Optional: override peak detection during sweep (often stabilizes the curve)
    peak_override = {
        # try small nonzero if you get many micro-peaks
        "min_prominence": 0.0002,
        # set if needed, depends on your period scale
        "min_distance_s": None,
        "endpoint_eps": 1e-15,
    }

    # ---- Run sweep (with loading bar) ----
    x, Th = sweep_one_parameter(
        PARAMS,
        sweep_path,
        sweep_values,
        peak_settings=peak_override,
        show_progress=True
    )

    # ============================================================
    # SAVE SWEEP DATA (ADD THIS BLOCK HERE)
    # ============================================================

    param_name = "_".join(sweep_path)  # e.g. coil_z_center
    param_label = ".".join(sweep_path)  # e.g. coil.z_center

    output_file = f"sweep_{param_name}.csv"

    data = np.column_stack((x, Th))
    header = f"{param_label},T_half"

    np.savetxt(
        output_file,
        data,
        delimiter=",",
        header=header,
        comments=""
    )

    print(f"Sweep data saved to {output_file}")

    # ---- Plot sweep ----
    if PARAMS["plot_sweep"]:
        xlabel = ".".join(sweep_path)
        plot_sweep(x, Th, xlabel=xlabel, title=f"T_half vs {xlabel}")

    # ---- Optional: plot one representative run (middle of sweep) ----
    if PARAMS["plot_single_run"]:
        mid = float(sweep_values[len(sweep_values) // 2])
        p_mid = copy.deepcopy(PARAMS)
        set_param_by_path(p_mid, sweep_path, mid)

        t, z, _v = run_simulation(p_mid)
        pk = p_mid["peaks"]
        peak_times, peak_amps = _find_peaks_abs_with_endpoints(
            t, z,
            min_prominence=pk["min_prominence"],
            min_distance_s=pk["min_distance_s"],
            endpoint_eps=pk["endpoint_eps"]
        )
        hd = half_decay_time_from_peak_sequence(peak_times, peak_amps)

        print("\n=== Representative run ===")
        print(f"{'.'.join(sweep_path)} = {mid:.6g}")
        print(f"A0     = {hd['A0']:.6g} m")
        print(f"A_half = {hd['A_half']:.6g} m")
        print(f"T_half = {hd['T_half']:.6g} s")
        if hd["reason"]:
            print(f"Note: {hd['reason']}")

        plot_single_run_with_peaks_and_halfdecay(t, z, hd)


if __name__ == "__main__":
    main()

