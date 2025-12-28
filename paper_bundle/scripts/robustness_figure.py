from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.workflow.lib.paper_paths import (  # noqa: E402
    BUNDLE_ROOT,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    INTERMEDIATE_DIR,
    TABLES_DIR,
)
from paper.workflow.lib import modeling_framework_paper as mf_paper  # noqa: E402
from paper.workflow.plot import paper_fig2_temperature_amplifies_bvocs as fig2  # noqa: E402
from paper.workflow.plot import paper_fig3_mechanism_validation as fig3  # noqa: E402
from src.workflow import core as workflow_core  # noqa: E402
from src.workflow.config import default_config  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    run_id: int
    label: str
    factor: str
    signal_q: float
    window_minutes: int
    d_shift: int
    cs_dmax_shift: int
    kernel_form: str


def _resample_numeric_with_interpolation(df: pd.DataFrame, rule: str, limit: int) -> pd.DataFrame:
    resampled = df.resample(rule).mean()
    for col in resampled.columns:
        resampled[col] = resampled[col].interpolate(limit=limit, limit_direction="both")
    return resampled


def _align_hour_min_prediction(pred_series: pd.Series, time_index: pd.DatetimeIndex) -> pd.Series:
    if pred_series is None or time_index.empty:
        return pd.Series(index=time_index, dtype=float)
    pred_map = pred_series.groupby(pred_series.index.strftime("%H:%M")).mean()
    hour_min = pd.Series(time_index.strftime("%H:%M"), index=time_index)
    return hour_min.map(pred_map)


def _select_nearest_number_bins(
    number_cols: List[str],
    target_um: Tuple[float, float],
    shift_bins: int = 0,
) -> Tuple[str, str, np.ndarray, int, int]:
    diam_um = np.array([workflow_core._column_to_um(col) for col in number_cols], dtype=float)
    order = np.argsort(diam_um)
    diam_um_sorted = diam_um[order]
    cols_sorted = [number_cols[idx] for idx in order]
    idx1 = int(np.argmin(np.abs(diam_um_sorted - target_um[0])))
    idx2 = int(np.argmin(np.abs(diam_um_sorted - target_um[1])))
    idx1 = int(np.clip(idx1 + shift_bins, 0, len(cols_sorted) - 1))
    idx2 = int(np.clip(idx2 + shift_bins, 0, len(cols_sorted) - 1))
    if idx1 == idx2:
        if idx2 < len(cols_sorted) - 1:
            idx2 += 1
        elif idx1 > 0:
            idx1 -= 1
    return cols_sorted[idx1], cols_sorted[idx2], diam_um_sorted, idx1, idx2


def _estimate_bin_width_nm(diam_um_sorted: np.ndarray, idx: int) -> float:
    if diam_um_sorted.size < 2:
        return float("nan")
    if idx >= diam_um_sorted.size - 1:
        width_um = diam_um_sorted[idx] - diam_um_sorted[idx - 1]
    else:
        width_um = diam_um_sorted[idx + 1] - diam_um_sorted[idx]
    return max(float(width_um), 1e-6) * 1000.0


def _compute_cs_with_bounds(
    df: pd.DataFrame,
    cfg,
    dmax_um: float | None,
) -> pd.Series:
    number_cols = [c for c in df.columns if c.startswith("C") and c.endswith("um")]
    if not number_cols:
        raise ValueError("No number concentration columns found for CS.")
    if dmax_um is not None:
        number_cols = [
            c for c in number_cols if workflow_core._column_to_um(c) <= dmax_um
        ]
    # References: Fuchs and Sutugin (1971).
    # Equation: CS = sum_i 4*pi*D_v*r_i*F(Kn_i)*N_i.
    # Parameters: D_v vapor diffusivity (m^2 s^-1), r_i particle radius (m),
    # F(Kn_i) slip correction, N_i number concentration (m^-3).
    cs = workflow_core.compute_condensation_sink(
        df[number_cols],
        df["temperature_c"],
        cfg.cs_diffusivity_m2_s,
        cfg.mean_free_path_nm,
        cfg.accommodation_coefficient,
    )
    return cs.replace([np.inf, -np.inf], np.nan)


def _median_implied_eta(survival: pd.Series, ratio: pd.Series) -> float:
    implied_eta = ((1.0 / survival) - 1.0) / ratio
    implied_eta = implied_eta.replace([np.inf, -np.inf], np.nan)
    implied_eta = implied_eta[(implied_eta > 0) & implied_eta.notna()]
    if implied_eta.empty:
        return float("nan")
    return float(implied_eta.median())


def _compute_empirical_survival(
    output: pd.DataFrame,
    size_data: Dict[str, pd.DataFrame],
    cfg,
    *,
    d1_um: float,
    d2_um: float,
    shift_bins: int,
    window_minutes: int,
    signal_q: float,
    gap_minutes: float,
    smooth_steps: int,
    max_lag_min: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    dt_seconds = float(pd.to_timedelta(cfg.resample_rule).total_seconds())
    window_steps = max(3, int(round(window_minutes * 60.0 / dt_seconds)))
    min_steps = max(3, window_steps // 2)
    frames: List[pd.DataFrame] = []
    meta: Dict[str, float] = {}
    places = sorted(output["place"].dropna().unique())
    for place in places:
        out_place = output[output["place"] == place]
        size_df = size_data.get(place)
        if size_df is None or size_df.empty:
            continue
        _, number_cols = workflow_core._split_size_columns(size_df.columns)
        if not number_cols:
            continue
        col1, col2, diam_um_sorted, idx1, idx2 = _select_nearest_number_bins(
            number_cols, (d1_um, d2_um), shift_bins=shift_bins
        )
        d1_used = float(workflow_core._column_to_um(col1))
        d2_used = float(workflow_core._column_to_um(col2))
        delta1_nm = _estimate_bin_width_nm(diam_um_sorted, idx1)
        delta2_nm = _estimate_bin_width_nm(diam_um_sorted, idx2)
        if not np.isfinite(delta1_nm) or not np.isfinite(delta2_nm):
            continue

        size_sub = size_df[[col1, col2]].copy()
        out_sub = out_place[["CS_star", "G_abs"]].copy()
        size_sub = _resample_numeric_with_interpolation(
            size_sub, rule=cfg.resample_rule, limit=cfg.short_gap_limit
        )
        out_sub = _resample_numeric_with_interpolation(
            out_sub, rule=cfg.resample_rule, limit=cfg.short_gap_limit
        )
        joined = size_sub.join(out_sub, how="inner").dropna()
        if joined.empty:
            continue
        # References: time series segmentation to avoid cross-gap rolling.
        # Equation: seg_id = cumsum(delta_t > gap_minutes).
        # Parameters: gap_minutes controls segment breaks, window_steps is rolling window length.
        if gap_minutes > 0:
            seg_id = (
                joined.index.to_series()
                .diff()
                .gt(pd.Timedelta(minutes=gap_minutes))
                .cumsum()
            )
        else:
            seg_id = pd.Series(0, index=joined.index)

        for _, seg in joined.groupby(seg_id):
            if seg.shape[0] < window_steps:
                continue

            n1 = pd.to_numeric(seg[col1], errors="coerce")
            n2 = pd.to_numeric(seg[col2], errors="coerce")
            if smooth_steps > 1:
                n1 = n1.rolling(smooth_steps, center=True, min_periods=1).mean()
                n2 = n2.rolling(smooth_steps, center=True, min_periods=1).mean()
            dn1_dt = n1.diff() / dt_seconds
            dn2_dt = n2.diff() / dt_seconds
            g_abs = pd.to_numeric(seg["G_abs"], errors="coerce").abs().clip(lower=1e-6)
            cs_star = pd.to_numeric(seg["CS_star"], errors="coerce").clip(lower=0.0)

            # References: Kerminen and Kulmala (2002); Kulmala et al. (2007).
            # Equation: J(D,t) = dN_dt + K_loss*N + (G/delta_D)*N.
            # Parameters: J formation rate, N number concentration, K_loss uses CS_star (s^-1),
            # G growth rate (nm/s), delta_D bin width (nm).
            j1 = (dn1_dt + cs_star * n1 + (g_abs / delta1_nm) * n1).clip(lower=0.0)
            j2 = (dn2_dt + cs_star * n2 + (g_abs / delta2_nm) * n2).clip(lower=0.0)

            # References: Kerminen and Kulmala (2002); Kulmala et al. (2007).
            # Equation: tau = (D2 - D1) / G_med, J2_aligned(t) = J2(t + tau).
            # Parameters: D1 and D2 diameters (nm), G_med median growth rate (nm/s), tau lag (s).
            d_nm = (d2_used - d1_used) * 1000.0
            g_med = float(g_abs.median())
            tau_seconds = d_nm / max(g_med, 1e-6)
            max_shift_steps = int(round(max_lag_min * 60.0 / dt_seconds))
            shift_steps = int(round(tau_seconds / dt_seconds))
            shift_steps = min(max(shift_steps, 0), max_shift_steps)
            if shift_steps > 0:
                j2 = j2.shift(-shift_steps)

            # References: size-space continuity survival estimate.
            # Equation: S_surv_obs = sum_window(J2_aligned) / sum_window(J1).
            # Parameters: window_minutes sets the rolling integration window.
            roll_j1 = j1.rolling(window_steps, min_periods=min_steps).sum()
            roll_j2 = j2.rolling(window_steps, min_periods=min_steps).sum()
            s_obs = (roll_j2 / roll_j1).replace([np.inf, -np.inf], np.nan)
            s_obs = s_obs.clip(lower=0.0, upper=1.0)
            ratio = (cs_star / g_abs).replace([np.inf, -np.inf], np.nan)
            ratio_window = ratio.rolling(window_steps, min_periods=min_steps).median()

            # References: robust event filtering for noisy formation rates.
            # Equation: keep samples where sum_window(J1) > quantile(signal_q).
            # Parameters: signal_q controls retained high-signal fraction.
            mask_signal = roll_j1 > roll_j1.quantile(signal_q)
            df_obs = pd.DataFrame({"S_surv_obs": s_obs, "ratio": ratio_window}).dropna()
            if not df_obs.empty:
                df_obs = df_obs[mask_signal.loc[df_obs.index]]
            if df_obs.empty:
                continue
            df_obs = df_obs[df_obs["ratio"] > 0]
            df_obs["place"] = place
            frames.append(df_obs)
            meta["d1_um"] = d1_used
            meta["d2_um"] = d2_used

    if not frames:
        return pd.DataFrame(), meta
    combined = pd.concat(frames, axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    return combined, meta


def _fit_kernel_mm(cs: pd.Series, ratio: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    return mf_paper._fit_cs_scaling(cs, ratio)


def _fit_kernel_hill(cs: pd.Series, ratio: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    cs_clean = cs.replace([np.inf, -np.inf], np.nan)
    mask = cs_clean.notna() & ratio.notna() & (cs_clean > 0) & (ratio > 0)
    x = cs_clean[mask].to_numpy()
    y = ratio[mask].to_numpy()
    if x.size < 5:
        return np.array([np.nan, np.nan, np.nan]), np.full((3, 3), np.nan)

    def hill_func(c, bmax, cs0, n):
        return bmax * (c**n) / (c**n + cs0**n)

    p0 = [np.nanpercentile(y, 90), np.nanmedian(x), 1.5]
    bounds = ([0.0, 0.0, 0.5], [np.inf, np.inf, 4.0])
    try:
        popt, pcov = curve_fit(hill_func, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        popt, pcov = [np.nan, np.nan, np.nan], np.full((3, 3), np.nan)
    return np.asarray(popt, dtype=float), np.asarray(pcov, dtype=float)


def _kernel_eval(cs: np.ndarray, params: Dict[str, float], form: str) -> np.ndarray:
    cs = np.asarray(cs, dtype=float)
    cs = np.clip(cs, 0.0, np.inf)
    if form == "hill":
        n_val = float(params.get("n", 1.0))
        return params["beta_max"] * (cs**n_val) / (cs**n_val + params["cs0"] ** n_val)
    return params["beta_max"] * cs / (cs + params["cs0"])


def _block_bootstrap_sample(df: pd.DataFrame, block_steps: int, rng: np.random.Generator) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_index()
    n = df.shape[0]
    if block_steps <= 1 or n <= block_steps:
        idx = rng.integers(0, n, size=n)
        return df.iloc[idx]
    max_start = max(n - block_steps, 1)
    starts = rng.integers(0, max_start, size=max(1, int(np.ceil(n / block_steps))))
    blocks = [df.iloc[s : s + block_steps] for s in starts]
    sample = pd.concat(blocks, axis=0).iloc[:n]
    return sample


def _bootstrap_eta_curves(
    surv_df: pd.DataFrame,
    chi_grid: np.ndarray,
    block_hours: float,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if surv_df.empty or n_boot <= 0:
        return np.array([]), np.empty((0, chi_grid.size))
    surv_df = surv_df.sort_index()
    dt_seconds = float(surv_df.index.to_series().diff().dropna().median().total_seconds())
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 60.0
    block_steps = max(2, int(round(block_hours * 3600.0 / dt_seconds)))
    rng = np.random.default_rng(seed)
    eta_vals: List[float] = []
    curves: List[np.ndarray] = []
    for _ in range(n_boot):
        sample = _block_bootstrap_sample(surv_df, block_steps, rng)
        eta_val = _median_implied_eta(sample["S_surv_obs"], sample["ratio"])
        eta_vals.append(eta_val)
        if np.isfinite(eta_val):
            curves.append(1.0 / (1.0 + eta_val * chi_grid))
        else:
            curves.append(np.full_like(chi_grid, np.nan))
    return np.asarray(eta_vals, dtype=float), np.vstack(curves)


def _bootstrap_kernel_curves(
    df: pd.DataFrame,
    cs_grid: np.ndarray,
    form: str,
    block_hours: float,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if df.empty or n_boot <= 0:
        return np.array([]), np.array([]), np.array([]), np.empty((0, cs_grid.size))
    df = df.sort_index()
    dt_seconds = float(df.index.to_series().diff().dropna().median().total_seconds())
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 60.0
    block_steps = max(2, int(round(block_hours * 3600.0 / dt_seconds)))
    rng = np.random.default_rng(seed)
    beta_vals: List[float] = []
    cs0_vals: List[float] = []
    n_vals: List[float] = []
    curves: List[np.ndarray] = []
    for _ in range(n_boot):
        sample = _block_bootstrap_sample(df, block_steps, rng)
        if form == "hill":
            popt, _ = _fit_kernel_hill(sample["cs"], sample["ratio"])
            beta_max, cs0, n_val = popt
        else:
            popt, _ = _fit_kernel_mm(sample["cs"], sample["ratio"])
            beta_max, cs0 = popt
            n_val = 1.0
        beta_vals.append(beta_max)
        cs0_vals.append(cs0)
        n_vals.append(n_val)
        params = {"beta_max": beta_max, "cs0": cs0, "n": n_val}
        curves.append(_kernel_eval(cs_grid, params, form=form))
    return np.asarray(beta_vals), np.asarray(cs0_vals), np.asarray(n_vals), np.vstack(curves)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 3:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _time_block_cv_metrics(
    df: pd.DataFrame,
    form: str,
    block_days: int,
) -> Dict[str, object]:
    if df.empty:
        return {
            "folds": [],
            "crps": float("nan"),
            "crps_low": float("nan"),
            "crps_high": float("nan"),
            "r2": float("nan"),
        }
    df = df.sort_index()
    block_label = df.index.floor(f"{block_days}D")
    metrics: List[Dict[str, float]] = []
    folds: List[Dict[str, float]] = []
    for block in block_label.unique():
        test = df[block_label == block]
        train = df[block_label != block]
        if train.shape[0] < 20 or test.shape[0] < 10:
            continue
        if form == "hill":
            popt, _ = _fit_kernel_hill(train["cs"], train["ratio"])
            beta_max, cs0, n_val = popt
        else:
            popt, _ = _fit_kernel_mm(train["cs"], train["ratio"])
            beta_max, cs0 = popt
            n_val = 1.0
        params = {"beta_max": float(beta_max), "cs0": float(cs0), "n": float(n_val)}
        pred_train = train["lin_pred"] * _kernel_eval(train["cs"].to_numpy(), params, form=form)
        pred_test = test["lin_pred"] * _kernel_eval(test["cs"].to_numpy(), params, form=form)
        resid_train = train["obs"] - pred_train
        sigma = float(np.nanstd(resid_train.to_numpy(), ddof=1))
        sigma = max(sigma, 1e-6)
        # References: Gneiting and Raftery (2007); Hersbach (2000).
        # Equation: CRPS(N(mu,sigma^2), y) for Gaussian predictive distribution.
        # Parameters: mu predictive mean, sigma predictive std, y observations.
        crps_vals = fig2._crps_normal_analytical(pred_test.to_numpy(), sigma, test["obs"].to_numpy())
        crps = float(np.nanmean(crps_vals))
        r2 = _r2_score(test["obs"].to_numpy(), pred_test.to_numpy())
        metrics.append({"crps": crps, "r2": r2})
        folds.append({"block": str(block), "crps": crps, "r2": r2})
    if not metrics:
        return {
            "folds": [],
            "crps": float("nan"),
            "crps_low": float("nan"),
            "crps_high": float("nan"),
            "r2": float("nan"),
        }
    crps_vals = np.array([m["crps"] for m in metrics], dtype=float)
    r2_vals = np.array([m["r2"] for m in metrics], dtype=float)
    return {
        "folds": folds,
        "crps": float(np.nanmean(crps_vals)),
        "crps_low": float(np.nanpercentile(crps_vals, 5)),
        "crps_high": float(np.nanpercentile(crps_vals, 95)),
        "r2": float(np.nanmean(r2_vals)),
    }


def build_run_specs() -> List[RunSpec]:
    base_signal_q = float(fig3.SURV_OBS_SIGNAL_Q)
    base_window_min = int(fig3.SURV_OBS_WINDOW_MIN)
    signal_low = max(0.05, base_signal_q * 0.8)
    signal_high = min(0.95, base_signal_q * 1.2)
    window_wide = int(round(base_window_min * 1.5))
    window_narrow = max(10, int(round(base_window_min * 0.5)))
    return [
        RunSpec(0, "Baseline", "baseline", base_signal_q, base_window_min, 0, 0, "mm"),
        RunSpec(1, "Burst threshold -20%", "burst_threshold", signal_low, base_window_min, 0, 0, "mm"),
        RunSpec(2, "Burst threshold +20%", "burst_threshold", signal_high, base_window_min, 0, 0, "mm"),
        RunSpec(3, "Window x1.5", "window_length", base_signal_q, window_wide, 0, 0, "mm"),
        RunSpec(4, "Window x0.5", "window_length", base_signal_q, window_narrow, 0, 0, "mm"),
        RunSpec(5, "Growth interval shift -1 bin", "growth_interval", base_signal_q, base_window_min, -1, 0, "mm"),
        RunSpec(6, "Growth interval shift +1 bin", "growth_interval", base_signal_q, base_window_min, 1, 0, "mm"),
        RunSpec(7, "CS Dmax -1 bin", "cs_definition", base_signal_q, base_window_min, 0, -1, "mm"),
        RunSpec(8, "CS Dmax +1 bin", "cs_definition", base_signal_q, base_window_min, 0, 1, "mm"),
        RunSpec(9, "Kernel MM", "kernel_form", base_signal_q, base_window_min, 0, 0, "mm"),
        RunSpec(10, "Kernel Hill", "kernel_form", base_signal_q, base_window_min, 0, 0, "hill"),
    ]


def _affects_mass(run: RunSpec) -> bool:
    if run.run_id == 9:
        return False
    return (run.run_id == 0) or (run.factor in {"cs_definition", "kernel_form"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness figure for survival gate and mass kernel.")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="Bootstrap replicates per run.")
    parser.add_argument("--boot-block-hours", type=float, default=6.0, help="Block length in hours for bootstrap.")
    parser.add_argument("--cv-block-days", type=int, default=14, help="Block length in days for CV.")
    parser.add_argument("--seed", type=int, default=202501, help="Random seed.")
    parser.add_argument("--figure-name", type=str, default="SI_Fig_Robustness_survival_mass_kernel")
    parser.add_argument("--table-name", type=str, default="Table_SI_Robustness_summary")
    args = parser.parse_args()

    mf_paper.set_plot_style()
    cfg = default_config()

    df_base = mf_paper.load_base_data(cfg)
    df_base = df_base.sort_index()
    df_growth = pd.read_parquet(CHECKPOINT_DIR / "step04_growth_metrics.parquet")
    df_growth.index = pd.to_datetime(df_growth.index)
    df_growth = df_growth.sort_index()

    _, cat1_outputs, _, _, _ = mf_paper.load_cached_results()
    pred_map = cat1_outputs.get("predictions", {})

    size_data: Dict[str, pd.DataFrame] = {}
    for place, path in cfg.size_distribution_csv_sites.items():
        path = Path(path)
        if not path.is_absolute():
            path = BUNDLE_ROOT / path
        if not path.exists():
            continue
        size_df = workflow_core.load_size_distribution(path)
        size_df = _resample_numeric_with_interpolation(size_df, cfg.resample_rule, cfg.short_gap_limit)
        size_data[place] = size_df

    number_cols = [c for c in df_base.columns if c.startswith("C") and c.endswith("um")]
    diam_sorted = np.array(sorted({workflow_core._column_to_um(c) for c in number_cols}), dtype=float)
    max_idx = len(diam_sorted) - 1 if diam_sorted.size else 0

    runs = build_run_specs()
    summary_rows: List[Dict[str, object]] = []
    surv_curves: List[np.ndarray] = []
    kenv_curves: List[np.ndarray] = []
    surv_boot_baseline: np.ndarray | None = None
    kenv_boot_baseline: np.ndarray | None = None
    chi_obs = np.array([], dtype=float)
    cs_obs = np.array([], dtype=float)
    pooled_params: Dict[int, Dict[str, float]] = {}
    pooled_param_ci: Dict[int, Dict[str, Tuple[float, float]]] = {}
    metrics_by_run: Dict[int, Dict[str, Dict[str, float]]] = {}

    chi_grid = None
    cs_grid = np.linspace(0.01, 1.0, 120)
    cs_grid_set = False

    for run in runs:
        dmax_idx = int(np.clip(max_idx + run.cs_dmax_shift, 0, max_idx))
        dmax_um = float(diam_sorted[dmax_idx]) if diam_sorted.size else float("nan")

        cs_series = _compute_cs_with_bounds(df_base, cfg, dmax_um=dmax_um)
        output = df_growth.copy()
        output["CS_star"] = cs_series.reindex(df_growth.index)

        surv_df, meta = _compute_empirical_survival(
            output,
            size_data,
            cfg,
            d1_um=float(fig3.SURV_OBS_D1_UM),
            d2_um=float(fig3.SURV_OBS_D2_UM),
            shift_bins=run.d_shift,
            window_minutes=run.window_minutes,
            signal_q=run.signal_q,
            gap_minutes=10.0,
            smooth_steps=int(fig3.SURV_OBS_SMOOTH_STEPS),
            max_lag_min=int(fig3.SURV_OBS_MAX_LAG_MIN),
        )
        if run.run_id == 0 and not surv_df.empty:
            chi_obs = surv_df["ratio"].to_numpy(dtype=float)
        eta_all = _median_implied_eta(surv_df["S_surv_obs"], surv_df["ratio"]) if not surv_df.empty else float("nan")

        if chi_grid is None:
            ratio_vals = surv_df["ratio"].to_numpy(dtype=float) if not surv_df.empty else np.array([1.0])
            ratio_vals = ratio_vals[np.isfinite(ratio_vals) & (ratio_vals > 0)]
            if ratio_vals.size < 5:
                chi_grid = np.logspace(-1, 3, 120)
            else:
                lo, hi = np.nanpercentile(ratio_vals, [5, 95])
                hi = max(hi, lo * 1.05)
                chi_grid = np.logspace(np.log10(lo), np.log10(hi), 120)

        surv_curve = 1.0 / (1.0 + eta_all * chi_grid) if np.isfinite(eta_all) else np.full_like(chi_grid, np.nan)
        surv_curves.append(surv_curve)

        eta_boot, surv_boot = _bootstrap_eta_curves(
            surv_df,
            chi_grid,
            args.boot_block_hours,
            args.n_bootstrap,
            args.seed + run.run_id,
        )
        if surv_boot.size and run.run_id == 0:
            surv_boot_baseline = surv_boot
        eta_ci = (float(np.nanpercentile(eta_boot, 5)), float(np.nanpercentile(eta_boot, 95))) if eta_boot.size else (float("nan"), float("nan"))

        kenv_rows: List[pd.DataFrame] = []
        place_params: Dict[str, Dict[str, float]] = {}
        place_ci: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for place in sorted(df_base["place"].dropna().unique()):
            sub = df_base[df_base["place"] == place]
            lin_pred = pred_map.get((place, "2"))
            if lin_pred is None:
                continue
            sub_cs = cs_series.reindex(sub.index)
            frame = pd.DataFrame(
                {
                    "Time": sub.index,
                    "obs": sub["SOA"].to_numpy(),
                    "lin_pred": _align_hour_min_prediction(lin_pred, sub.index).to_numpy(),
                    "cs": sub_cs.to_numpy(),
                }
            ).dropna()
            frame = frame.set_index("Time")
            frame = frame[(frame["obs"] > 0) & (frame["lin_pred"] > 0) & (frame["cs"] > 0)]
            frame["ratio"] = frame["obs"] / frame["lin_pred"]
            if frame.empty:
                continue
            kenv_rows.append(frame)
            if run.kernel_form == "hill":
                popt, _ = _fit_kernel_hill(frame["cs"], frame["ratio"])
                beta_max, cs0, n_val = popt
            else:
                popt, _ = _fit_kernel_mm(frame["cs"], frame["ratio"])
                beta_max, cs0 = popt
                n_val = 1.0
            place_params[place] = {"beta_max": float(beta_max), "cs0": float(cs0), "n": float(n_val)}

            beta_boot, cs0_boot, n_boot, _ = _bootstrap_kernel_curves(
                frame,
                cs_grid if cs_grid is not None else np.array([1.0]),
                run.kernel_form,
                args.boot_block_hours,
                args.n_bootstrap,
                args.seed + run.run_id + 1000,
            )
            if beta_boot.size:
                place_ci[place] = {
                    "beta_max": (float(np.nanpercentile(beta_boot, 5)), float(np.nanpercentile(beta_boot, 95))),
                    "cs0": (float(np.nanpercentile(cs0_boot, 5)), float(np.nanpercentile(cs0_boot, 95))),
                    "n": (float(np.nanpercentile(n_boot, 5)), float(np.nanpercentile(n_boot, 95))),
                }
            else:
                place_ci[place] = {
                    "beta_max": (float("nan"), float("nan")),
                    "cs0": (float("nan"), float("nan")),
                    "n": (float("nan"), float("nan")),
                }

            cv_metrics = _time_block_cv_metrics(frame, run.kernel_form, args.cv_block_days)
            metrics_by_run.setdefault(run.run_id, {})[place] = cv_metrics

        pooled = pd.concat(kenv_rows, axis=0) if kenv_rows else pd.DataFrame()
        if run.run_id == 0 and not pooled.empty:
            cs_obs = pooled["cs"].to_numpy(dtype=float)
        if not pooled.empty and not cs_grid_set:
            cs_vals = pooled["cs"].to_numpy(dtype=float)
            cs_vals = cs_vals[np.isfinite(cs_vals) & (cs_vals > 0)]
            if cs_vals.size >= 5:
                lo, hi = np.nanpercentile(cs_vals, [5, 95])
                hi = max(hi, lo * 1.05)
                cs_grid = np.linspace(lo, hi, 120)
            cs_grid_set = True

        if pooled.empty:
            pooled_params[run.run_id] = {"beta_max": float("nan"), "cs0": float("nan"), "n": float("nan")}
            pooled_param_ci[run.run_id] = {
                "beta_max": (float("nan"), float("nan")),
                "cs0": (float("nan"), float("nan")),
                "n": (float("nan"), float("nan")),
            }
            kenv_curve = np.full_like(cs_grid, np.nan)
        else:
            if run.kernel_form == "hill":
                popt, _ = _fit_kernel_hill(pooled["cs"], pooled["ratio"])
                beta_max, cs0, n_val = popt
            else:
                popt, _ = _fit_kernel_mm(pooled["cs"], pooled["ratio"])
                beta_max, cs0 = popt
                n_val = 1.0
            pooled_params[run.run_id] = {"beta_max": float(beta_max), "cs0": float(cs0), "n": float(n_val)}
            beta_boot, cs0_boot, n_boot, kenv_boot = _bootstrap_kernel_curves(
                pooled,
                cs_grid,
                run.kernel_form,
                args.boot_block_hours,
                args.n_bootstrap,
                args.seed + run.run_id + 2000,
            )
            if kenv_boot.size and run.run_id == 0:
                kenv_boot_baseline = kenv_boot
            pooled_param_ci[run.run_id] = {
                "beta_max": (float(np.nanpercentile(beta_boot, 5)), float(np.nanpercentile(beta_boot, 95))),
                "cs0": (float(np.nanpercentile(cs0_boot, 5)), float(np.nanpercentile(cs0_boot, 95))),
                "n": (float(np.nanpercentile(n_boot, 5)), float(np.nanpercentile(n_boot, 95))),
            }
            params = {"beta_max": float(beta_max), "cs0": float(cs0), "n": float(n_val)}
            kenv_curve = _kernel_eval(cs_grid, params, form=run.kernel_form)
        kenv_curves.append(kenv_curve)

        places = sorted(df_base["place"].dropna().unique())
        for place in places:
            eta_place = float("nan")
            eta_ci_place = (float("nan"), float("nan"))
            if not surv_df.empty:
                sub = surv_df[surv_df["place"] == place]
                if not sub.empty:
                    eta_place = _median_implied_eta(sub["S_surv_obs"], sub["ratio"])
                    eta_boot_place, _ = _bootstrap_eta_curves(
                        sub,
                        chi_grid,
                        args.boot_block_hours,
                        args.n_bootstrap,
                        args.seed + run.run_id + 3000,
                    )
                    if eta_boot_place.size:
                        eta_ci_place = (
                            float(np.nanpercentile(eta_boot_place, 5)),
                            float(np.nanpercentile(eta_boot_place, 95)),
                        )
            params_place = place_params.get(place, {"beta_max": float("nan"), "cs0": float("nan"), "n": float("nan")})
            ci_place = place_ci.get(place, {"beta_max": (float("nan"), float("nan")), "cs0": (float("nan"), float("nan")), "n": (float("nan"), float("nan"))})
            metric_place = metrics_by_run.get(run.run_id, {}).get(place, {"crps": float("nan"), "crps_low": float("nan"), "crps_high": float("nan"), "r2": float("nan")})
            summary_rows.append(
                {
                    "run_id": run.run_id,
                    "run_label": run.label,
                    "factor": run.factor,
                    "place": place,
                    "signal_q": run.signal_q,
                    "window_minutes": run.window_minutes,
                    "d_shift": run.d_shift,
                    "cs_dmax_um": dmax_um,
                    "kernel_form": run.kernel_form,
                    "d1_um": meta.get("d1_um", float("nan")),
                    "d2_um": meta.get("d2_um", float("nan")),
                    "eta": eta_place,
                    "eta_ci_low": eta_ci_place[0],
                    "eta_ci_high": eta_ci_place[1],
                    "beta_max": params_place.get("beta_max"),
                    "beta_max_ci_low": ci_place["beta_max"][0],
                    "beta_max_ci_high": ci_place["beta_max"][1],
                    "cs0": params_place.get("cs0"),
                    "cs0_ci_low": ci_place["cs0"][0],
                    "cs0_ci_high": ci_place["cs0"][1],
                    "hill_n": params_place.get("n"),
                    "hill_n_ci_low": ci_place["n"][0],
                    "hill_n_ci_high": ci_place["n"][1],
                    "crps": metric_place.get("crps"),
                    "crps_low": metric_place.get("crps_low"),
                    "crps_high": metric_place.get("crps_high"),
                    "r2": metric_place.get("r2"),
                }
            )

        pooled_param_ci[run.run_id]["eta"] = eta_ci
        pooled_params[run.run_id]["eta"] = eta_all

    baseline_metrics = metrics_by_run.get(0, {})
    baseline_fold_map: Dict[str, Dict[str, float]] = {}
    for place, metrics in baseline_metrics.items():
        fold_map: Dict[str, float] = {}
        for fold in metrics.get("folds", []):
            crps_val = fold.get("crps")
            if np.isfinite(crps_val) and crps_val > 0:
                fold_map[fold.get("block")] = float(crps_val)
        baseline_fold_map[place] = fold_map

    for row in summary_rows:
        place = row["place"]
        run_id = row["run_id"]
        run_metrics = metrics_by_run.get(run_id, {}).get(place, {})
        fold_map = baseline_fold_map.get(place, {})
        ratios: List[float] = []
        for fold in run_metrics.get("folds", []):
            block = fold.get("block")
            crps_run = fold.get("crps")
            crps_base = fold_map.get(block)
            if block is None or not np.isfinite(crps_run) or not np.isfinite(crps_base) or crps_base <= 0:
                continue
            ratios.append(float(crps_run) / float(crps_base))
        if ratios:
            ratio_vals = np.array(ratios, dtype=float)
            row["crps_ratio"] = float(np.nanmean(ratio_vals))
            row["crps_ratio_low"] = float(np.nanpercentile(ratio_vals, 5))
            row["crps_ratio_high"] = float(np.nanpercentile(ratio_vals, 95))
            crpss_vals = 1.0 - ratio_vals
            row["crpss"] = float(np.nanmean(crpss_vals))
            row["crpss_low"] = float(np.nanpercentile(crpss_vals, 5))
            row["crpss_high"] = float(np.nanpercentile(crpss_vals, 95))
        else:
            row["crpss"] = float("nan")
            row["crpss_low"] = float("nan")
            row["crpss_high"] = float("nan")
            row["crps_ratio"] = float("nan")
            row["crps_ratio_low"] = float("nan")
            row["crps_ratio_high"] = float("nan")

    summary_df = pd.DataFrame(summary_rows)
    mf_paper.save_table(summary_df, args.table_name, "Robustness")

    surv_curves_arr = np.vstack(surv_curves)
    surv_oat_lo = np.nanpercentile(surv_curves_arr, 5, axis=0)
    surv_oat_hi = np.nanpercentile(surv_curves_arr, 95, axis=0)
    if surv_boot_baseline is not None and surv_boot_baseline.size:
        surv_boot_lo = np.nanpercentile(surv_boot_baseline, 5, axis=0)
        surv_boot_hi = np.nanpercentile(surv_boot_baseline, 95, axis=0)
    else:
        surv_boot_lo = np.full_like(chi_grid, np.nan)
        surv_boot_hi = np.full_like(chi_grid, np.nan)

    kenv_curves_arr = np.vstack(kenv_curves)
    kenv_oat_lo = np.nanpercentile(kenv_curves_arr, 5, axis=0)
    kenv_oat_hi = np.nanpercentile(kenv_curves_arr, 95, axis=0)
    if kenv_boot_baseline is not None and kenv_boot_baseline.size:
        kenv_boot_lo = np.nanpercentile(kenv_boot_baseline, 5, axis=0)
        kenv_boot_hi = np.nanpercentile(kenv_boot_baseline, 95, axis=0)
    else:
        kenv_boot_lo = np.full_like(cs_grid, np.nan)
        kenv_boot_hi = np.full_like(cs_grid, np.nan)

    chi_obs = chi_obs[np.isfinite(chi_obs) & (chi_obs > 0)]
    cs_obs = cs_obs[np.isfinite(cs_obs) & (cs_obs > 0)]
    chi_p5 = float(np.nanpercentile(chi_obs, 5)) if chi_obs.size else float("nan")
    chi_p95 = float(np.nanpercentile(chi_obs, 95)) if chi_obs.size else float("nan")
    cs_p5 = float(np.nanpercentile(cs_obs, 5)) if cs_obs.size else float("nan")
    cs_p95 = float(np.nanpercentile(cs_obs, 95)) if cs_obs.size else float("nan")

    surv_curve_df = pd.DataFrame(surv_curves_arr, columns=chi_grid)
    surv_curve_df.insert(0, "run_id", [r.run_id for r in runs])
    surv_curve_df.to_parquet(INTERMEDIATE_DIR / "robustness_survival_curves.parquet", index=False)

    kenv_curve_df = pd.DataFrame(kenv_curves_arr, columns=cs_grid)
    kenv_curve_df.insert(0, "run_id", [r.run_id for r in runs])
    kenv_curve_df.to_parquet(INTERMEDIATE_DIR / "robustness_kenv_curves.parquet", index=False)

    fig = plt.figure(figsize=(15.6, 8.4))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.25, 1.25, 1.05, 1.05],
        hspace=0.42,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_c1 = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c2 = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[1, 3])
    rng = np.random.default_rng(args.seed + 123)
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    note_font = "Helvetica" if "Helvetica" in available_fonts else "Arial"

    base_params = pooled_params.get(0, {})
    base_eta = base_params.get("eta", float("nan"))
    base_beta = base_params.get("beta_max", float("nan"))
    base_cs0 = base_params.get("cs0", float("nan"))
    label_map = {r.run_id: r.label for r in runs}
    short_label_map = {
        0: "Baseline",
        1: "Burst -20%",
        2: "Burst +20%",
        3: "Window x1.5",
        4: "Window x0.5",
        5: "Growth -1 bin",
        6: "Growth +1 bin",
        7: "CS Dmax -1",
        8: "CS Dmax +1",
        9: "Kernel MM",
        10: "Kernel Hill",
    }
    eta_ci = pooled_param_ci.get(0, {}).get("eta", (float("nan"), float("nan")))
    beta_ci = pooled_param_ci.get(0, {}).get("beta_max", (float("nan"), float("nan")))
    cs0_ci = pooled_param_ci.get(0, {}).get("cs0", (float("nan"), float("nan")))

    def _fmt_val(val: float) -> str:
        return f"{val:.3g}" if np.isfinite(val) else "NA"

    def _fmt_sci_tex(val: float) -> str:
        if not np.isfinite(val):
            return "NA"
        if val == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(val))))
        mant = val / (10 ** exp)
        return f"{mant:.2f}\\times 10^{{{exp}}}"

    def _fmt_ci(ci: Tuple[float, float]) -> str:
        if not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
            return "NA"
        return f"{_fmt_val(ci[0])}-{_fmt_val(ci[1])}"

    def _fmt_ci_sci_tex(ci: Tuple[float, float]) -> str:
        if not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
            return "NA"
        return f"{_fmt_sci_tex(ci[0])}-{_fmt_sci_tex(ci[1])}"

    # Panel A: survival gate
    if np.isfinite(surv_boot_lo).any():
        ax_a.fill_between(chi_grid, surv_boot_lo, surv_boot_hi, color="#f0f0f0", alpha=1.0, label="Baseline bootstrap (5-95%)")
    ax_a.fill_between(chi_grid, surv_oat_lo, surv_oat_hi, color="#bdbdbd", alpha=0.8, label="OAT envelope (5-95%)")
    ax_a.plot(chi_grid, surv_curves_arr[0], color="#000000", linewidth=2.3, label="Baseline")
    if np.isfinite(chi_p5) and chi_p5 > 0:
        ax_a.axvline(chi_p5, color="#888888", linestyle=":", linewidth=0.9)
    if np.isfinite(chi_p95) and chi_p95 > 0:
        ax_a.axvline(chi_p95, color="#888888", linestyle=":", linewidth=0.9)
    if chi_obs.size:
        chi_xlo = float(np.nanpercentile(chi_obs, 1))
        chi_xhi = float(np.nanpercentile(chi_obs, 99))
        chi_floor = float(np.nanmin(chi_grid[chi_grid > 0])) if np.any(chi_grid > 0) else 1e-6
        chi_xlo = max(chi_xlo, chi_floor)
        if np.isfinite(chi_xlo) and np.isfinite(chi_xhi) and chi_xhi > chi_xlo:
            ax_a.set_xlim(chi_xlo, chi_xhi)
    if chi_obs.size:
        rug_n = min(1500, chi_obs.size)
        rug_idx = rng.choice(chi_obs.size, size=rug_n, replace=False)
        rug_x = np.sort(chi_obs[rug_idx])
        if ax_a.get_xlim()[0] > 0:
            rug_x = rug_x[(rug_x >= ax_a.get_xlim()[0]) & (rug_x <= ax_a.get_xlim()[1])]
        ax_a.vlines(
            rug_x,
            0.0,
            0.025,
            transform=ax_a.get_xaxis_transform(),
            color="#CFCFCF",
            linewidth=0.45,
            alpha=0.45,
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"$\chi = CS^*/|G|$ (nm$^{-1}$)")
    ax_a.set_ylabel(r"$S_{\rm surv}$")
    ax_a.set_title("Survival gate robustness", loc="left", fontweight="bold")
    ax_a.text(
        0.02,
        0.02,
        rf"$N={chi_obs.size};\ \eta_0={_fmt_sci_tex(base_eta)}\ (5-95\%:\ {_fmt_ci_sci_tex(eta_ci)})$",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#000000",
        fontfamily=note_font,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        clip_on=True,
    )
    ax_a.legend(frameon=False, fontsize=8)

    # Panel B: mass kernel
    if np.isfinite(kenv_boot_lo).any():
        ax_b.fill_between(cs_grid, kenv_boot_lo, kenv_boot_hi, color="#f0f0f0", alpha=1.0, label="Baseline bootstrap (5-95%)")
    ax_b.fill_between(cs_grid, kenv_oat_lo, kenv_oat_hi, color="#bdbdbd", alpha=0.8, label="OAT envelope (5-95%)")
    ax_b.plot(cs_grid, kenv_curves_arr[0], color="#000000", linewidth=2.3, label="Baseline")
    if np.isfinite(cs_p5) and cs_p5 > 0:
        ax_b.axvline(cs_p5, color="#888888", linestyle=":", linewidth=0.9)
    if np.isfinite(cs_p95) and cs_p95 > 0:
        ax_b.axvline(cs_p95, color="#888888", linestyle=":", linewidth=0.9)
    if cs_obs.size:
        cs_xlo = float(np.nanpercentile(cs_obs, 5))
        cs_xhi = float(np.nanpercentile(cs_obs, 95))
        cs_xlo = max(cs_xlo, 0.0)
        pad = 0.1 * (cs_xhi - cs_xlo) if np.isfinite(cs_xhi - cs_xlo) else 0.0
        cs_xlo = max(cs_xlo - pad, 0.0)
        cs_xhi = cs_xhi + pad
        if np.isfinite(cs_xlo) and np.isfinite(cs_xhi) and cs_xhi > cs_xlo:
            ax_b.set_xlim(cs_xlo, cs_xhi)
    if cs_obs.size:
        rug_n = min(1500, cs_obs.size)
        rug_idx = rng.choice(cs_obs.size, size=rug_n, replace=False)
        rug_x = np.sort(cs_obs[rug_idx])
        if ax_b.get_xlim()[1] > ax_b.get_xlim()[0]:
            rug_x = rug_x[(rug_x >= ax_b.get_xlim()[0]) & (rug_x <= ax_b.get_xlim()[1])]
        ax_b.vlines(
            rug_x,
            0.0,
            0.025,
            transform=ax_b.get_xaxis_transform(),
            color="#CFCFCF",
            linewidth=0.45,
            alpha=0.45,
        )
    ax_b.set_xlabel(r"$CS$ (s$^{-1}$)")
    ax_b.set_ylabel(r"$k_{\rm env}$")
    ax_b.set_title("Mass kernel robustness", loc="left", fontweight="bold")
    ax_b.text(
        0.98,
        0.02,
        rf"$N={cs_obs.size};\ \beta_{{\max,0}}={_fmt_val(base_beta)}\ ({_fmt_ci(beta_ci)});\ "
        rf"CS_{{1/2,0}}={_fmt_val(base_cs0)}\ ({_fmt_ci(cs0_ci)})$",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#000000",
        fontfamily=note_font,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        clip_on=True,
    )

    # Panel C1: eta sensitivity
    c1_run_order = [1, 2, 4, 3, 5, 6, 7, 8, 10]
    y_c1 = np.arange(len(c1_run_order), dtype=float)
    for i, run_id in enumerate(c1_run_order):
        val = pooled_params.get(run_id, {}).get("eta", float("nan"))
        if not np.isfinite(val) or not np.isfinite(base_eta) or val <= 0 or base_eta <= 0:
            continue
        delta_log = np.log10(val) - np.log10(base_eta)
        ci = pooled_param_ci.get(run_id, {}).get("eta", (float("nan"), float("nan")))
        ci_low = np.log10(ci[0]) - np.log10(base_eta) if np.isfinite(ci[0]) and ci[0] > 0 else np.nan
        ci_high = np.log10(ci[1]) - np.log10(base_eta) if np.isfinite(ci[1]) and ci[1] > 0 else np.nan
        ax_c1.errorbar(
            delta_log,
            i,
            xerr=[[delta_log - ci_low], [ci_high - delta_log]],
            fmt="o",
            color="#009E73",
            markersize=4,
            capsize=2,
        )
    ax_c1.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    for sep in [1.5, 3.5, 5.5, 7.5]:
        ax_c1.axhline(sep, color="#e0e0e0", linewidth=0.8)
    ax_c1.set_yticks(y_c1)
    ax_c1.set_yticklabels([short_label_map.get(rid, label_map.get(rid, str(rid))) for rid in c1_run_order], fontsize=8)
    ax_c1.set_xlabel(r"$\Delta \log_{10}(\eta)$")
    ax_c1.set_title("Eta sensitivity", loc="left", fontweight="bold")
    ax_c1.set_ylim(-0.5, len(c1_run_order) - 0.5)
    ax_c1.invert_yaxis()

    # Panel C2: kernel parameters
    c2_run_order = [7, 8, 10]
    y_c2 = np.arange(len(c2_run_order), dtype=float)
    param_keys = ["beta_max", "cs0"]
    param_labels = {"beta_max": r"$\beta_{\max}$", "cs0": r"$CS_{1/2}$"}
    param_colors = {"beta_max": "#E69F00", "cs0": "#CC79A7"}
    offsets = {"beta_max": -0.12, "cs0": 0.12}
    for i, run_id in enumerate(c2_run_order):
        for param in param_keys:
            val = pooled_params.get(run_id, {}).get(param, float("nan"))
            base_val = {"beta_max": base_beta, "cs0": base_cs0}.get(param, float("nan"))
            if not np.isfinite(val) or not np.isfinite(base_val) or base_val == 0:
                continue
            delta_pct = (val - base_val) / base_val * 100.0
            ci = pooled_param_ci.get(run_id, {}).get(param, (float("nan"), float("nan")))
            ci_low = (ci[0] - base_val) / base_val * 100.0 if np.isfinite(ci[0]) else np.nan
            ci_high = (ci[1] - base_val) / base_val * 100.0 if np.isfinite(ci[1]) else np.nan
            ax_c2.errorbar(
                delta_pct,
                i + offsets[param],
                xerr=[[delta_pct - ci_low], [ci_high - delta_pct]],
                fmt="o",
                color=param_colors[param],
                markersize=4,
                capsize=2,
            )
    ax_c2.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax_c2.set_yticks(y_c2)
    ax_c2.set_yticklabels([short_label_map.get(rid, label_map.get(rid, str(rid))) for rid in c2_run_order], fontsize=8)
    ax_c2.set_xlabel(r"$\Delta\theta/\theta_0$ (\%)")
    ax_c2.set_title("Kernel parameter shifts", loc="left", fontweight="bold")
    ax_c2.set_ylim(-0.5, len(c2_run_order) - 0.5)
    ax_c2.invert_yaxis()
    ax_c2.legend(
        handles=[plt.Line2D([0], [0], color=param_colors[p], marker="o", linestyle="None", label=param_labels[p]) for p in param_keys],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    hill_n = pooled_params.get(10, {}).get("n", float("nan"))
    hill_ci = pooled_param_ci.get(10, {}).get("n", (float("nan"), float("nan")))
    if np.isfinite(hill_n) and np.isfinite(hill_ci[0]) and np.isfinite(hill_ci[1]):
        ax_c2.text(
            0.98,
            y_c2[-1],
            f"n={_fmt_val(hill_n)} ({_fmt_val(hill_ci[0])}-{_fmt_val(hill_ci[1])})",
            transform=ax_c2.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7,
            color="#000000",
            fontfamily=note_font,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6),
            clip_on=True,
        )

    # Panels D and E: forest plots for out-of-sample skill
    skill_rows = [
        (short_label_map.get(0, "Baseline"), 0),
        (short_label_map.get(7, "CS Dmax -1"), 7),
        (short_label_map.get(8, "CS Dmax +1"), 8),
        (short_label_map.get(10, "Kernel Hill"), 10),
    ]
    y_skill = np.arange(len(skill_rows), dtype=float)
    colors_place = {"CM": "#0072B2", "JH": "#D55E00"}
    place_offsets = {"CM": -0.12, "JH": 0.12}
    baseline_fold_map: Dict[str, Dict[str, float]] = {}
    for place, metrics in metrics_by_run.get(0, {}).items():
        fold_map: Dict[str, float] = {}
        for fold in metrics.get("folds", []):
            block = fold.get("block")
            crps_val = fold.get("crps")
            if block is not None and np.isfinite(crps_val) and crps_val > 0:
                fold_map[block] = float(crps_val)
        baseline_fold_map[place] = fold_map

    def _fetch_metric(run_id: int | None, place: str, key: str) -> float:
        rows = summary_df[(summary_df["run_id"] == run_id) & (summary_df["place"] == place)]
        if rows.empty:
            return float("nan")
        return float(rows.iloc[0][key])

    def _fold_ratios(place: str, run_id: int) -> np.ndarray:
        run_metrics = metrics_by_run.get(run_id, {}).get(place, {})
        base_map = baseline_fold_map.get(place, {})
        if run_id == 0 and base_map:
            return np.ones(len(base_map), dtype=float)
        vals: List[float] = []
        for fold in run_metrics.get("folds", []):
            block = fold.get("block")
            crps_run = fold.get("crps")
            crps0 = base_map.get(block)
            if block is None or not np.isfinite(crps_run) or not np.isfinite(crps0) or crps0 <= 0:
                continue
            vals.append(float(crps_run) / float(crps0))
        return np.asarray(vals, dtype=float)

    for place, color in colors_place.items():
        for i, (_, run_id) in enumerate(skill_rows):
            x = _fetch_metric(run_id, place, "crps_ratio")
            xlo = _fetch_metric(run_id, place, "crps_ratio_low")
            xhi = _fetch_metric(run_id, place, "crps_ratio_high")
            ratios = _fold_ratios(place, run_id)
            if ratios.size:
                yjit = (i + place_offsets[place]) + rng.normal(0.0, 0.03, size=ratios.size)
                ax_d.scatter(ratios, yjit, s=7, color=color, alpha=0.12, edgecolors="none")
                ax_e.scatter(1.0 - ratios, yjit, s=7, color=color, alpha=0.12, edgecolors="none")
            if np.isfinite(x):
                if run_id == 0 or not np.isfinite(xlo) or not np.isfinite(xhi):
                    xlo = x
                    xhi = x
                ax_d.errorbar(
                    x,
                    i + place_offsets[place],
                    xerr=[[x - xlo], [xhi - x]],
                    fmt="o",
                    color=color,
                    markersize=4,
                    capsize=2,
                    label=place if i == 0 else None,
                )
            x = _fetch_metric(run_id, place, "crpss")
            xlo = _fetch_metric(run_id, place, "crpss_low")
            xhi = _fetch_metric(run_id, place, "crpss_high")
            if np.isfinite(x):
                if run_id == 0 or not np.isfinite(xlo) or not np.isfinite(xhi):
                    xlo = x
                    xhi = x
                ax_e.errorbar(
                    x,
                    i + place_offsets[place],
                    xerr=[[x - xlo], [xhi - x]],
                    fmt="o",
                    color=color,
                    markersize=4,
                    capsize=2,
                )

    ax_d.axvline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    ax_e.axvline(0, color="#333333", linestyle="--", linewidth=0.9)
    ax_d.grid(axis="y", color="#EFEFEF", linewidth=0.8)
    ax_e.grid(axis="y", color="#EFEFEF", linewidth=0.8)
    ax_d.set_yticks(y_skill)
    ax_d.set_yticklabels([label for label, _ in skill_rows], fontsize=8)
    ax_e.set_yticks(y_skill)
    ax_e.set_yticklabels([])
    ax_d.set_xlabel(r"$\mathrm{CRPS}/\mathrm{CRPS}_0$")
    ax_e.set_xlabel("CRPSS")
    ax_d.set_title("Out-of-sample CRPS ratio", loc="left", fontweight="bold")
    ax_e.set_title("CRPSS vs baseline", loc="left", fontweight="bold")
    ax_d.set_ylim(-0.5, len(skill_rows) - 0.5)
    ax_e.set_ylim(-0.5, len(skill_rows) - 0.5)
    ax_d.invert_yaxis()
    ax_e.invert_yaxis()
    ax_d.legend(frameon=False, fontsize=8, loc="upper left")
    ax_d.text(
        0.02,
        -0.18,
        "Gate-only perturbations do not enter mass prediction.",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#000000",
        fontfamily=note_font,
        fontweight="bold",
        clip_on=False,
    )
    for i, (_, run_id) in enumerate(skill_rows):
        n_cm = _fold_ratios("CM", run_id).size
        n_jh = _fold_ratios("JH", run_id).size
        if n_cm + n_jh == 0:
            continue
        ax_d.text(
            1.02,
            i,
            f"n={n_cm}/{n_jh}",
            transform=ax_d.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            color="#000000",
            fontfamily=note_font,
            fontweight="bold",
            clip_on=False,
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path_png = FIGURES_DIR / f"{args.figure_name}.png"
    fig_path_svg = FIGURES_DIR / f"{args.figure_name}.svg"
    fig.savefig(fig_path_png, dpi=500, bbox_inches="tight")
    fig.savefig(fig_path_svg, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {fig_path_png}")
    print(f"[OK] Wrote {TABLES_DIR / f'{args.table_name}.csv'}")


if __name__ == "__main__":
    main()
