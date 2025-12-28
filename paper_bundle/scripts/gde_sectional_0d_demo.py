from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow import core  # noqa: E402
from src.workflow.config import default_config  # noqa: E402


def _parse_diameter_um(col: str) -> float:
    name = col.replace("C", "").replace("um", "")
    return float(name)


def _build_sectional_grid(mass_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    pairs = [(_parse_diameter_um(c), c) for c in mass_cols]
    pairs = sorted(pairs, key=lambda tup: tup[0])
    diam_um = np.array([p[0] for p in pairs], dtype=float)
    cols_sorted = [p[1] for p in pairs]
    diam_nm = diam_um * 1000.0
    return diam_nm, cols_sorted


# References: Seinfeld and Pandis (2016) Atmospheric Chemistry and Physics, size-space advection form of the GDE.
# Equation (condensation only, no coagulation or sources): d n(d,t)/dt + d[G(d,t) n(d,t)]/dd = 0.
# Discrete scheme: for bin i with center d_i and width Delta d_i, and growth speed G_i:
#   Delta d = G_i * Delta t; fraction moved to neighbor = f_i = clip(Delta d / Delta d_i, 0, 1).
# Mass update (growth, Delta d >= 0): M_i^{new} = (1-f_i) M_i^{old}, M_{i+1}^{new} += f_i M_i^{old}.
# Mass update (evaporation, Delta d < 0): M_i^{new} = (1-f_i) M_i^{old}, M_{i-1}^{new} += f_i M_i^{old}.
# Parameters: M_i bin mass concentration, d_i bin diameter (nm), G_i growth speed (nm s^-1), Delta t time step (s).
def condensation_step(
    mass_prev: np.ndarray,
    diam_nm: np.ndarray,
    growth_nm_s: float,
    dt_s: float,
) -> np.ndarray:
    mass_prev = np.asarray(mass_prev, dtype=float)
    out = np.zeros_like(mass_prev)
    n_bins = mass_prev.size
    if n_bins == 0:
        return out
    delta_d = float(growth_nm_s) * float(dt_s)
    if delta_d == 0.0:
        return mass_prev.copy()
    if delta_d > 0:
        for i in range(n_bins):
            m_i = mass_prev[i]
            if not np.isfinite(m_i) or m_i <= 0:
                continue
            if i == n_bins - 1:
                out[i] += m_i
                continue
            width = max(diam_nm[i + 1] - diam_nm[i], 1e-6)
            frac = np.clip(delta_d / width, 0.0, 1.0)
            transfer = frac * m_i
            out[i] += m_i - transfer
            out[i + 1] += transfer
    else:
        delta_d_abs = -delta_d
        for i in range(n_bins - 1, -1, -1):
            m_i = mass_prev[i]
            if not np.isfinite(m_i) or m_i <= 0:
                continue
            if i == 0:
                out[i] += m_i
                continue
            width = max(diam_nm[i] - diam_nm[i - 1], 1e-6)
            frac = np.clip(delta_d_abs / width, 0.0, 1.0)
            transfer = frac * m_i
            out[i] += m_i - transfer
            out[i - 1] += transfer
    out = np.maximum(out, 0.0)
    return out


def _prepare_gde_demo_dataframe(cfg) -> pd.DataFrame:
    clean_path = Path("intermediate/step01_clean.parquet")
    if not clean_path.exists():
        raise FileNotFoundError(
            "Missing intermediate/step01_clean.parquet. "
            "Run scripts/step01_data_ingestion_qc.py first."
        )
    df = pd.read_parquet(clean_path)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    number_cols = [c for c in df.columns if c.startswith("C") and c.endswith("um")]
    if not number_cols:
        raise ValueError("No number concentration columns C*um available in clean dataframe.")

    # References: Fuchs and Sutugin (1971); core.compute_condensation_sink implements
    # CS = sum_i 4 pi D_v r_i F(Kn_i) N_i with accommodation-corrected F(Kn_i).
    cs_series = core.compute_condensation_sink(
        df[number_cols],
        df["temperature_c"],
        cfg.cs_diffusivity_m2_s,
        cfg.mean_free_path_nm,
        cfg.accommodation_coefficient,
    )
    df["CS_star"] = cs_series

    # References: Kulmala et al. (2007, 2013); effective diameter d_eff from small-mode mass and number.
    d_eff = core.compute_effective_diameter(df["M_1_20"], df["N_1_20"], cfg.particle_density_g_cm3)
    df["d_eff_nm"] = d_eff
    g_rate = core.compute_growth_rate(d_eff, freq_minutes=1)
    df["G_nm_s"] = g_rate
    return df


def _select_demo_window(df: pd.DataFrame, place: str, n_steps: int = 180) -> pd.DataFrame:
    sub = df[df["place"] == place].copy()
    sub = sub.dropna(
        subset=["M_1_20", "N_1_20", "G_nm_s", "CS_star", "temperature_c", "rh_pct"],
        how="any",
    )
    if sub.shape[0] <= n_steps:
        return sub
    return sub.iloc[:n_steps].copy()


def _compute_psd_skill(mass_obs: np.ndarray, mass_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    obs_flat = mass_obs.ravel()
    pred_flat = mass_pred.ravel()
    mask = np.isfinite(obs_flat) & np.isfinite(pred_flat)
    if mask.sum() < 3:
        return np.nan, np.full(mass_obs.shape[1], np.nan)
    obs_use = obs_flat[mask]
    pred_use = pred_flat[mask]
    denom = np.sum((obs_use - obs_use.mean()) ** 2)
    if denom <= 0:
        r2_overall = np.nan
    else:
        r2_overall = 1.0 - np.sum((obs_use - pred_use) ** 2) / denom

    n_bins = mass_obs.shape[1]
    r2_bins = np.full(n_bins, np.nan)
    for j in range(n_bins):
        o_j = mass_obs[:, j]
        p_j = mass_pred[:, j]
        mask_j = np.isfinite(o_j) & np.isfinite(p_j)
        if mask_j.sum() < 3:
            continue
        o_use = o_j[mask_j]
        p_use = p_j[mask_j]
        denom_j = np.sum((o_use - o_use.mean()) ** 2)
        if denom_j <= 0:
            continue
        r2_bins[j] = 1.0 - np.sum((o_use - p_use) ** 2) / denom_j
    return r2_overall, r2_bins


def _fit_growth_vs_inputs(sub: pd.DataFrame) -> Tuple[float, np.ndarray, np.ndarray]:
    work = sub.copy()
    work["CS_star"] = pd.to_numeric(work["CS_star"], errors="coerce").clip(lower=1e-6)
    work["temperature_c"] = pd.to_numeric(work["temperature_c"], errors="coerce")
    work["rh_pct"] = pd.to_numeric(work["rh_pct"], errors="coerce")
    work["G_nm_s"] = pd.to_numeric(work["G_nm_s"], errors="coerce")

    # References: Kulmala et al. (1998) and subsequent CS-based growth parameterizations.
    # Equation: G(t) = b0 + b1 log10(CS_star) + b2 T + b3 RH.
    # Parameters: b0 intercept, b1 CS sensitivity, b2 temperature sensitivity, b3 humidity sensitivity.
    X = np.column_stack(
        [
            np.log10(work["CS_star"].to_numpy(dtype=float)),
            work["temperature_c"].to_numpy(dtype=float),
            work["rh_pct"].to_numpy(dtype=float),
        ]
    )
    y = work["G_nm_s"].to_numpy(dtype=float)
    mask = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
    )
    if mask.sum() < 10:
        return np.nan, np.full(4, np.nan), np.full(mask.shape[0], np.nan)
    X_use = X[mask]
    y_use = y[mask]
    model = LinearRegression()
    model.fit(X_use, y_use)
    y_pred = model.predict(X_use)
    denom = np.sum((y_use - y_use.mean()) ** 2)
    if denom <= 0:
        r2 = np.nan
    else:
        r2 = 1.0 - np.sum((y_use - y_pred) ** 2) / denom
    coefs = np.concatenate(([model.intercept_], model.coef_))
    full_pred = model.predict(X)
    return r2, coefs, full_pred


def _plot_nature_style(
    diam_nm: np.ndarray,
    mass_obs: np.ndarray,
    mass_pred: np.ndarray,
    r2_bins: np.ndarray,
    sub: pd.DataFrame,
    growth_pred: np.ndarray,
    place: str,
) -> None:
    # Define Nature-style colors (colorblind-friendly)
    COLORS = {
        "obs_start": "#1b9e77", # Greenish
        "obs_end": "#d95f02",   # Orange
        "model_start": "#7570b3", # Purple
        "model_end": "#e7298a",   # Pink
        "r2": "#4e79a7",
        "scatter": "#1b9e77",
        "line": "#333333"
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "lines.linewidth": 1.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    sns.set_style("ticks")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # Panel (a): PSD Comparison (Start vs End)
    ax_psd = fig.add_subplot(gs[0, 0])
    idx0 = 0
    idx1 = mass_obs.shape[0] - 1
    
    ax_psd.plot(diam_nm, mass_obs[idx0], color=COLORS["obs_start"], linestyle="-", marker="o", markersize=4, label="Obs (Start)")
    ax_psd.plot(diam_nm, mass_pred[idx0], color=COLORS["model_start"], linestyle="--", label="GDE (Start)")
    ax_psd.plot(diam_nm, mass_obs[idx1], color=COLORS["obs_end"], linestyle="-", marker="s", markersize=4, label="Obs (End)")
    ax_psd.plot(diam_nm, mass_pred[idx1], color=COLORS["model_end"], linestyle="--", label="GDE (End)")
    
    ax_psd.set_xscale("log")
    ax_psd.set_xlabel("Diameter (nm)")
    ax_psd.set_ylabel("Mass Concentration ($\mu g m^{-3}$)")
    ax_psd.set_title(f"(a) Sectional PSD Evolution ({place})")
    ax_psd.grid(True, which="major", linestyle="--")
    ax_psd.legend(frameon=False, loc="upper right")
    ax_psd.xaxis.set_major_locator(MaxNLocator(5))

    # Panel (b): Skill (R2) vs Diameter
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_r2.plot(diam_nm, r2_bins, color=COLORS["r2"], marker="D", markersize=5, linestyle="-", linewidth=1.5)
    ax_r2.set_xscale("log")
    ax_r2.set_ylim(-0.1, 1.1)
    ax_r2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_r2.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax_r2.set_xlabel("Diameter (nm)")
    ax_r2.set_ylabel("Coefficient of Determination ($R^2$)")
    ax_r2.set_title("(b) Model Skill by Particle Size")
    ax_r2.grid(True, which="major", linestyle="--")

    # Panel (c): Growth Rate Validation
    ax_g = fig.add_subplot(gs[1, 0])
    g_true = sub["G_nm_s"].to_numpy(dtype=float)
    mask = np.isfinite(g_true) & np.isfinite(growth_pred)
    g_true = g_true[mask]
    g_fit = growth_pred[mask]
    
    # Scatter with density coloring if possible, else simple
    ax_g.scatter(g_true, g_fit, s=15, alpha=0.6, color=COLORS["scatter"], edgecolors="white", linewidth=0.5)
    
    lim_min = min(np.min(g_true), np.min(g_fit))
    lim_max = max(np.max(g_true), np.max(g_fit))
    buffer = (lim_max - lim_min) * 0.1
    ax_g.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black", linewidth=1.0)
    
    ax_g.set_xlim(lim_min - buffer, lim_max + buffer)
    ax_g.set_ylim(lim_min - buffer, lim_max + buffer)
    ax_g.set_xlabel("Observed Growth Rate ($nm s^{-1}$)")
    ax_g.set_ylabel("Predicted Growth Rate ($nm s^{-1}$)")
    ax_g.set_title("(c) Growth Rate Parameterization")
    ax_g.grid(True, linestyle="--")

    # Panel (d): Residual Heatmap (Obs - Pred) over Diameter and Time
    ax_res = fig.add_subplot(gs[1, 1])
    # Calculate residuals
    residuals = mass_obs - mass_pred
    # Create a meshgrid for plotting
    # Time on y, Diameter on x (log scale tricky with imshow, use pcolormesh)
    time_steps = np.arange(residuals.shape[0])
    X, Y = np.meshgrid(diam_nm, time_steps)
    
    # Use a diverging colormap centered at 0
    mesh = ax_res.pcolormesh(X, Y, residuals, cmap="RdBu_r", shading='auto', vmin=-np.nanmax(np.abs(residuals)), vmax=np.nanmax(np.abs(residuals)))
    ax_res.set_xscale("log")
    ax_res.set_xlabel("Diameter (nm)")
    ax_res.set_ylabel("Time Step")
    ax_res.set_title("(d) Residuals (Obs - Mod)")
    fig.colorbar(mesh, ax=ax_res, label="Error ($\mu g m^{-3}$)")

    plt.tight_layout()
    
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Fig_GDE_sectional_demo_{place}"
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[GDE] Saved sectional demo figure to figures/{stem}.png")


def aggregate_by_hour_min(df: pd.DataFrame) -> pd.DataFrame:
    if "place" not in df.columns:
        return df
    work = df.copy()
    work["hour_min"] = work.index.strftime("%H:%M")
    numeric_cols = work.select_dtypes(include=[np.number]).columns
    grouped = work.groupby(["place", "hour_min"])[numeric_cols].mean().reset_index()
    
    base_date = pd.to_datetime("2000-01-01")
    grouped["minutes_since_midnight"] = grouped["hour_min"].apply(
        lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
    )
    grouped = grouped.sort_values(["place", "minutes_since_midnight"])
    place_offsets = {p: i for i, p in enumerate(grouped["place"].unique())}
    grouped["Time"] = grouped.apply(
        lambda row: base_date
        + pd.Timedelta(days=place_offsets.get(row["place"], 0))
        + pd.to_timedelta(row["minutes_since_midnight"], unit="m"),
        axis=1,
    )
    grouped = grouped.drop(columns=["minutes_since_midnight"])
    grouped = grouped.set_index("Time")
    return grouped

def main() -> None:
    cfg = default_config()
    print("[GDE Demo] Loading data and aggregating to Hour_Min diurnal cycle...")
    df = _prepare_gde_demo_dataframe(cfg)
    
    # Aggregate to diurnal cycle
    df = aggregate_by_hour_min(df)
    
    places = sorted(df["place"].dropna().unique().tolist())
    if not places:
        raise RuntimeError("No place column or no sites found in clean dataframe.")
    place = places[0]
    
    # Use full diurnal cycle (1440 mins) instead of 180 steps
    sub = _select_demo_window(df, place=place, n_steps=1440)

    mass_cols = [c for c in sub.columns if c.endswith("um") and not c.startswith("C")]
    if not mass_cols:
        raise ValueError("No mass size-distribution columns (*um) found for GDE demo.")
    # Focus on submicron sizes where the proxy CS and growth diagnostics have been calibrated.
    mass_cols_small = [c for c in mass_cols if _parse_diameter_um(c) <= 1.0]
    if not mass_cols_small:
        mass_cols_small = mass_cols
    diam_nm, cols_sorted = _build_sectional_grid(mass_cols_small)
    psd_obs = sub[cols_sorted].to_numpy(dtype=float)

    g_series = sub["G_nm_s"].to_numpy(dtype=float)
    g_series = pd.Series(g_series, index=sub.index)
    g_series = g_series.fillna(method="bfill").fillna(method="ffill").fillna(0.0)

    dt_seconds = 60.0 # Hour_Min data is 1-minute resolution
    n_steps, n_bins = psd_obs.shape
    psd_pred = np.zeros_like(psd_obs)
    psd_pred[0] = np.maximum(psd_obs[0], 0.0)
    for k in range(1, n_steps):
        g_k = float(g_series.iloc[k - 1])
        psd_pred[k] = condensation_step(psd_pred[k - 1], diam_nm, g_k, dt_seconds)

    r2_overall, r2_bins = _compute_psd_skill(psd_obs, psd_pred)
    print("[GDE] Demo site:", place)
    print("[GDE] Time steps (Diurnal), bins:", n_steps, n_bins)
    print(f"[GDE] Overall sectional PSD R2: {r2_overall:.3f}" if np.isfinite(r2_overall) else "[GDE] Overall R2: nan")

    r2_growth, coefs, g_pred = _fit_growth_vs_inputs(sub)
    if np.isfinite(r2_growth):
        print(f"[GDE] Growth vs (CS, T, RH) R2: {r2_growth:.3f}")
        print("[GDE] Growth regression coefficients (intercept, log10(CS), T, RH):")
        print(coefs)
    else:
        print("[GDE] Growth vs inputs regression could not be fitted (insufficient valid samples).")

    _plot_nature_style(diam_nm, psd_obs, psd_pred, r2_bins, sub, g_pred, place)


if __name__ == "__main__":
    main()

