from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow.config import default_config  # noqa: E402
from src.workflow.gde_solver import GDEConfig, run_gde_simulation_full, load_high_freq_state  # noqa: E402


def _set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "lines.linewidth": 1.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    sns.set_style("whitegrid")
    sns.set_palette(["#1b9e77", "#d95f02", "#7570b3", "#e7298a"])


def _plot_site_demo(place: str, payload: Dict[str, object]) -> None:
    time_index = payload["time"]
    mass_obs = payload["mass_obs"]
    mass_sim = payload["mass_sim"]
    r2_overall = payload["r2_overall"]
    c_gas = payload.get("C_gas")
    cs_tot = payload.get("cs_tot")
    p_series = payload.get("p_series")

    n_steps, n_bins = mass_obs.shape
    total_obs = np.nansum(mass_obs, axis=1)
    total_sim = np.nansum(mass_sim, axis=1)
    total_err = total_sim - total_obs
    
    with np.errstate(divide="ignore", invalid="ignore"):
        total_ratio = total_sim / total_obs
    
    mask_tot = np.isfinite(total_obs) & np.isfinite(total_sim)
    rmse_tot = float(np.sqrt(np.mean((total_obs[mask_tot] - total_sim[mask_tot]) ** 2))) if mask_tot.any() else np.nan
    bias_tot = float(np.mean(total_sim[mask_tot] - total_obs[mask_tot])) if mask_tot.any() else np.nan
    
    # Nature-style styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    sns.set_style("ticks")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_cgas = fig.add_subplot(gs[1, 1])

    # Panel (a): Time Series
    t_rel_hours = (time_index - time_index[0]).total_seconds() / 3600.0
    ax_ts.plot(t_rel_hours, total_obs, color="#1b9e77", label="Observed", linewidth=1.5)
    ax_ts.plot(t_rel_hours, total_sim, color="#d95f02", linestyle="--", label="Simulated", linewidth=1.5)
    ax_ts.set_xlabel("Elapsed Time (h)")
    ax_ts.set_ylabel("Total Mass ($\\mu g m^{-3}$)")
    ax_ts.set_title(f"(a) Total Small-Mode Mass Evolution ({place})")
    ax_ts.grid(True)
    ax_ts.legend(frameon=False, loc="best")
    
    stats_text = f"$R^2={r2_overall:.2f}$\nRMSE={rmse_tot:.2f}\nBias={bias_tot:.2f}"
    ax_ts.text(0.02, 0.95, stats_text, transform=ax_ts.transAxes, va="top", ha="left", 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Panel (b): Error and Ratio
    ax_err.plot(t_rel_hours, total_err, color="#e7298a", label="Error (Sim-Obs)")
    ax_err.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_err.set_xlabel("Elapsed Time (h)")
    ax_err.set_ylabel("Mass Error ($\\mu g m^{-3}$)")
    ax_err.set_title("(b) Model Error Dynamics")
    ax_err.grid(True)
    
    # Panel (c): Error Distribution
    valid_err = total_err[np.isfinite(total_err)]
    sns.histplot(valid_err, kde=True, ax=ax_hist, color="#4e79a7", edgecolor="white", alpha=0.6)
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax_hist.set_xlabel("Mass Error ($\\mu g m^{-3}$)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("(c) Error Distribution")
    ax_hist.grid(True)

    # Panel (d): Gas Phase Dynamics
    if c_gas is not None:
        l1 = ax_cgas.plot(t_rel_hours, c_gas, color="#7570b3", label="Sim. Vapor ($C_{gas}$)")
    if cs_tot is not None:
        ax_cgas_r = ax_cgas.twinx()
        l2 = ax_cgas_r.plot(t_rel_hours[:len(cs_tot)], cs_tot, color="#66a61e", linestyle="--", label="Cond. Sink ($CS$)")
        ax_cgas_r.set_ylabel("CS ($s^{-1}$)", color="#66a61e")
        ax_cgas_r.tick_params(axis='y', labelcolor="#66a61e")
    
    ax_cgas.set_xlabel("Elapsed Time (h)")
    ax_cgas.set_ylabel("Vapor Conc. ($\\mu g m^{-3}$)", color="#7570b3")
    ax_cgas.tick_params(axis='y', labelcolor="#7570b3")
    ax_cgas.set_title("(d) Vapor and Sink Dynamics")
    ax_cgas.grid(True)

    plt.tight_layout()
    
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Fig_GDE_PSD_demo_{place}"
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[GDE] Saved site demo figure to figures/{stem}.png")


def aggregate_by_hour_min(df: pd.DataFrame) -> pd.DataFrame:
    if "place" not in df.columns:
        return df
    work = df.copy()
    
    # Ensure Time index
    if "Time" in work.columns:
        work["Time"] = pd.to_datetime(work["Time"])
        work = work.set_index("Time")
    
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'Time' column for aggregation.")

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

def prepare_aggregated_data() -> Dict[str, pd.DataFrame]:
    print("[GDE Demo] Loading and aggregating data to diurnal cycle...")
    base_path = Path("intermediate/step01_clean.parquet")
    growth_path = Path("intermediate/step04_growth_metrics_hf.parquet")
    japp_path = Path("intermediate/step05_japp_survival.parquet")
    
    df_base = pd.read_parquet(base_path)
    df_growth = pd.read_parquet(growth_path)
    df_japp = pd.read_parquet(japp_path)
    
    agg_base = aggregate_by_hour_min(df_base)
    
    if "place" not in df_growth.columns and "Place" in df_growth.columns:
        df_growth = df_growth.rename(columns={"Place": "place"})
    if "place" not in df_japp.columns and "Place" in df_japp.columns:
        df_japp = df_japp.rename(columns={"Place": "place"})
        
    agg_growth = aggregate_by_hour_min(df_growth)
    agg_japp = aggregate_by_hour_min(df_japp)
    
    return {
        "base": agg_base,
        "growth": agg_growth,
        "japp": agg_japp
    }

def main() -> None:
    cfg = default_config()
    # Configure for 1min resolution consistent with aggregated data
    gde_cfg = GDEConfig(
        use_coagulation=True,
        hf_rule="1min"
    )
    _set_nature_style()
    
    # Load aggregated data
    raw_diurnal_data = prepare_aggregated_data()
    
    # Process into site payloads
    print("[GDE Demo] Processing aggregated data into simulation payloads...")
    site_payloads = load_high_freq_state(cfg, gde_cfg, override_inputs=raw_diurnal_data)
    
    # Run simulation
    results = run_gde_simulation_full(cfg, gde_cfg, preloaded_inputs=site_payloads)
    
    for place, payload in results.items():
        _plot_site_demo(place, payload)
    print("[GDE demo] Finished. Figures saved to figures\\Fig_GDE_PSD_demo_{place}.svg/.png for each place.")


if __name__ == "__main__":
    main()