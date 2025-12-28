from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.workflow.lib.paper_paths import (
    BUNDLE_ROOT,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    INTERMEDIATE_DIR,
)
from src.workflow.modeling_framework import load_cached_results, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mass closure: Delta M vs integral of J_app * S_surv.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURE_DIR, help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=CHECKPOINT_DIR, help="Directory to save tables.")
    parser.add_argument(
        "--deltas-min",
        type=float,
        nargs="+",
        default=[10.0 / 60.0, 30.0 / 60.0, 1.0, 10.0, 30.0, 60.0],
        help="Integration windows (minutes). Default: 10s, 30s, 1, 10, 30, 60 min; high-frequency advantage when delta<=1 min.",
    )
    parser.add_argument("--d_soA_nm", type=float, default=200.0, help="Target diameter for mass closure (nm).")
    parser.add_argument("--d_nuc_nm", type=float, default=2.0, help="Nucleation diameter (nm).")
    parser.add_argument("--rho_g_cm3", type=float, default=1.3, help="Particle density (g cm^-3).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument(
        "--min-I-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold on I_delta to define high-signal windows (e.g., 0.8 keeps top 20%).",
    )
    parser.add_argument(
        "--lags-min",
        type=float,
        nargs="*",
        default=[0.0],
        help="Optional lags (minutes) to shift DeltaM relative to I_delta for regression (default 0).",
    )
    return parser.parse_args()


def _compute_mass_per_particle(d_soA_nm: float, d_nuc_nm: float, rho_g_cm3: float) -> float:
    """
    References: spherical particle mass m_p = (4/3)*pi*rho*(d^3-d_nuc^3)/8.
    Parameters: d in nm, rho in g cm^-3. Returns ug per particle.
    """
    d_soA_cm = d_soA_nm * 1e-7
    d_nuc_cm = d_nuc_nm * 1e-7
    volume_cm3 = (np.pi / 6.0) * (d_soA_cm**3 - d_nuc_cm**3)
    mass_g = volume_cm3 * rho_g_cm3
    return mass_g * 1e6  # ug


def _rolling_integral(prod: pd.Series, window_steps: int, dt_seconds: float) -> pd.Series:
    return prod.rolling(window_steps, min_periods=1).sum() * dt_seconds


def _apply_nature_style() -> None:
    """Apply bold, Nature-like styling for all text and axes."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 13,
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.6,
        }
    )
    sns.set_style("whitegrid", {"grid.linestyle": "-", "grid.color": "#d9d9d9", "axes.edgecolor": "#222222"})


def _format_sci_10x(val: float) -> str:
    """Format number in scientific notation using LaTeX 10^{k} form without caret in plain text."""
    if not np.isfinite(val) or val == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10.0**exponent)
    return r"$" + f"{mantissa:.2f}\\times10^{{{exponent}}}" + r"$"


def main() -> None:
    args = parse_args()
    set_plot_style()
    _apply_nature_style()
    # Local overrides removed to respect global Nature style from set_plot_style()
    
    df_sde, cat1, cat2, ml_outputs, labels_cfg = load_cached_results(CHECKPOINT_DIR, CHECKPOINT_DIR)
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)
    step05_path = CHECKPOINT_DIR / "step05_japp_survival.parquet"
    if not step05_path.exists():
        raise FileNotFoundError(f"{step05_path} not found; run step05_japp_survival.py first.")
    hf = pd.read_parquet(step05_path)
    if "Time" in hf.columns:
        hf["Time"] = pd.to_datetime(hf["Time"])
        hf = hf.set_index("Time")
    hf.index = pd.to_datetime(hf.index)
    hf = hf.sort_index()
    if "place" not in hf.columns:
        hf["place"] = "ALL"
    # Use fixed 10 s grid per requirement; ignore irregular median intervals.
    dt_seconds = 10.0

    # Use delta_mass from step05 as high-frequency mass increment (10 s horizon in step05).
    delta_mass_hf = pd.to_numeric(hf.get("delta_mass"), errors="coerce")

    figures_dir = args.figures_dir
    tables_dir = args.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Reference: Kulmala et al. (2007) Atmos. Chem. Phys.; Equation: m_p = (pi/6) * rho * (d_f^3 - d_nuc^3); Parameters: d_f is effective final diameter (nm), d_nuc is nucleation diameter (nm), rho is particle density (g cm^-3).
    # Use site-specific effective final diameters d_f derived from high-frequency growth metrics so that the theoretical mass per particle reflects observed growth at each site.
    place_diam_nm: Dict[str, float] = {}
    step04_path = INTERMEDIATE_DIR / "step04_growth_metrics_hf.parquet"
    if step04_path.exists():
        df_g = pd.read_parquet(step04_path)
        if "place" in df_g.columns and "d_eff_nm" in df_g.columns:
            # Use 0.75 quantile of d_eff_nm as effective final diameter only for JH; CM falls back to global target diameter to avoid overestimating its theoretical mass.
            q = df_g.groupby("place")["d_eff_nm"].quantile(0.75)
            place_diam_nm = q.to_dict()
    m_p_ug_global = _compute_mass_per_particle(args.d_soA_nm, args.d_nuc_nm, args.rho_g_cm3)
    rows: List[Dict[str, object]] = []

    # Respect requested integration windows; high-frequency detail is strongest when delta<=1 min.
    deltas = sorted(set(args.deltas_min))
    places = sorted(hf["place"].dropna().unique())
    n_cols = 3
    n_rows = max(1, int(np.ceil(len(deltas) / n_cols)))
    fig_width = 12.0
    fig_height = 3.8 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).ravel()

    # Nature-style colors with higher contrast
    colors = {"JH": "#0b8043", "CM": "#b35900"}
    
    # Collect per-panel ranges to set axis limits based on quantiles (zoom out extremes).
    panel_x_values: Dict[float, List[float]] = {d: [] for d in deltas}
    panel_y_values: Dict[float, List[float]] = {d: [] for d in deltas}
    # Store fit parameters per panel and place so that lines can be redrawn with uniform length.
    fit_params: Dict[float, Dict[str, Dict[str, float]]] = {d: {} for d in deltas}
    for i, delta in enumerate(deltas):
        win_steps = max(1, int(round(delta * 60.0 / dt_seconds)))
        ax = axes[i]
        
        for place in places:
            sub = hf[hf["place"] == place].sort_index()
            if sub.empty:
                continue
            if place == "JH" and place in place_diam_nm:
                d_target_nm = float(place_diam_nm[place])
            else:
                d_target_nm = float(args.d_soA_nm)
            m_p_ug = _compute_mass_per_particle(d_target_nm, args.d_nuc_nm, args.rho_g_cm3)
            alpha_theory = m_p_ug * 1e6  # ug * (#/cm3 to #/m3)
            alpha_theory = m_p_ug * 1e6  # ug * (#/cm3 to #/m3)
            prod = (sub["J_app"] * sub["S_surv"]).fillna(0.0)
            integral = _rolling_integral(prod, win_steps, dt_seconds)
            dm_series = delta_mass_hf.loc[sub.index]
            # delta_mass is 10 s increment; aggregate over window by rolling sum.
            delta_m = dm_series.rolling(win_steps, min_periods=1).sum()
            for lag_min in args.lags_min:
                lag_steps = int(round(lag_min * 60.0 / dt_seconds))
                y_lag = delta_m if lag_steps == 0 else delta_m.shift(-lag_steps)
                valid = integral.notna() & y_lag.notna()
                if valid.sum() < 5:
                    continue
                if np.isclose(integral[valid].std(), 0) or np.isclose(y_lag[valid].std(), 0):
                    continue
                x = integral[valid]
                y = y_lag[valid]
                panel_x_values[delta].append(x.values)
                panel_y_values[delta].append(y.values)
                slope, intercept = np.polyfit(x, y, 1)
                r = x.corr(y)
                # Uncentered R^2: R^2_uc = 1 - sum((y - yhat)^2) / sum(y^2).
                y_hat = slope * x + intercept
                ss_res = float(((y - y_hat) ** 2).sum())
                ss_tot_unc = float((y**2).sum())
                r_uc2 = 1.0 - ss_res / ss_tot_unc if ss_tot_unc > 0 else np.nan
                # High-signal subset on I_delta
                q = args.min_I_quantile
                if 0.0 < q < 1.0:
                    threshold = x.quantile(q)
                    mask_sig = x >= threshold
                    x_sig = x[mask_sig]
                    y_sig = y[mask_sig]
                    if len(x_sig) >= 5 and x_sig.std() > 0 and y_sig.std() > 0:
                        slope_sig, _ = np.polyfit(x_sig, y_sig, 1)
                        r_sig = x_sig.corr(y_sig)
                        # Uncentered R^2 only in high-signal region: R^2_uc_hi = 1 - sum((y - yhat)^2) / sum(y^2).
                        y_hat_sig = slope_sig * x_sig + intercept
                        ss_res_sig = float(((y_sig - y_hat_sig) ** 2).sum())
                        ss_tot_unc_sig = float((y_sig**2).sum())
                        r_uc2_sig = 1.0 - ss_res_sig / ss_tot_unc_sig if ss_tot_unc_sig > 0 else np.nan
                    else:
                        slope_sig, r_sig, r_uc2_sig = np.nan, np.nan, np.nan
                    n_high = int(mask_sig.sum())
                else:
                    slope_sig, r_sig, r_uc2_sig, n_high = np.nan, np.nan, np.nan, np.nan

                rows.append(
                    {
                        "Place": place,
                        "Delta_min": delta,
                        "Lag_min": lag_min,
                        "alpha_hat": slope,
                        "alpha_theory": alpha_theory,
                        "r": r,
                        "r_uc2": r_uc2,
                        "n": int(valid.sum()),
                        "alpha_hat_highI": slope_sig,
                        "r_highI": r_sig,
                        "r_uc2_highI": r_uc2_sig,
                        "n_highI": n_high,
                        "I_delta_median": float(x.median()),
                        "I_delta_q90": float(x.quantile(0.9)),
                        "DeltaM_median": float(y.median()),
                        "DeltaM_q90": float(y.quantile(0.9)),
                    }
                    )

                if lag_min == 0.0:
                    fit_params[delta][place] = {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "alpha_theory": float(alpha_theory),
                    }
                    color = colors.get(place, "#333333")
                    
                    # 1. Scatter: zorder=1 (bottom), downsample to avoid clutter.
                    max_points = 3000
                    if len(x) > max_points:
                        sample_idx = np.random.choice(x.index, size=max_points, replace=False)
                        xs = x.loc[sample_idx]
                        ys = y.loc[sample_idx]
                    else:
                        xs = x
                        ys = y
                    marker = "o" if place == "CM" else "s"
                    sns.scatterplot(
                        x=xs,
                        y=ys,
                        s=7,
                        alpha=0.10,
                        color=color,
                        ax=ax,
                        edgecolor="none",
                        marker=marker,
                        zorder=1,
                    )

                    # Use panel-wide x-range (later limited via quantiles) to build smooth lines.
                    x_min_local = max(0.0, float(x.min()))
                    x_max_local = float(x.max())
                    xline = np.linspace(x_min_local, x_max_local, 100)
                    
                    # 2. Theory line: dark, distinct linestyle per place, zorder=2
                    theory_ls = "--" if place == "JH" else "-."
                    theory_label = None
                    if i == 0:
                        theory_label = f"Theory ({place})"
                    
                    ax.plot(
                        xline,
                        alpha_theory * xline,
                        color="#222222", # Always dark
                        linestyle=theory_ls,
                        linewidth=2.5,
                        alpha=0.8,
                        label=theory_label,
                        zorder=3
                    )

                    # 3. Regression line: colored, solid, highest zorder
                    fit_label = None
                    if i == 0:
                        fit_label = f"Fit ({place})"
                    ax.plot(
                        xline,
                        slope * xline + intercept,
                        color=color,
                        linestyle="-",
                        linewidth=1.8,
                        label=fit_label,
                        zorder=4
                    )
                    
                    # 4. Annotation
                    annot_text = (
                        f"$\\bf{{{place}}}$\n"
                        f"Slope (fit): {_format_sci_10x(slope)}\n"
                        f"Slope (theory): {_format_sci_10x(alpha_theory)}\n"
                        f"$r$: {r:.2f}, $R^2_{{uc,hi}}$: {r_uc2_sig:.2f}"
                    )
                    
                    # Place JH and CM annotations in the lower-right corner but with sufficient vertical separation.
                    y_pos = 0.05 if place == "JH" else 0.32
                    ax.text(
                        0.98,
                        y_pos,
                        annot_text,
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        bbox=None,
                        zorder=10,
                    )
        
        delta_label = f"{delta * 60:.0f} s" if delta < 1.0 else f"{delta:g} min"
        ax.set_xlabel(r"$\int J_{app} S_{surv} dt$ (cm$^{-3}$)", fontweight="bold")
        ax.set_ylabel(r"$\Delta M$ ($\mu$g m$^{-3}$)", fontweight="bold")
        ax.set_title(f"{delta_label} integration window", fontweight="bold")
        ax.grid(alpha=0.35, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=11, width=1.1, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Ensure no per-axis legend remains; use only the global figure legend.
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        # Zoom axis limits based on central quantiles to avoid extreme outliers dominating.
        if panel_x_values[delta]:
            import numpy as _np

            x_all = _np.concatenate(panel_x_values[delta])
            y_all = _np.concatenate(panel_y_values[delta])
            x_max = _np.nanquantile(x_all, 0.995)
            y_lo = _np.nanquantile(y_all, 0.005)
            y_hi = _np.nanquantile(y_all, 0.995)
            ax.set_xlim(left=0.0, right=x_max * 1.05 if x_max > 0 else None)
            pad = 0.1 * (y_hi - y_lo if y_hi > y_lo else 1.0)
            ax.set_ylim(y_lo - pad, y_hi + pad)
        # Redraw theory and fit lines over the full x-range so that CM and JH lines have equal length in each panel.
        params_by_place = fit_params.get(delta, {})
        if params_by_place:
            x_min_plot, x_max_plot = ax.get_xlim()
            xline_full = np.linspace(max(0.0, x_min_plot), x_max_plot, 100)
            for place in places:
                if place not in params_by_place:
                    continue
                p = params_by_place[place]
                color = colors.get(place, "#333333")
                theory_ls = "--" if place == "JH" else "-."
                # Theory line (no extra labels to avoid legend clutter, higher zorder).
                line_theory = ax.plot(
                    xline_full,
                    p["alpha_theory"] * xline_full,
                    color="#222222",
                    linestyle=theory_ls,
                    linewidth=2.0,
                    alpha=0.8,
                    zorder=5,
                )[0]
                # Make dash patterns more open so theory lines are visually distinct from solid fits.
                if place == "JH":
                    line_theory.set_dashes([6, 4])  # longer dash, clear gaps
                else:
                    line_theory.set_dashes([2, 4])  # dotted style with visible spacing
                # Fit line
                ax.plot(
                    xline_full,
                    p["slope"] * xline_full + p["intercept"],
                    color=color,
                    linestyle="-",
                    linewidth=1.6,
                    zorder=6,
                )
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("#222222")

    for j in range(len(deltas), len(axes)):
        axes[j].axis("off")

    # Single shared legend using handles from first axis.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, lbl in zip(handles, labels):
            if lbl and lbl not in uniq:
                uniq[lbl] = h
        fig.legend(
            uniq.values(),
            uniq.keys(),
            frameon=False,
            fontsize=12,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.02),
            prop={"weight": "bold"},
            columnspacing=1.2,
            handletextpad=0.6,
            markerscale=1.3,
        )

    fig.tight_layout()
    fig.savefig(figures_dir / "SOA_mass_closure_deltaM_vs_I.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "Table_MassClosure_alpha.csv", index=False)
    print("Mass closure analysis complete.")


if __name__ == "__main__":
    main()
