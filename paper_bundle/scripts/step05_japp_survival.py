from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow.config import default_config  # noqa: E402

HF_RULE = "10s"
HF_SECONDS = int(pd.to_timedelta(HF_RULE).total_seconds())
SITE_COLORS = {
    "CM": {"japp": "#004c6d", "surv": "#f28e2b"},
    "JH": {"japp": "#b31b1b", "surv": "#59a14f"},
}
SINGLE_COLORS = {"japp": "#1f77b4", "surv": "#ff7f0e"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.2,
    }
)


def save_dual(fig: plt.Figure, stem: str) -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"figures/{stem}.svg", bbox_inches="tight")
    fig.savefig(f"figures/{stem}.png", dpi=300, bbox_inches="tight")


def resample_high_freq(df: pd.DataFrame, rule: str, cfg) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).resample(rule).asfreq()
    base_dt = pd.to_timedelta(cfg.resample_rule)
    target_dt = pd.to_timedelta(rule)
    max_gap = cfg.short_gap_limit * base_dt
    limit_steps = max(1, int(np.ceil(max_gap / target_dt)))
    numeric = numeric.interpolate(method="time", limit=limit_steps, limit_direction="both")
    out = numeric.copy()
    for col in df.columns:
        if col in numeric.columns:
            continue
        out[col] = df[col].resample(rule).ffill()
    return out


def compute_targets(df: pd.DataFrame, horizon_minutes: int, freq_seconds: int) -> pd.Series:
    steps = max(1, int(round(horizon_minutes * 60 / freq_seconds)))
    future = df["M_1_20"].shift(-steps)
    delta = future - df["M_1_20"]
    return delta.rename("delta_mass")


def evaluate_combo(
    df: pd.DataFrame,
    dndt: pd.Series,
    g_abs: pd.Series,
    cs_star: pd.Series,
    params: Tuple[float, float, float],
    delta_d_nm: float,
    freq_seconds: int,
    horizon_minutes: int,
) -> Dict[str, float]:
    kappa, theta, eta = params
    term_growth = theta * (g_abs / delta_d_nm) * df["N_1_20"]
    j_app = (dndt + kappa * df["N_1_20"] + term_growth).clip(lower=0)
    survival = 1.0 / (1.0 + eta * (cs_star / g_abs))
    indicator = (j_app * survival).rename("I_survived")

    target = compute_targets(df, horizon_minutes=horizon_minutes, freq_seconds=freq_seconds)
    aligned = pd.concat([indicator, target], axis=1).dropna()
    if aligned.empty:
        return {"r2": np.nan, "rmse": np.nan, "kappa": kappa, "theta": theta, "eta": eta}

    residual = aligned["I_survived"] - aligned["delta_mass"]
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((aligned["delta_mass"] - aligned["delta_mass"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    rmse = np.sqrt(np.mean(residual**2))
    return {
        "r2": r2,
        "rmse": rmse,
        "kappa": kappa,
        "theta": theta,
        "eta": eta,
    }


def main() -> None:
    cfg = default_config()
    df = pd.read_parquet("intermediate/step01_clean.parquet")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")

    hf_df = resample_high_freq(df, HF_RULE, cfg)
    growth = pd.read_parquet("intermediate/step04_growth_metrics_hf.parquet")
    growth_noplace = growth.drop(columns=["place"], errors="ignore")
    merged = hf_df.join(growth_noplace, how="inner")

    freq_seconds = HF_SECONDS
    horizon_minutes = cfg.delta_mass_minutes

    if "place" in merged.columns:
        dndt = merged.groupby("place")["N_1_20"].diff() / freq_seconds
    else:
        dndt = merged["N_1_20"].diff() / freq_seconds

    g_base = merged["G_nm_s"].replace(0, np.nan).ffill().bfill()
    g_abs = g_base.abs().clip(lower=1e-4)

    grid_results = []
    best_metrics = None
    for kappa in cfg.kappa_grid:
        for theta in cfg.theta_out_grid:
            for eta in cfg.eta_grid:
                metrics = evaluate_combo(
                    merged,
                    dndt,
                    g_abs,
                    merged["CS_star"],
                    (kappa, theta, eta),
                    cfg.delta_d_nm,
                    freq_seconds,
                    horizon_minutes,
                )
                grid_results.append(metrics)
                if best_metrics is None or (
                    np.nan_to_num(metrics["r2"], nan=-np.inf) > np.nan_to_num(best_metrics["r2"], nan=-np.inf)
                ):
                    best_metrics = metrics

    result_df = pd.DataFrame(grid_results).sort_values("r2", ascending=False)
    result_df.to_csv("tables/Table05_GridSearch.csv", index=False)

    best_params = (best_metrics["kappa"], best_metrics["theta"], best_metrics["eta"])
    kappa, theta, eta = best_params
    j_app = (
        (dndt + kappa * merged["N_1_20"] + theta * (g_abs / cfg.delta_d_nm) * merged["N_1_20"])
        .clip(lower=0)
        .rename("J_app")
    )
    survival = (1.0 / (1.0 + eta * (merged["CS_star"] / g_abs))).rename("S_surv")
    indicator = (j_app * survival).rename("I_indicator")
    delta_mass = compute_targets(merged, horizon_minutes=cfg.delta_mass_minutes, freq_seconds=freq_seconds)
    output = pd.concat([j_app, survival, indicator, delta_mass], axis=1)
    if "place" in merged.columns:
        output["place"] = merged["place"]
    output.to_parquet("intermediate/step05_japp_survival.parquet")

    ratio_df = pd.DataFrame(
        {
            "ratio": (merged["CS_star"] / g_abs).replace([np.inf, -np.inf], np.nan),
            "survival": survival,
        }
    ).dropna()
    ratio_df = ratio_df[ratio_df["ratio"] > 0]
    valid_pairs = output[["I_indicator", "delta_mass"]].dropna()
    if not ratio_df.empty:
        ratio_log = np.log10(ratio_df["ratio"])
        surv_corr = ratio_log.corr(ratio_df["survival"], method="spearman")
    else:
        ratio_log = pd.Series(dtype=float)
        surv_corr = np.nan

    window = slice("2022-08-06", "2022-08-08")
    window_data = output.loc[window].copy()
    if window_data.empty:
        window_data = output.copy()
    window_data = window_data.dropna(subset=["J_app", "S_surv"], how="any")
    active_mask = (window_data["J_app"].abs() > 1e-6) & (window_data["S_surv"].abs() > 1e-6)
    window_data = window_data.loc[active_mask]
    if window_data.empty:
        window_data = output.dropna(subset=["J_app", "S_surv"], how="any")
    reference_time = window_data.index.min()
    window_data = window_data.assign(
        elapsed_hours=(window_data.index - reference_time).total_seconds() / 3600.0
    )

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.3)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top2 = ax_top.twinx()
    legend_entries: list[tuple[str, str, str]] = []
    if "place" in window_data.columns and window_data["place"].notna().any():
        for place, sub in window_data.groupby("place"):
            if sub.empty:
                continue
            sub = sub.dropna(subset=["J_app", "S_surv"], how="any")
            if sub.empty:
                continue
            gap = sub.index.to_series().diff().gt(pd.Timedelta(minutes=10))
            j_series = sub["J_app"].copy()
            s_series = sub["S_surv"].copy()
            j_series[gap] = np.nan
            s_series[gap] = np.nan
            ehrs = sub["elapsed_hours"]
            colors = SITE_COLORS.get(place, SINGLE_COLORS)
            (line_j,) = ax_top.plot(ehrs, j_series, color=colors["japp"], label=rf"$J_{{\mathrm{{app}}}}$ ({place})", zorder=2)
            (line_s,) = ax_top2.plot(
                ehrs,
                s_series,
                color=colors["surv"],
                linestyle="--",
                alpha=0.25,
                linewidth=1.0,
                zorder=0,
                label=f"$S_{{surv}}$ {place}",
            )
            legend_entries.extend(
                [
                    (fr"$J_{{\mathrm{{app}}}}$ ({place})", colors["japp"], "-"),
                    (fr"$S_{{\mathrm{{surv}}}}$ ({place})", colors["surv"], "--"),
                ]
            )
    else:
        gap = window_data.index.to_series().diff().gt(pd.Timedelta(minutes=10))
        j_series = window_data["J_app"].copy()
        s_series = window_data["S_surv"].copy()
        j_series[gap] = np.nan
        s_series[gap] = np.nan
        line_j = ax_top.plot(window_data["elapsed_hours"], j_series, color=SINGLE_COLORS["japp"], label=r"$J_{\mathrm{app}}$", zorder=2)[0]
        line_s = ax_top2.plot(
            window_data["elapsed_hours"],
            s_series,
            color=SINGLE_COLORS["surv"],
            linestyle="--",
            alpha=0.25,
            linewidth=1.0,
            zorder=0,
            label="$S_{surv}$",
        )[0]
        legend_entries.extend(
            [
                (r"$J_{\mathrm{app}}$", SINGLE_COLORS["japp"], "-"),
                (r"$S_{\mathrm{surv}}$", SINGLE_COLORS["surv"], "--"),
            ]
        )
    if legend_entries:
        proxies: list[Line2D] = []
        labels: list[str] = []
        seen = set()
        for label, color, linestyle in legend_entries:
            if label in seen:
                continue
            seen.add(label)
            proxies.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5))
            labels.append(label)
        ax_top.legend(proxies, labels, frameon=False, loc="upper right", ncols=2)
    ax_top.set_ylabel(r"$J_{\mathrm{app}}$ (cm$^{-3}$ s$^{-1}$)")
    ax_top2.set_ylabel(r"$S_{\mathrm{surv}}$ (-)", color="#aa6c39")
    ax_top2.tick_params(axis="y", labelcolor="#aa6c39")
    if not window_data.empty:
        max_hours = window_data["elapsed_hours"].max()
        ax_top.set_xlim(0, max_hours)
        ax_top.set_ylim(0, window_data["J_app"].quantile(0.99) * 1.1)
        ax_top2.set_ylim(0, window_data["S_surv"].quantile(0.99) * 1.1)
    ax_top.set_xlabel("Elapsed time (h)")
    ax_top.set_title("(a) High-frequency formation vs. survival (10 s)")
    ax_top.grid(alpha=0.3, linewidth=0.8)

    ax_mid = fig.add_subplot(gs[1, 0])
    hb_mid = ax_mid.hexbin(
        np.log10(ratio_df["ratio"]),
        ratio_df["survival"],
        gridsize=50,
        cmap="cividis",
        mincnt=10,
    )
    x_curve = np.logspace(
        np.log10(max(ratio_df["ratio"].min(), 1e-4)),
        np.log10(ratio_df["ratio"].max()),
        200,
    )
    s_curve = 1.0 / (1.0 + eta * x_curve)
    ax_mid.plot(np.log10(x_curve), s_curve, color="#ff7f0e", linewidth=2, label=r"theory $S_{\mathrm{surv}}$")
    ax_mid.set_xlabel(r"$\log_{10}(\mathrm{CS}^*/|G|)$")
    ax_mid.set_ylabel(r"$S_{\mathrm{surv}}$ (-)")
    ax_mid.set_title("(b) Survival gate vs. CS*/|G|")
    ax_mid.legend(loc="upper right", frameon=False)
    cb = fig.colorbar(hb_mid, ax=ax_mid, label="counts")
    cb.outline.set_visible(False)

    ax_bot = fig.add_subplot(gs[1, 1])
    i_limit = max(valid_pairs["I_indicator"].quantile(0.995), 1.0)
    trimmed_pairs = valid_pairs[valid_pairs["I_indicator"] <= i_limit]
    if not trimmed_pairs.empty:
        i_corr = trimmed_pairs["I_indicator"].corr(trimmed_pairs["delta_mass"])
        q1 = trimmed_pairs["I_indicator"].quantile(0.25)
        q3 = trimmed_pairs["I_indicator"].quantile(0.75)
        low_med = trimmed_pairs.loc[trimmed_pairs["I_indicator"] <= q1, "delta_mass"].median()
        high_med = trimmed_pairs.loc[trimmed_pairs["I_indicator"] >= q3, "delta_mass"].median()
        delta_shift = high_med - low_med
        z = np.polyfit(trimmed_pairs["I_indicator"], trimmed_pairs["delta_mass"], 2)
        x_fit = np.linspace(0, trimmed_pairs["I_indicator"].max(), 200)
        y_fit = np.polyval(z, x_fit)
    else:
        i_corr = np.nan
        delta_shift = np.nan
        x_fit = y_fit = np.array([])
    hb = ax_bot.hexbin(
        trimmed_pairs["I_indicator"],
        trimmed_pairs["delta_mass"],
        gridsize=60,
        cmap="plasma",
        mincnt=5,
    )
    ax_bot.set_xlim(0, trimmed_pairs["I_indicator"].max() * 1.05 if not trimmed_pairs.empty else 1)
    if x_fit.size:
        ax_bot.plot(x_fit, y_fit, color="#ff7f0e", linewidth=2, label="quadratic fit")
    ax_bot.set_xlabel(r"$I(t)$")
    ax_bot.set_ylabel(r"$\Delta M_{10\mathrm{s}}$ ($\mu$g m$^{-3}$)")
    ax_bot.set_title("(c) Formation-survival driver vs. mass growth")
    if not np.isnan(i_corr):
        ax_bot.annotate(
            f"Pearson r = {i_corr:.2f}\nΔM(Q3)-ΔM(Q1) = {delta_shift:.3f}",
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
    cb2 = fig.colorbar(hb, ax=ax_bot, label="counts")
    cb2.outline.set_visible(False)

    save_dual(fig, "Fig05_Japp_Surv")
    plt.close(fig)

    print("Grid search top 5:")
    print(result_df.head(5))
    print("Best parameters:", best_params)


if __name__ == "__main__":
    main()
