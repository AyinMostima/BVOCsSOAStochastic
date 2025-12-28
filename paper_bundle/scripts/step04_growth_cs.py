from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow import core  # noqa: E402
from src.workflow.config import default_config  # noqa: E402

HF_RULE = "10s"
HF_SECONDS = int(pd.to_timedelta(HF_RULE).total_seconds())
HF_FREQ_MIN = HF_SECONDS / 60.0
SITE_COLORS = {
    "CM": {"d_eff": "#004c6d", "growth": "#f28e2b"},
    "JH": {"d_eff": "#b31b1b", "growth": "#59a14f"},
}
SINGLE_SERIES_COLORS = {"d_eff": "#1f77b4", "growth": "#ff7f0e"}

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


def build_growth_metrics(
    df: pd.DataFrame,
    number_cols: list[str],
    cfg,
    freq_minutes: float,
) -> pd.DataFrame:
    d_eff = core.compute_effective_diameter(df["M_1_20"], df["N_1_20"], cfg.particle_density_g_cm3)
    g_rate = core.compute_growth_rate(d_eff, freq_minutes=freq_minutes)
    g_abs = g_rate.abs().clip(lower=1e-5).rename("G_abs")
    cs_star = core.compute_condensation_sink(
        df[number_cols],
        df["temperature_c"],
        cfg.cs_diffusivity_m2_s,
        cfg.mean_free_path_nm,
        cfg.accommodation_coefficient,
    )
    cs_relative = (cs_star / cs_star.median()).rename("CS_relative")
    ratio = (cs_star / g_abs).replace([np.inf, -np.inf], np.nan).rename("CS_over_G")
    out = pd.concat([d_eff, g_rate, g_abs, cs_star, cs_relative, ratio], axis=1)
    if "place" in df.columns:
        out["place"] = df["place"]
    return out


def main() -> None:
    cfg = default_config()
    df = pd.read_parquet("intermediate/step01_clean.parquet")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")

    number_cols = [col for col in df.columns if col.startswith("C") and col.endswith("um")]

    base_metrics = build_growth_metrics(df, number_cols, cfg, freq_minutes=1.0)
    base_metrics.to_parquet("intermediate/step04_growth_metrics.parquet")

    hf_df = resample_high_freq(df, HF_RULE, cfg)
    hf_metrics = build_growth_metrics(hf_df, number_cols, cfg, freq_minutes=HF_FREQ_MIN)
    hf_metrics.to_parquet("intermediate/step04_growth_metrics_hf.parquet")

    window = slice("2022-08-06", "2022-08-09")
    window_data = hf_metrics.loc[window].copy()
    if window_data.empty:
        window_data = hf_metrics.copy()
    window_data = window_data.dropna(subset=["d_eff_nm", "G_nm_s"], how="all")
    if window_data.empty:
        window_data = hf_metrics.dropna(subset=["d_eff_nm", "G_nm_s"], how="all")
    reference_time = window_data.index.min()
    window_data = window_data.assign(
        elapsed_hours=(window_data.index - reference_time).total_seconds() / 3600.0
    )

    valid_ratio = hf_metrics["CS_over_G"].dropna()
    valid_ratio = valid_ratio[valid_ratio > 0]
    log_ratio = np.log10(valid_ratio)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts2 = ax_ts.twinx()
    legend_entries: list[tuple[str, str, str]] = []
    if "place" in window_data.columns and window_data["place"].notna().any():
        for place, sub in window_data.groupby("place"):
            if sub.empty:
                continue
            colors = SITE_COLORS.get(place, SINGLE_SERIES_COLORS)
            sub_d = sub.dropna(subset=["d_eff_nm"])
            if not sub_d.empty:
                d_hours = sub_d["elapsed_hours"]
                d_vals = sub_d["d_eff_nm"].copy()
                gap_mask = d_hours.diff().gt(0.5)
                d_vals.loc[gap_mask] = np.nan
                (line_d_eff,) = ax_ts.plot(
                    d_hours,
                    d_vals,
                    color=colors["d_eff"],
                    label=f"d_eff {place}",
                    zorder=2,
                )
            legend_entries.append((rf"$d_{{\mathrm{{eff}}}}$ ({place})", colors["d_eff"], "-"))
            sub_g = sub.dropna(subset=["G_nm_s"])
            if not sub_g.empty:
                g_hours = sub_g["elapsed_hours"]
                g_vals = sub_g["G_nm_s"].copy()
                g_vals.loc[g_hours.diff().gt(0.5)] = np.nan
                (line_g,) = ax_ts2.plot(
                    g_hours,
                    g_vals,
                    linestyle="--",
                    color=colors["growth"],
                    alpha=0.15,
                    linewidth=0.8,
                    zorder=1,
                    label=f"G {place}",
                )
            legend_entries.append((rf"$G$ ({place})", colors["growth"], "--"))
    else:
        sub_d = window_data.dropna(subset=["d_eff_nm"])
        if not sub_d.empty:
            d_hours = sub_d["elapsed_hours"]
            d_vals = sub_d["d_eff_nm"].copy()
            d_vals.loc[d_hours.diff().gt(0.5)] = np.nan
            line_d_eff = ax_ts.plot(
                d_hours,
                d_vals,
                color=SINGLE_SERIES_COLORS["d_eff"],
                label="d_eff",
                zorder=2,
            )[0]
            legend_entries.append((r"$d_{\mathrm{eff}}$", SINGLE_SERIES_COLORS["d_eff"], "-"))
        sub_g = window_data.dropna(subset=["G_nm_s"])
        if not sub_g.empty:
            g_hours = sub_g["elapsed_hours"]
            g_vals = sub_g["G_nm_s"].copy()
            g_vals.loc[g_hours.diff().gt(0.5)] = np.nan
            line_g = ax_ts2.plot(
                g_hours,
                g_vals,
                linestyle="--",
                color=SINGLE_SERIES_COLORS["growth"],
                alpha=0.15,
                linewidth=0.8,
                zorder=1,
                label="G",
            )[0]
            legend_entries.append((r"$G$", SINGLE_SERIES_COLORS["growth"], "--"))
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
        ax_ts.legend(proxies, labels, frameon=False, loc="lower left", ncols=2)
    elapsed_hours = window_data["elapsed_hours"]
    ax_ts.set_ylabel(r"$d_{\mathrm{eff}}$ (nm)")
    ax_ts2.set_ylabel(r"$G$ (nm s$^{-1}$)", color="#aa6c39")
    ax_ts2.tick_params(axis="y", labelcolor="#aa6c39")
    if not window_data.empty:
        ax_ts.set_xlim(0, elapsed_hours.max())
        d_bounds = window_data["d_eff_nm"].quantile([0.02, 0.98])
        g_bounds = window_data["G_nm_s"].quantile([0.02, 0.98])
        ax_ts.set_ylim(d_bounds.iloc[0] - 2, d_bounds.iloc[1] + 2)
        ax_ts2.set_ylim(g_bounds.iloc[0] * 1.1, g_bounds.iloc[1] * 1.1)
    ax_ts.set_xlabel("Elapsed time (h)")
    ax_ts.set_title("(a) High-frequency d_eff and G (10 s grid)")
    ax_ts.grid(alpha=0.3, linewidth=0.8)

    ax_hex = fig.add_subplot(gs[0, 1])
    valid_cs = hf_metrics["CS_star"].clip(lower=1e-5)
    g_abs = hf_metrics["G_abs"]
    log_cs = np.log10(valid_cs)
    log_g = np.log10(g_abs)
    valid_mask = log_cs >= 0.0
    log_cs = log_cs[valid_mask]
    log_g = log_g[valid_mask]
    hb = ax_hex.hexbin(
        log_cs,
        log_g,
        gridsize=60,
        cmap="cividis",
        mincnt=5,
        bins="log",
    )
    ratios_to_plot = [1, 10, 100, 1000]
    x_vals = np.linspace(log_cs.min(), log_cs.max(), 200)
    for r in ratios_to_plot:
        ax_hex.plot(x_vals, x_vals - np.log10(r), linestyle="--", linewidth=1, label=f"CS*/|G|={r}")
    ax_hex.set_xlabel(r"$\log_{10}(\mathrm{CS}^*)$ (s$^{-1}$)")
    ax_hex.set_ylabel(r"$\log_{10}(|G|)$ (nm s$^{-1}$)")
    ax_hex.set_title("(b) Transition regime at 10 s cadence")
    ax_hex.legend(loc="upper left", fontsize=8, frameon=False)
    cb_hex = fig.colorbar(hb, ax=ax_hex, label="counts")
    cb_hex.outline.set_visible(False)

    ax_ratio = fig.add_subplot(gs[1, 0])
    d_eff_match = hf_metrics.loc[valid_ratio.index, "d_eff_nm"]
    plot_ratio = log_ratio
    plot_d_eff = d_eff_match
    hb2 = ax_ratio.hexbin(
        plot_ratio,
        plot_d_eff,
        gridsize=60,
        cmap="inferno",
        mincnt=5,
        bins="log",
    )
    ax_ratio.set_xlabel(r"$\log_{10}(\mathrm{CS}^*/|G|)$")
    ax_ratio.set_ylabel(r"$d_{\mathrm{eff}}$ (nm)")
    ax_ratio.set_title("(c) d_eff vs. CS*/|G| (10 s)")
    r_corr = np.corrcoef(plot_ratio, plot_d_eff)[0, 1]
    ax_ratio.annotate(f"r = {r_corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction")
    cb_ratio = fig.colorbar(hb2, ax=ax_ratio, label="counts")
    cb_ratio.outline.set_visible(False)

    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.hist(
        log_ratio,
        bins=40,
        color="#4c72b0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )
    q1, q2, q3 = np.percentile(log_ratio, [25, 50, 75])
    ax_hist.axvline(q1, color="#d95f02", linestyle="--")
    ax_hist.axvline(q3, color="#d95f02", linestyle="--")
    ax_hist.axvline(q2, color="#2ca02c", linestyle="-.")
    ax_hist.set_xlabel(r"$\log_{10}(\mathrm{CS}^*/|G|)$")
    ax_hist.set_ylabel("counts")
    ax_hist.set_title("(d) CS*/|G| distribution (10 s grid)")

    fig.suptitle("Effective diameter, growth, and condensation sink diagnostics", fontsize=13)

    save_dual(fig, "Fig04_G_and_CS")
    plt.close(fig)

    table = pd.DataFrame(
        [
            {"parameter": "particle_density", "value": cfg.particle_density_g_cm3, "unit": "g cm^-3"},
            {"parameter": "vapor_diffusivity", "value": cfg.cs_diffusivity_m2_s, "unit": "m^2 s^-1"},
            {"parameter": "mean_d_eff", "value": hf_metrics["d_eff_nm"].mean(), "unit": "nm"},
            {"parameter": "mean_growth_rate", "value": hf_metrics["G_nm_s"].mean(), "unit": "nm s^-1"},
            {"parameter": "mean_CS", "value": hf_metrics["CS_star"].mean(), "unit": "s^-1"},
        ]
    )
    table.to_csv("tables/Table04_Params.csv", index=False)

    print("Step04 head(3):")
    print(hf_metrics.head(3))
    print("Step04 tail(2):")
    print(hf_metrics.tail(2))


if __name__ == "__main__":
    main()
