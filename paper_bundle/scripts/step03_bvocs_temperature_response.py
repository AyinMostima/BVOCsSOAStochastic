from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow import core  # noqa: E402
from src.workflow.config import WorkflowConfig, default_config  # noqa: E402

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


def fit_linear_relation(df: pd.DataFrame, target: str) -> dict:
    slope, intercept, r_val, p_val, stderr = stats.linregress(df["temp_bin"], df[target])
    return {
        "target": target,
        "slope": slope,
        "intercept": intercept,
        "r2": r_val**2,
        "p_value": p_val,
        "stderr": stderr,
    }


def main(cfg: WorkflowConfig | None = None) -> None:
    cfg = cfg or default_config()
    data = pd.read_parquet("intermediate/step01_clean.parquet")
    data["Time"] = pd.to_datetime(data["Time"])
    data = data.set_index("Time")

    stats_df = core.estimate_temperature_response(
        data["bvocs"],
        data["temperature_c"],
        bin_width=cfg.temperature_bin_c,
        min_count=cfg.min_samples_per_temp_bin,
    )
    stats_df.to_parquet("intermediate/step03_temp_bins.parquet", index=False)

    # Pooled fit
    mu_fit = fit_linear_relation(stats_df, "mu")
    sigma_fit = fit_linear_relation(stats_df, "sigma")
    param_table = pd.DataFrame([mu_fit, sigma_fit])
    param_table.to_csv("tables/Table03_SDE_Params.csv", index=False)

    # By-site fit (if available)
    site_params = []
    if "place" in data.columns:
        for place, sub in data.groupby("place"):
            sstats = core.estimate_temperature_response(
                sub["bvocs"], sub["temperature_c"], bin_width=cfg.temperature_bin_c, min_count=cfg.min_samples_per_temp_bin
            )
            if sstats.empty:
                continue
            mfit = fit_linear_relation(sstats, "mu"); mfit["place"] = place
            sfit = fit_linear_relation(sstats, "sigma"); sfit["place"] = place
            mfit["target"] = "mu"; sfit["target"] = "sigma"
            site_params.extend([mfit, sfit])
        if site_params:
            pd.DataFrame(site_params).to_csv("tables/Table03_SDE_Params_by_site.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    palette = {"CM": "#1b9e77", "JH": "#d95f02"}
    if "place" in data.columns:
        for place, sub in data.groupby("place"):
            sstats = core.estimate_temperature_response(sub["bvocs"], sub["temperature_c"], bin_width=cfg.temperature_bin_c, min_count=cfg.min_samples_per_temp_bin)
            if sstats.empty:
                continue
            axes[0].scatter(sstats["temp_bin"], sstats["mu"], color=palette.get(place, "#7570b3"), label=f"{place}")
            mf = fit_linear_relation(sstats, "mu")
            axes[0].plot(sstats["temp_bin"], mf["intercept"] + mf["slope"] * sstats["temp_bin"], color=palette.get(place, "#7570b3"))
        axes[0].legend(frameon=False)
    else:
        axes[0].scatter(stats_df["temp_bin"], stats_df["mu"], color="#1b9e77")
        axes[0].plot(stats_df["temp_bin"], mu_fit["intercept"] + mu_fit["slope"] * stats_df["temp_bin"], color="#d95f02")
    axes[0].set_xlabel("Temperature (deg C)")
    axes[0].set_ylabel("mu (ug m$^{-3}$ min$^{-1}$)")
    axes[0].set_title("(a) BVOCs drift vs temperature")
    axes[0].annotate(
        f"slope = {mu_fit['slope']:.2f}\np = {mu_fit['p_value']:.3f}",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    if "place" in data.columns:
        for place, sub in data.groupby("place"):
            sstats = core.estimate_temperature_response(sub["bvocs"], sub["temperature_c"], bin_width=cfg.temperature_bin_c, min_count=cfg.min_samples_per_temp_bin)
            if sstats.empty:
                continue
            axes[1].scatter(sstats["temp_bin"], sstats["sigma"], color=palette.get(place, "#7570b3"), label=f"{place}")
            sf = fit_linear_relation(sstats, "sigma")
            axes[1].plot(sstats["temp_bin"], sf["intercept"] + sf["slope"] * sstats["temp_bin"], color=palette.get(place, "#7570b3"))
        axes[1].legend(frameon=False)
    else:
        axes[1].scatter(stats_df["temp_bin"], stats_df["sigma"], color="#7570b3")
        axes[1].plot(stats_df["temp_bin"], sigma_fit["intercept"] + sigma_fit["slope"] * stats_df["temp_bin"], color="#d95f02")
    axes[1].set_xlabel("Temperature (deg C)")
    axes[1].set_ylabel("sigma (ug m$^{-3}$ min$^{-1}$)")
    axes[1].set_title("(b) BVOCs diffusion vs temperature")
    axes[1].annotate(
        f"slope = {sigma_fit['slope']:.2f}\np = {sigma_fit['p_value']:.3f}",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    hottest_bin = stats_df.loc[stats_df["temp_bin"].idxmax()]
    subset = data.loc[
        (data["temperature_c"] // cfg.temperature_bin_c) * cfg.temperature_bin_c == hottest_bin["temp_bin"],
        "bvocs",
    ]
    osm, osr = stats.probplot(subset.dropna(), dist="norm")
    theoretical, ordered = osm
    slope, intercept, _ = osr
    axes[2].scatter(theoretical, ordered, s=12, color="#1b9e77")
    axes[2].plot(theoretical, slope * theoretical + intercept, color="#d95f02")
    axes[2].set_xlabel("Theoretical quantiles")
    axes[2].set_ylabel("Observed BVOCs")
    axes[2].set_title(f"(c) QQ plot (warm bin {hottest_bin['temp_bin']:.0f} deg C)")

    fig.suptitle("Temperature control on BVOCs moments")
    save_dual(fig, "Fig03_BVOC_T_MeanVar")
    plt.close(fig)

    mu_series = mu_fit["intercept"] + mu_fit["slope"] * data["temperature_c"]
    sigma_series = sigma_fit["intercept"] + sigma_fit["slope"] * data["temperature_c"]
    sigma_series = sigma_series.clip(lower=0.05)

    n_steps = min(720, data.shape[0])
    paths = core.simulate_sde_paths(
        mu_series.iloc[:n_steps],
        sigma_series.iloc[:n_steps],
        steps=n_steps,
        n_paths=200,
        delta_minutes=1.0,
        seed=cfg.random_seed,
        initial_value=float(data["bvocs"].iloc[0]),
    )
    quantiles = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    elapsed_hours = np.arange(n_steps) / 60.0

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(elapsed_hours, quantiles[0], quantiles[2], color="#1b9e77", alpha=0.2, label="MC 10-90%")
    ax.plot(elapsed_hours, quantiles[1], color="#1b9e77", label="MC median")
    ax.plot(elapsed_hours, data["bvocs"].iloc[:n_steps], color="#d95f02", label="Observed")
    ax.set_ylabel("BVOCs (ug m$^{-3}$)")
    ax.set_xlabel("Elapsed time (h)")
    ax.set_title("Monte Carlo BVOCs envelope (coverage 10-90%)")
    ax.grid(alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    save_dual(fig2, "Fig03_BVOC_MC")
    plt.close(fig2)

    print("Table03 head(3):")
    print(stats_df.head(3))
    print("Table03 tail(2):")
    print(stats_df.tail(2))


if __name__ == "__main__":
    main()
