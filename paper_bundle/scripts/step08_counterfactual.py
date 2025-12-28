from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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


def load_data() -> pd.DataFrame:
    """Load merged dataset for counterfactual analysis.

    This function removes duplicate label columns prior to join (e.g.,
    "place") to avoid pandas overlap errors on Windows/Python 3.13.
    """
    base = pd.read_parquet("intermediate/step01_clean.parquet")
    base["Time"] = pd.to_datetime(base["Time"])
    base = base.set_index("Time")
    growth = pd.read_parquet("intermediate/step04_growth_metrics.parquet")
    japp = pd.read_parquet("intermediate/step05_japp_survival.parquet")
    growth = growth.drop(columns=["place"], errors="ignore")
    japp = japp.drop(columns=["place"], errors="ignore")
    return base.join(growth, how="inner").join(japp, how="inner").dropna()


def load_sde_params() -> Dict[str, float]:
    params = pd.read_csv("tables/Table03_SDE_Params.csv")
    lookup = {}
    for _, row in params.iterrows():
        lookup[row["target"]] = row
    return {
        "mu_intercept": lookup["mu"]["intercept"],
        "mu_slope": lookup["mu"]["slope"],
        "sigma_intercept": lookup["sigma"]["intercept"],
        "sigma_slope": lookup["sigma"]["slope"],
    }


def fit_stage_models(df: pd.DataFrame, weights: pd.Series) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, sm.regression.linear_model.RegressionResultsWrapper, callable]:
    exog_stage1 = sm.add_constant(df["temperature_c"])
    stage1 = sm.WLS(df["bvocs"], exog_stage1, weights=weights).fit()
    bvocs_hat = stage1.predict(sm.add_constant(df["temperature_c"]))

    stage2_features = pd.DataFrame(
        {
            "bvocs_hat": bvocs_hat,
            "NOx": df["NOx"],
            "O3": df["O3"],
            "SO2": df["SO2"],
            "rh_pct": df["rh_pct"],
            "rad_w_m2": df["rad_w_m2"],
            "CS_relative": df["CS_relative"],
        }
    ).astype(float)
    stage2 = sm.WLS(df["M_1_20"], sm.add_constant(stage2_features), weights=weights).fit()
    resid = (df["M_1_20"] - stage2.predict(sm.add_constant(stage2_features))).clip(lower=0)

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(df["I_indicator"], resid)

    def predict(temp_series: pd.Series, i_series: pd.Series) -> pd.Series:
        bvocs_hat_cf = stage1.predict(sm.add_constant(temp_series))
        exog_cf = pd.DataFrame(
            {
                "bvocs_hat": bvocs_hat_cf,
                "NOx": df["NOx"],
                "O3": df["O3"],
                "SO2": df["SO2"],
                "rh_pct": df["rh_pct"],
                "rad_w_m2": df["rad_w_m2"],
                "CS_relative": df["CS_relative"],
            }
        ).astype(float)
        linear = stage2.predict(sm.add_constant(exog_cf))
        return linear + iso.transform(i_series)

    return stage1, stage2, predict


def main() -> None:
    df = load_data()
    sde_params = load_sde_params()
    sigma = (sde_params["sigma_intercept"] + sde_params["sigma_slope"] * df["temperature_c"]).clip(lower=0.5)
    weights = 1.0 / (sigma**2)
    _, _, predictor = fit_stage_models(df, weights)

    day_mask = (df.index.hour >= 6) & (df.index.hour < 18)
    df["daypart"] = np.where(day_mask, "day", "night")

    deltas = [0.0, 1.5, 2.0, 3.0]
    scenarios = {}
    for delta in deltas:
        temp_cf = df["temperature_c"] + delta
        scenarios[f"+{delta:.1f}C"] = predictor(temp_cf, df["I_indicator"])

    stats_rows = []
    baseline = scenarios["+0.0C"]
    thresholds = baseline.groupby(df["daypart"]).apply(lambda s: s.mean() + 2 * s.std())

    for name, series in scenarios.items():
        for part in ["day", "night"]:
            mask = df["daypart"] == part
            subset = series[mask]
            thr = thresholds.loc[part]
            exceed_prob = (subset >= thr).mean()
            stats_rows.append(
                {
                    "scenario": name,
                    "daypart": part,
                    "mean": subset.mean(),
                    "variance": subset.var(),
                    "exceedance_prob": exceed_prob,
                }
            )

    stats_table = pd.DataFrame(stats_rows)
    stats_table.to_csv("tables/Table08_ExceedProb.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    scenario_order = list(scenarios.keys())
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for idx, part in enumerate(["day", "night"]):
        ax = axes[0, idx]
        for color, name in zip(colors, scenario_order):
            subset = scenarios[name][df["daypart"] == part]
            sns.kdeplot(subset, ax=ax, label=name if idx == 0 else None, color=color, linewidth=1.2, fill=True, alpha=0.15)
        ax.set_title(f"({'a' if idx == 0 else 'b'}) {part.capitalize()} distribution")
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3, linewidth=0.8)
        if idx == 0:
            ax.legend(frameon=False)

    mean_table = stats_table.pivot(index="scenario", columns="daypart", values="mean")
    var_table = stats_table.pivot(index="scenario", columns="daypart", values="variance")
    prob_table = stats_table.pivot(index="scenario", columns="daypart", values="exceedance_prob")
    x = np.arange(len(scenario_order))
    width = 0.18

    ax_mean = axes[1, 0]
    for j, part in enumerate(["day", "night"]):
        mean_vals = mean_table[part].reindex(scenario_order)
        std_vals = np.sqrt(var_table[part].reindex(scenario_order))
        ax_mean.bar(x + (j - 0.5) * width, mean_vals, width=width, color=colors[j], label=f"{part} mean")
        ax_mean.errorbar(x + (j - 0.5) * width, mean_vals, yerr=std_vals, fmt="none", ecolor="black", capsize=3)
    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels(scenario_order)
    ax_mean.set_ylabel("Mean ± sd (ug m$^{-3}$)")
    ax_mean.set_title("(c) Mean response by scenario")
    ax_mean.grid(alpha=0.3, linewidth=0.8, axis="y")
    ax_mean.legend(frameon=False)

    ax_prob = axes[1, 1]
    for j, part in enumerate(["day", "night"]):
        probs = prob_table[part].reindex(scenario_order)
        ax_prob.bar(x + (j - 0.5) * width, probs * 100, width=width, color=colors[j + 2], label=f"{part} prob")
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(scenario_order)
    ax_prob.set_ylabel("Exceedance probability (%)")
    ax_prob.set_title("(d) P(M >= mu+2sigma)")
    ax_prob.grid(alpha=0.3, linewidth=0.8, axis="y")
    ax_prob.legend(frameon=False)

    fig.subplots_adjust(hspace=0.35)
    fig.suptitle("Counterfactual warming scenarios (temperature perturbations only)")
    save_dual(fig, "Fig08_Counterfactual")
    plt.close(fig)

    print("Counterfactual stats head:")
    print(stats_table.head(6))


if __name__ == "__main__":
    main()
