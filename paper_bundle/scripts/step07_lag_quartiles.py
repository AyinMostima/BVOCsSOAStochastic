from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def load_combined() -> pd.DataFrame:
    """Load merged dataset for lag/quartile analysis.

    Notes (Windows friendly):
    - This function carefully avoids duplicate columns when joining, in
      particular the optional "place" column which may be present in both
      growth metrics and J_app/survival outputs. We drop the duplicate to
      ensure an inner join succeeds deterministically.
    """
    base = pd.read_parquet("intermediate/step01_clean.parquet")
    base["Time"] = pd.to_datetime(base["Time"])
    base = base.set_index("Time")
    growth = pd.read_parquet("intermediate/step04_growth_metrics.parquet")
    japp = pd.read_parquet("intermediate/step05_japp_survival.parquet")
    # Drop duplicate non-numeric labels before join to prevent overlap errors
    growth = growth.drop(columns=["place"], errors="ignore")
    japp = japp.drop(columns=["place"], errors="ignore")
    merged = base.join(growth, how="inner").join(japp, how="inner")
    return merged.dropna()


def compute_lag_corr(indicator: pd.Series, mass: pd.Series, lags: List[int]) -> pd.DataFrame:
    rows = []
    for lag in lags:
        future = mass.shift(-lag)
        delta = future - mass
        aligned = pd.concat([indicator, delta], axis=1).dropna()
        if aligned.empty:
            corr = np.nan
        else:
            corr = aligned.corr().iloc[0, 1]
        rows.append({"lag_min": lag, "corr": corr})
    return pd.DataFrame(rows)


def compute_quartile_boxes(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["CS_star"] / np.maximum(df["G_nm_s"].abs(), 1e-4)
    df = df.assign(cs_g_ratio=ratio)
    precursor_mask = (df["I_indicator"] >= df["I_indicator"].quantile(0.4)) & (
        df["I_indicator"] <= df["I_indicator"].quantile(0.6)
    )
    subset = df.loc[precursor_mask].copy()
    subset["ratio_quartile"] = pd.qcut(subset["cs_g_ratio"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    delta = subset["M_1_20"].shift(-120) - subset["M_1_20"].shift(-60)
    subset["delta_60_120"] = delta
    return subset.dropna(subset=["delta_60_120"])


def main() -> None:
    df = load_combined()
    indicator = df["I_indicator"]
    lags = list(range(0, 241, 30))
    lag_df = compute_lag_corr(indicator, df["M_1_20"], lags)
    lag_df.to_csv("tables/Table07_LagCorr.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    hours = lag_df["lag_min"] / 60.0
    ax.axvspan(1, 3, color="#fddbc7", alpha=0.4, label="1-3 h growth window")
    ax.plot(hours, lag_df["corr"], marker="o", color="#1b9e77")
    peak_row = lag_df.iloc[lag_df["corr"].idxmax()]
    peak_hr = peak_row["lag_min"] / 60.0
    ax.axvline(peak_hr, color="#d95f02", linestyle="--", label=f"Peak {peak_hr:.1f} h")
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel("Corr(I, Delta M)")
    ax.set_title("(a) Lag correlation between indicator and mass growth")
    ax.annotate(
        f"max corr = {peak_row['corr']:.3f}",
        xy=(peak_hr, peak_row["corr"]),
        xytext=(peak_hr + 0.15, peak_row["corr"] - 0.03),
        arrowprops=dict(arrowstyle="->", color="#d95f02"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax.grid(alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False)
    save_dual(fig, "Fig07_LagCorr")
    plt.close(fig)



    quartile_df = compute_quartile_boxes(df)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=quartile_df,
        x="ratio_quartile",
        y="delta_60_120",
        ax=ax2,
        palette="Blues",
        showfliers=False,
    )
    sns.stripplot(
        data=quartile_df,
        x="ratio_quartile",
        y="delta_60_120",
        ax=ax2,
        color="#d95f02",
        alpha=0.4,
        jitter=0.15,
        size=3,
    )
    ax2.set_xlabel("CS/G quartile")
    ax2.set_ylabel("Delta M (60-120 min)")
    ax2.set_title("(b) Delta M conditioned on CS/G quartiles")
    medians = quartile_df.groupby("ratio_quartile")["delta_60_120"].median()
    for i, (label, value) in enumerate(medians.items()):
        ax2.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    save_dual(fig2, "Fig07_Quartiles_Box")
    plt.close(fig2)



    p_rows = []
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i in range(len(labels) - 1):
        a = quartile_df.loc[quartile_df["ratio_quartile"] == labels[i], "delta_60_120"]
        b = quartile_df.loc[quartile_df["ratio_quartile"] == labels[i + 1], "delta_60_120"]
        stat, pval = stats.mannwhitneyu(a, b, alternative="greater")
        p_rows.append({"pair": f"{labels[i]} vs {labels[i+1]}", "statistic": stat, "p_value": pval})
    p_table = pd.DataFrame(p_rows)
    p_table.to_csv("tables/Table07_Tests.csv", index=False)

    print("Lag correlations head(3):")
    print(lag_df.head(3))
    print("Quartile test rows:")
    print(p_table)


if __name__ == "__main__":
    main()
