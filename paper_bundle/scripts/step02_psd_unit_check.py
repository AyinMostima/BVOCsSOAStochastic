from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
sns.set_palette("colorblind")


def parse_diameter(col: str) -> float:
    return float(col.replace("C", "").replace("um", "")) * 1000


def save_dual(fig: plt.Figure, stem: str) -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"figures/{stem}.svg", bbox_inches="tight")
    fig.savefig(f"figures/{stem}.png", dpi=300, bbox_inches="tight")


def build_psd(df: pd.DataFrame, cols: list[str], mask: pd.Series) -> pd.DataFrame:
    subset = df.loc[mask, cols]
    profile = subset.mean(axis=0, skipna=True)
    return pd.DataFrame(
        {
            "diameter_nm": [parse_diameter(col) for col in profile.index],
            "value": profile.values,
        }
    ).sort_values("diameter_nm")


def main() -> None:
    cleaned = pd.read_parquet("intermediate/step01_clean.parquet")
    cleaned["Time"] = pd.to_datetime(cleaned["Time"])
    cleaned = cleaned.set_index("Time")
    bin_meta = pd.read_parquet("intermediate/bin_metadata.parquet")

    mass_cols = [col for col in cleaned.columns if col.endswith("um") and not col.startswith("C")]
    number_cols = [col for col in cleaned.columns if col.startswith("C") and col.endswith("um")]

    temp_quantiles = cleaned["temperature_c"].quantile([0.25, 0.75])
    cold_mask = cleaned["temperature_c"] <= temp_quantiles.iloc[0]
    hot_mask = cleaned["temperature_c"] >= temp_quantiles.iloc[1]

    places = cleaned["place"].unique().tolist() if "place" in cleaned.columns else ["ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    proxy_min, proxy_max = 250, 400
    linestyles = {"CM": "-", "JH": "--", "ALL": "-"}
    for ax, cols, ylabel, title in [
        (axes[0], number_cols, "Number concentration (cm$^{-3}$)", "(a) Number PSD"),
        (axes[1], mass_cols, "Mass concentration (ug m$^{-3}$)", "(b) Mass PSD"),
    ]:
        ax.axvspan(proxy_min, proxy_max, color="#fddbc7", alpha=0.4, label="Proxy window (<=0.40 um)")
        for place in places:
            data_p = cleaned if place == "ALL" else cleaned[cleaned["place"] == place]
            cold_p = build_psd(data_p, cols, data_p["temperature_c"] <= data_p["temperature_c"].quantile(0.25))
            hot_p = build_psd(data_p, cols, data_p["temperature_c"] >= data_p["temperature_c"].quantile(0.75))
            ls = linestyles.get(place, "-")
            label_cold = f"Low T ({place})" if place != "ALL" else "Low T"
            label_hot = f"High T ({place})" if place != "ALL" else "High T"
            ax.plot(cold_p["diameter_nm"], cold_p["value"], color="#1b9e77", linewidth=1.2, linestyle=ls, label=label_cold)
            ax.plot(hot_p["diameter_nm"], hot_p["value"], color="#d95f02", linewidth=1.2, linestyle=ls, label=label_hot)
        ax.set_xscale("log")
        if ax is axes[0]:
            ax.set_yscale("log")
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.legend(frameon=False, ncols=2, fontsize=8)
    fig.suptitle("Temperature contrast of particle size distributions (shaded = proxy range)")
    save_dual(fig, "Fig02_PSD_Number")
    save_dual(fig, "Fig02_PSD_Mass")
    plt.close(fig)

    bin_meta["diameter_nm"] = bin_meta["bin"].apply(parse_diameter)
    bin_meta["unit"] = np.where(bin_meta["type"] == "mass", "ug m^-3", "cm^-3")
    bin_meta["small_bin_flag"] = bin_meta["diameter_um"] <= 0.40
    bin_meta.to_csv("tables/Table02_BinMeta.csv", index=False)

    print("PSD plots saved for places:", places)


if __name__ == "__main__":
    main()
