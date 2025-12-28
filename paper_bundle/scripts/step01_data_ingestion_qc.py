from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow import core  # noqa: E402
from src.workflow.config import default_config  # noqa: E402

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


def save_figure(fig: plt.Figure, name: str) -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{name}.svg"
    png_path = out_dir / f"{name}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


def plot_qc_panels(df: pd.DataFrame) -> None:
    # Choose a representative 48h window from the combined timeline
    window_df = df.loc["2022-08-06":"2022-08-07"].copy()
    if window_df.empty:
        window_df = df.copy()
    # For elapsed hours per site (if present)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False)
    panels = [
        ("bvocs", "(a) BVOCs (ug m$^{-3}$)"),
        ("temperature_c", "(b) Temperature (deg C)"),
        ("rad_w_m2", "(c) Net radiation (W m$^{-2}$)"),
        ("N_1_20", "(d) N$_{1-20}$ (cm$^{-3}$)"),
    ]
    palette = {"CM": "#1b9e77", "JH": "#d95f02"}
    has_place = "place" in df.columns
    for ax, (col, title) in zip(axes.ravel(), panels):
        if has_place:
            for place, sub in window_df.groupby("place"):
                if sub.empty:
                    continue
                elapsed_hours = (sub.index - sub.index[0]).total_seconds() / 3600.0
                ax.plot(elapsed_hours, sub[col], color=palette.get(place, "#7570b3"), label=place)
            ax.legend(frameon=False, loc="upper right")
            missing = df.groupby("place")[col].apply(lambda s: s.isna().mean() * 100)
            miss_text = ", ".join([f"{idx}:{val:.1f}%" for idx, val in missing.items()])
        else:
            res = window_df.resample("5min").mean(numeric_only=True)
            if res.empty:
                continue
            elapsed_hours = (res.index - res.index[0]).total_seconds() / 3600.0
            ax.plot(elapsed_hours, res[col], color="#1b9e77")
            missing = df[col].isna().mean() * 100
            miss_text = f"missing {missing:.1f}%"
        ax.set_title(f"{title}  |  {miss_text}")
        ax.set_ylabel(title.split(")", 1)[1].strip())
        ax.set_xlabel("Elapsed time (h)")
        ax.grid(alpha=0.3, linewidth=0.8)
    fig.suptitle("QC diagnostics across sites (5 min averages)")
    # Prevent title and xlabels from overlapping
    try:
        fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88, bottom=0.1)
    except Exception:
        pass
    save_figure(fig, "Fig01_Data_QC")
    plt.close(fig)


def main() -> None:
    pd.options.mode.copy_on_write = True
    cfg = default_config()
    core.ensure_output_dirs(cfg)
    start = time.perf_counter()

    df, unit_log, bin_meta = core.prepare_master_dataframe(cfg)
    data_dict = core.build_data_dictionary(df)

    out_clean = Path("intermediate/step01_clean.parquet")
    df.reset_index().to_parquet(out_clean, index=False)
    unit_log.to_csv("intermediate/unit_log.csv", index=False)
    bin_meta.to_parquet("intermediate/bin_metadata.parquet", index=False)
    data_dict.to_csv("tables/Table01_DataDictionary.csv", index=False)

    qc_table = (
        pd.DataFrame(
            {
                "field": df.columns,
                "missing_pct": df.isna().mean().values * 100,
                "min": df.min().values,
                "max": df.max().values,
            }
        )
        .sort_values("field")
    )
    qc_table.to_csv("tables/Table01_QCStats.csv", index=False)

    plot_qc_panels(df)

    print(f"Clean data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:15]} ...")
    print("Head(3):")
    print(df.head(3))
    print("Tail(2):")
    print(df.tail(2))

    runtime = time.perf_counter() - start
    summary = {
        "step": "step01",
        "records": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "runtime_seconds": runtime,
        "outputs": {
            "clean_parquet": str(out_clean),
            "unit_log": "intermediate/unit_log.csv",
            "data_dictionary": "tables/Table01_DataDictionary.csv",
            "qc_table": "tables/Table01_QCStats.csv",
            "figure": ["figures/Fig01_Data_QC.svg", "figures/Fig01_Data_QC.png"],
        },
    }
    Path("logs").mkdir(exist_ok=True, parents=True)
    with open("logs/step01_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
