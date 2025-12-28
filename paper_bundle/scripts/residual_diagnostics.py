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
    sys.path.append(str(ROOT))

from src.workflow.modeling_framework import load_cached_results, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual diagnostics for linear+CS and ML models.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"), help="Directory to save tables.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick runs.")
    return parser.parse_args()


def _get_best_ml_pred(ml_outputs: Dict[str, object], place: str):
    best = ml_outputs.get("best_predictions", {})
    entry = best.get((place, "Y1"))
    if entry is None:
        return None
    return entry.get("soa_pred")


def _collect_residuals(df: pd.DataFrame, cat1: Dict[str, object], cat2: Dict[str, object], ml_outputs: Dict[str, object]):
    rows: List[Dict[str, float | str]] = []
    diag = cat2.get("diag", {})
    for place, sub in df.groupby("place"):
        obs = sub["SOA"]
        lin_pred = cat1.get("predictions", {}).get((place, "2"))
        ml_pred = _get_best_ml_pred(ml_outputs, place)
        cs_diag = diag.get((place, "2_cs"))
        cs_pred = cs_diag.get("soa_pred_cs") if isinstance(cs_diag, dict) else None
        cs_series = cs_diag.get("cs") if isinstance(cs_diag, dict) else None
        temp = sub.get("temperature_c")
        for t in sub.index:
            row = {"Place": place, "Time": t, "SOA_obs": obs.loc[t] if t in obs.index else np.nan}
            row["pred_lin2"] = lin_pred.loc[t] if lin_pred is not None and t in lin_pred.index else np.nan
            row["pred_cs"] = cs_pred.loc[t] if cs_pred is not None and t in cs_pred.index else np.nan
            row["pred_ml"] = ml_pred.loc[t] if ml_pred is not None and t in ml_pred.index else np.nan
            row["CS"] = cs_series.loc[t] if cs_series is not None and t in cs_series.index else np.nan
            row["temperature_c"] = temp.loc[t] if temp is not None and t in temp.index else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_residuals(place: str, df_res: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_palette("colorblind")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    models = [
        ("pred_lin2", "Linear Model 2"),
        ("pred_cs", "Linear Model 2 + CS"),
        ("pred_ml", "Best ML (Y1)"),
    ]
    # panel (a): residual vs predicted (CS model if available, else lin2)
    pred_col = "pred_cs" if df_res["pred_cs"].notna().any() else "pred_lin2"
    mask = df_res["SOA_obs"].notna() & df_res[pred_col].notna()
    axes[0, 0].scatter(df_res.loc[mask, pred_col], df_res.loc[mask, "SOA_obs"] - df_res.loc[mask, pred_col], s=10, alpha=0.4)
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xlabel(f"Predicted SOA ({pred_col})")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].set_title(f"(a) Residual vs predicted ({place})")
    axes[0, 0].grid(alpha=0.3, linewidth=0.8)

    # panel (b): diurnal cycle of residual (CS model if available)
    df_res = df_res.copy()
    df_res["hour"] = df_res["Time"].dt.hour
    if df_res["pred_cs"].notna().any():
        res_series = df_res["SOA_obs"] - df_res["pred_cs"]
    else:
        res_series = df_res["SOA_obs"] - df_res["pred_lin2"]
    diurnal = res_series.groupby(df_res["hour"]).mean()
    axes[0, 1].plot(diurnal.index, diurnal.values, marker="o")
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xlabel("Hour (local)")
    axes[0, 1].set_ylabel("Residual mean")
    axes[0, 1].set_title("(b) Diurnal residual")
    axes[0, 1].grid(alpha=0.3, linewidth=0.8)

    # panel (c): residual vs temperature
    if "temperature_c" in df_res.columns:
        axes[1, 0].scatter(df_res["temperature_c"], res_series, s=8, alpha=0.35)
        axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1.0)
        axes[1, 0].set_xlabel("Temperature (C)")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_title("(c) Residual vs Temperature")
        axes[1, 0].grid(alpha=0.3, linewidth=0.8)
    else:
        axes[1, 0].axis("off")

    # panel (d): residual vs CS
    if df_res["CS"].notna().any():
        axes[1, 1].scatter(df_res["CS"], res_series, s=8, alpha=0.35)
        axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1.0)
        axes[1, 1].set_xlabel("CS (s^-1)")
        axes[1, 1].set_ylabel("Residual")
        axes[1, 1].set_title("(d) Residual vs CS")
        axes[1, 1].set_xscale("log")
        axes[1, 1].grid(alpha=0.3, linewidth=0.8, which="both")
    else:
        axes[1, 1].axis("off")

    fig.suptitle(f"Residual diagnostics ({place})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / f"Fig_residual_diagnostics_{place}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_plot_style()
    df_sde, cat1, cat2, ml_outputs, labels_cfg = load_cached_results()
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)
    df_res = _collect_residuals(df_sde, cat1, cat2, ml_outputs)
    if df_res.empty:
        raise RuntimeError("No residual data collected; ensure cached predictions exist.")
    tables_dir = args.tables_dir
    figures_dir = args.figures_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: List[Dict[str, object]] = []
    for place, sub in df_res.groupby("Place"):
        pred_lin = sub["pred_lin2"]
        pred_cs = sub["pred_cs"]
        pred_ml = sub["pred_ml"]
        obs = sub["SOA_obs"]
        for name, series in [("lin2", pred_lin), ("cs", pred_cs), ("ml", pred_ml)]:
            mask = obs.notna() & series.notna()
            if mask.sum() == 0:
                continue
            resid = obs[mask] - series[mask]
            stats_rows.append(
                {
                    "Place": place,
                    "Model": name,
                    "n": int(mask.sum()),
                    "Residual_mean": float(resid.mean()),
                    "Residual_std": float(resid.std()),
                    "Residual_rmse": float(np.sqrt(np.mean(resid**2))),
                }
            )
        _plot_residuals(place, sub, figures_dir)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(tables_dir / "Table_residuals_stats.csv", index=False)
    print("Residual diagnostics complete.")


if __name__ == "__main__":
    main()
