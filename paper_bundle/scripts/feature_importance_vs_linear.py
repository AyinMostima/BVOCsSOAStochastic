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

from src.workflow.modeling_framework import _build_env_features, load_cached_results, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare linear effect sizes vs SHAP feature importance.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"), help="Directory to save tables.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on cached rows for quick runs.")
    return parser.parse_args()


def _linear_effects(cat1_params: pd.DataFrame, df_place: pd.DataFrame) -> pd.DataFrame:
    env = _build_env_features(df_place)
    effects: List[Dict[str, object]] = []
    params = cat1_params.copy()
    if params["ModelID"].dtype != object:
        params["ModelID"] = params["ModelID"].astype(str)
    params_model2 = params[(params["ModelID"] == "2") & (params["Place"] == df_place["place"].iloc[0])]
    mean_env = env.mul(df_place["bvoc_mu_hat"], axis=0).mean()
    for _, row in params_model2.iterrows():
        if row["Parameter"] not in mean_env:
            continue
        effects.append(
            {
                "Feature": row["Parameter"],
                "ScaledEffect": row["Estimate"] * mean_env[row["Parameter"]],
                "Place": row["Place"],
            }
        )
    return pd.DataFrame(effects)


def _load_shap_importance(tables_dir: Path) -> pd.DataFrame:
    shap_path = tables_dir / "Table_Mtheta_SHAP_importance.csv"
    if not shap_path.exists():
        fallback = Path("tables") / "Table_Mtheta_SHAP_importance.csv"
        shap_path = fallback
    if not shap_path.exists():
        raise FileNotFoundError("Table_Mtheta_SHAP_importance.csv not found; run analysis_Mtheta_SHAP.py first.")
    return pd.read_csv(shap_path)


def main() -> None:
    args = parse_args()
    set_plot_style()
    df_sde, cat1, cat2, ml_outputs, labels_cfg = load_cached_results()
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)
    cat1_params = cat1.get("params", pd.DataFrame())
    if cat1_params.empty:
        raise RuntimeError("No linear parameters found; run Category I first.")
    shap_df = _load_shap_importance(args.tables_dir)

    effects_all: List[pd.DataFrame] = []
    places = sorted(df_sde["place"].dropna().unique())
    for place in places:
        df_place = df_sde[df_sde["place"] == place]
        if df_place.empty:
            continue
        eff = _linear_effects(cat1_params, df_place)
        effects_all.append(eff)
    effects_df = pd.concat(effects_all, ignore_index=True) if effects_all else pd.DataFrame()
    tables_dir = args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not effects_df.empty:
        out_linear = tables_dir / "Table_Mtheta_linear_effects.csv"
        effects_df.to_csv(out_linear, index=False)
    else:
        print("No linear effects computed; ensure Category I Model 2 params exist.")
        return
    if not shap_df.empty:
        merged_rows: List[Dict[str, object]] = []
        for place in places:
            shap_place = shap_df[shap_df["Place"] == place]
            eff_place = effects_df[effects_df["Place"] == place]
            for _, row in shap_place.iterrows():
                merged_rows.append(
                    {
                        "Place": place,
                        "SHAP_feature": row["feature"],
                        "SHAP_abs_mean": row["abs_mean_shap"],
                    }
                )
            for _, row in eff_place.iterrows():
                merged_rows.append(
                    {
                        "Place": place,
                        "Linear_feature": row["Feature"],
                        "Linear_scaled_effect": row["ScaledEffect"],
                    }
                )
        pd.DataFrame(merged_rows).to_csv(tables_dir / "Table_Mtheta_linear_vs_SHAP.csv", index=False)

    # Plot: for each place, two panels (Linear vs SHAP)
    for place in places:
        shap_place = shap_df[shap_df["Place"] == place]
        eff_place = effects_df[effects_df["Place"] == place]
        if shap_place.empty or eff_place.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
        eff_sorted = eff_place.sort_values("ScaledEffect")
        axes[0].barh(eff_sorted["Feature"], eff_sorted["ScaledEffect"], color="#4e79a7")
        axes[0].axvline(0, color="black", linewidth=1.0)
        axes[0].set_title(f"{place} Linear effects (Model 2)")
        axes[0].set_xlabel("Coefficient × mean(feature)")

        shap_sorted = shap_place.sort_values("abs_mean_shap", ascending=True)
        axes[1].barh(shap_sorted["feature"], shap_sorted["abs_mean_shap"], color="#d95f02")
        axes[1].set_title(f"{place} SHAP importance")
        axes[1].set_xlabel("Mean |SHAP|")
        fig.suptitle(f"Feature importance vs linear effects ({place})")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        stem = figures_dir / f"SI_Fig_feature_importance_vs_linear_effect_{place}"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.svg", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("Feature importance vs linear effect comparison complete.")


if __name__ == "__main__":
    main()
