from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

# Use Agg backend to avoid GUI issues on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow import modeling_framework as mf  # noqa: E402
from src.workflow.config import default_config  # noqa: E402
from src.workflow.explainers import (  # noqa: E402
    compute_shap_summary,
    fit_tree_model,
    plot_pdp_1d,
    plot_pdp_2d,
    plot_shap_interaction_heatmap,
)


def _build_dataset(
    df: pd.DataFrame, cat1: Dict[str, object], place: str, model_id: str = "2"
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    base_pred = cat1.get("predictions", {}).get((place, model_id))
    if base_pred is None:
        raise KeyError(f"Missing base prediction for {place}, model {model_id}")
    base_pred = base_pred.reindex(df.index)
    y = df["SOA"]
    mask = (y > 0) & (base_pred > 0)
    target = np.log(y[mask] / base_pred[mask])
    features = pd.DataFrame(
        {
            "O3": df.loc[mask, "O3"],
            "NOx": df.loc[mask, "NOx"],
            "SO2": df.loc[mask, "SO2"],
            "RH": df.loc[mask, "rh_pct"],
            "J": df.loc[mask, "rad_w_m2"],
            "Temperature": df.loc[mask, "temperature_c"],
        }
    )
    groups = features.index.floor("D")
    return features, target, groups


def _pdp_effect_summary(model, X: pd.DataFrame, feature: str, grid_resolution: int) -> Dict[str, object]:
    pdp_res = partial_dependence(model, X, [feature], grid_resolution=grid_resolution, kind="average")
    grid = pdp_res["grid_values"][0]
    values = pdp_res["average"][0]
    diffs = np.diff(values)
    trend = "mixed"
    turning = np.nan
    if np.all(diffs >= 0):
        trend = "monotonic_positive"
    elif np.all(diffs <= 0):
        trend = "monotonic_negative"
    else:
        sign_change = np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0]
        if sign_change.size > 0:
            idx = sign_change[0] + 1
            turning = float(grid[idx])
            trend = "sign_flip"
    amplitude = float(np.percentile(values, 90) - np.percentile(values, 10))
    return {
        "feature": feature,
        "trend": trend,
        "turning_point": turning,
        "amplitude_10_90": amplitude,
        "pdp_min": float(np.min(values)),
        "pdp_max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP/PDP analysis for chemical modulation kernel.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"), help="Directory to save tables.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional fraction to downsample rows.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on total rows.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--feature-display", type=int, default=12, help="Max features to display in SHAP beeswarm.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()
    seed = cfg.random_seed if args.seed is None else args.seed
    mf.set_plot_style()
    df_sde, cat1, cat2, ml_outputs, labels = mf.load_cached_results()
    if args.sample_frac:
        df_sde = df_sde.sample(frac=args.sample_frac, random_state=seed)
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)
    places = sorted(df_sde["place"].dropna().unique())
    metrics_rows: List[Dict[str, object]] = []
    shap_rows: List[Dict[str, object]] = []
    runmeta_rows: List[Dict[str, object]] = []
    pdp_effect_rows: List[Dict[str, object]] = []

    for place in places:
        df_place = df_sde[df_sde["place"] == place]
        if df_place.empty:
            continue
        try:
            X, y, groups = _build_dataset(df_place, cat1, place, model_id="2")
        except KeyError:
            continue
        res = fit_tree_model(
            X,
            y,
            model_type="xgb",
            groups=groups,
            random_state=seed,
            model_params=cfg.xgb_params,
        )
        res_rf = fit_tree_model(
            X,
            y,
            model_type="rf",
            groups=groups,
            random_state=seed,
            model_params=cfg.rf_params,
        )
        for tag, out in [("xgb", res), ("rf", res_rf)]:
            metrics_rows.append(
                {
                    "Place": place,
                    "Model": tag.upper(),
                    **out.metrics,
                }
            )
        shap_df = compute_shap_summary(
            res.model,
            X,
            stem=f"Fig_Mtheta_SHAP_{place}",
            output_dir=args.figures_dir,
            max_points=cfg.shap_max_points,
            feature_display=args.feature_display,
        )
        shap_df["Place"] = place
        shap_rows.append(shap_df)

        plot_shap_interaction_heatmap(
            res.model,
            X,
            stem=f"Fig_Mtheta_SHAP_interactions_{place}",
            output_dir=args.figures_dir,
        )

        pdp_features = ["O3", "NOx", "RH", "J", "Temperature"]
        plot_pdp_1d(
            res.model,
            X,
            pdp_features,
            stem=f"Fig_Mtheta_PDP_{place}",
            output_dir=args.figures_dir,
            grid_resolution=cfg.pdp_grid_resolution_1d,
        )
        for feat in pdp_features:
            pdp_effect_rows.append(
                {
                    "Place": place,
                    **_pdp_effect_summary(res.model, X, feat, cfg.pdp_grid_resolution_1d),
                }
            )
        if place == "CM":
            plot_pdp_1d(
                res.model,
                X,
                pdp_features,
                stem="Fig_Mtheta_PDP_main",
                output_dir=args.figures_dir,
                grid_resolution=cfg.pdp_grid_resolution_1d,
            )
        plot_pdp_2d(
            res.model,
            X,
            feature_pairs=[("O3", "NOx"), ("RH", "J")],
            stem=f"SI_Fig_Mtheta_PDP_2D_{place}",
            output_dir=args.figures_dir,
            grid_resolution=cfg.pdp_grid_resolution_2d,
        )
        for tag, out in [("xgb", res), ("rf", res_rf)]:
            runmeta_rows.append(
                {
                    "Place": place,
                    "Model": tag.upper(),
                    "Params": json.dumps(cfg.xgb_params if tag == "xgb" else cfg.rf_params),
                    "Groups": "GroupKFold_by_day" if groups.nunique() >= 2 else "Holdout_80_20",
                    "n_samples": out.metrics.get("n_samples", np.nan),
                    "n_features": out.metrics.get("n_features", np.nan),
                    "RandomSeed": seed,
                    "PDP_grid_1d": cfg.pdp_grid_resolution_1d,
                    "PDP_grid_2d": cfg.pdp_grid_resolution_2d,
                    "SHAP_max_points": cfg.shap_max_points,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    shap_summary = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    tables_dir = args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(tables_dir / "Table_Mtheta_SHAP_metrics.csv", index=False)
    shap_summary.to_csv(tables_dir / "Table_Mtheta_SHAP_importance.csv", index=False)
    if runmeta_rows:
        pd.DataFrame(runmeta_rows).to_csv(tables_dir / "Table_Mtheta_SHAP_runmeta.csv", index=False)
    if pdp_effect_rows:
        pd.DataFrame(pdp_effect_rows).to_csv(tables_dir / "Table_Mtheta_PDP_effects.csv", index=False)
    if not shap_summary.empty:
        site_compare = (
            shap_summary.groupby(["Place", "feature"])["abs_mean_shap"]
            .mean()
            .reset_index()
            .sort_values(["Place", "abs_mean_shap"], ascending=[True, False])
        )
        site_compare.to_csv(tables_dir / "Table_Mtheta_SHAP_site_compare.csv", index=False)
    print("M_theta SHAP analysis complete.")


if __name__ == "__main__":
    main()
