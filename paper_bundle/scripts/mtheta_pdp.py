from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow.config import default_config
from src.workflow.explainers import fit_tree_model, plot_pdp_1d, plot_pdp_2d
from src.workflow.modeling_framework import load_cached_results, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDP plots for chemical modulation kernel M_theta(X).")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"), help="Directory to save tables.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--clip-quantile", type=float, default=0.99, help="Clip features to [1-q, q] to reduce extremes.")
    parser.add_argument("--feature-display", type=int, default=12, help="Number of features to display in SHAP if needed.")
    return parser.parse_args()


def _build_dataset(df: pd.DataFrame, cat1: Dict[str, object], place: str) -> Tuple[pd.DataFrame, pd.Series]:
    base_pred = cat1.get("predictions", {}).get((place, "2"))
    if base_pred is None:
        raise KeyError(f"Missing base prediction for {place}, model 2.")
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
    return features, target


def _clip_features(X: pd.DataFrame, q: float) -> pd.DataFrame:
    lower = X.quantile(1 - q)
    upper = X.quantile(q)
    return X.clip(lower=lower, upper=upper, axis=1)


def main() -> None:
    args = parse_args()
    cfg = default_config()
    set_plot_style()
    df_sde, cat1, cat2, ml_outputs, labels_cfg = load_cached_results()
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)
    figures_dir = args.figures_dir
    tables_dir = args.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pdp_rows: List[Dict[str, object]] = []
    for place in sorted(df_sde["place"].dropna().unique()):
        df_place = df_sde[df_sde["place"] == place]
        if df_place.empty:
            continue
        try:
            X_raw, y = _build_dataset(df_place, cat1, place)
        except KeyError:
            continue
        X = _clip_features(X_raw, args.clip_quantile)
        model = fit_tree_model(
            X,
            y,
            model_type="xgb",
            groups=X.index.floor("D"),
            random_state=cfg.random_seed,
            model_params=cfg.xgb_params,
        ).model
        # 1D PDP
        plot_pdp_1d(
            model,
            X,
            features=["O3", "NOx", "RH", "J"],
            stem=f"Fig_Mtheta_PDP_1D_{place}",
            output_dir=figures_dir,
            grid_resolution=cfg.pdp_grid_resolution_1d,
        )
        # 2D PDP
        plot_pdp_2d(
            model,
            X,
            feature_pairs=[("O3", "NOx")],
            stem=f"Fig_Mtheta_PDP_2D_O3_NOx_{place}",
            output_dir=figures_dir,
            grid_resolution=cfg.pdp_grid_resolution_2d,
        )
        # Store summary stats for quick reference
        pdp_rows.append({"Place": place, "Model": "XGB", "n_samples": X.shape[0], "n_features": X.shape[1]})
    if pdp_rows:
        pd.DataFrame(pdp_rows).to_csv(tables_dir / "Table_Mtheta_PDP_summary.csv", index=False)
    print("M_theta PDP generation complete.")


if __name__ == "__main__":
    main()
