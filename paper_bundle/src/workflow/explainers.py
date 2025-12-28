from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise ImportError("shap is required for SHAP analysis; install via pip install shap") from exc

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]


@dataclass
class ModelResult:
    model_name: str
    model: object
    metrics: Dict[str, float]
    shap_values: Optional[np.ndarray]
    feature_names: List[str]


def _save_figures(fig: plt.Figure, stem: str, output_dir: str | Path = "figures") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def _build_cv(groups: pd.Series, n_splits: int = 5) -> GroupKFold:
    unique_groups = groups.dropna().unique()
    n_groups = unique_groups.shape[0]
    splits = min(n_splits, n_groups) if n_groups else 2
    splits = max(splits, 2)
    return GroupKFold(n_splits=splits)


def fit_tree_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgb",
    groups: Optional[pd.Series] = None,
    random_state: int = 42,
    model_params: Optional[Dict[str, object]] = None,
) -> ModelResult:
    """
    Fit a tree-based model for chemical modulation kernel interpretation.
    """
    mask = (~X.isna().any(axis=1)) & y.notna()
    Xc = X.loc[mask]
    yc = y.loc[mask]
    if groups is None:
        groups = pd.Series(index=yc.index, data=np.arange(len(yc)) % 5)
    else:
        if isinstance(groups, (pd.Index, pd.Series)):
            groups = pd.Series(groups, index=yc.index)
        else:
            groups = pd.Series(groups, index=yc.index)
    group_series = groups.loc[yc.index]
    unique_groups = group_series.dropna().unique()
    cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]]
    if unique_groups.shape[0] < 2 or Xc.shape[0] < 10:
        split_idx = int(0.8 * Xc.shape[0])
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, Xc.shape[0]) if split_idx < Xc.shape[0] else np.arange(split_idx)
        cv_splits = [(train_idx, test_idx)]
    else:
        cv = _build_cv(group_series)
        cv_splits = cv.split(Xc, yc, groups=group_series)

    params = model_params or {}
    if model_type.lower() == "xgb":
        if XGBRegressor is None:
            raise ImportError("XGBRegressor not available; install xgboost to use model_type='xgb'.")
        base_kwargs = {
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": 4,
            "tree_method": "hist",
        }
        base_kwargs.update(params)
        model = XGBRegressor(**base_kwargs)
    elif model_type.lower() == "rf":
        base_kwargs = {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": random_state,
            "n_jobs": 4,
        }
        base_kwargs.update(params)
        model = RandomForestRegressor(**base_kwargs)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    r2_scores: List[float] = []
    mae_scores: List[float] = []
    for train_idx, test_idx in cv_splits:
        X_train, X_test = Xc.iloc[train_idx], Xc.iloc[test_idx]
        y_train, y_test = yc.iloc[train_idx], yc.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    model.fit(Xc, yc)
    shap_values = None
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(Xc)
    except Exception:
        shap_values = None

    metrics = {
        "R2_mean": float(np.nanmean(r2_scores)) if r2_scores else math.nan,
        "MAE_mean": float(np.nanmean(mae_scores)) if mae_scores else math.nan,
        "R2_std": float(np.nanstd(r2_scores)) if r2_scores else math.nan,
        "MAE_std": float(np.nanstd(mae_scores)) if mae_scores else math.nan,
        "n_samples": int(Xc.shape[0]),
        "n_features": int(Xc.shape[1]),
    }
    return ModelResult(model_type.lower(), model, metrics, shap_values, list(Xc.columns))


def compute_shap_summary(
    model: object,
    X: pd.DataFrame,
    stem: str,
    output_dir: str | Path = "figures",
    max_points: int = 2000,
    feature_display: int = 12,
) -> pd.DataFrame:
    """
    Compute SHAP values and write beeswarm summary plot.
    """
    mask = (~X.isna().any(axis=1))
    Xc = X.loc[mask]
    if max_points and Xc.shape[0] > max_points:
        Xc = Xc.sample(n=max_points, random_state=42)
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(Xc)
    except Exception:
        fallback = shap.Explainer(model.predict, Xc, feature_names=Xc.columns)
        shap_values = fallback(Xc).values
    plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values,
        Xc,
        max_display=min(feature_display, Xc.shape[1]),
        color=OKABE_ITO[1],
        show=False,
    )
    plt.title("SHAP summary", fontsize=12)
    plt.tight_layout()
    _save_figures(plt.gcf(), stem, output_dir=output_dir)
    plt.close()
    abs_mean = np.abs(shap_values).mean(axis=0) if shap_values is not None else np.zeros(Xc.shape[1])
    return pd.DataFrame({"feature": Xc.columns, "abs_mean_shap": abs_mean})


def plot_shap_interaction_heatmap(
    model: object,
    X: pd.DataFrame,
    stem: str,
    output_dir: str | Path = "figures",
    max_points: int = 800,
) -> None:
    """
    Plot SHAP interaction strength matrix (mean absolute interaction values).
    """
    sample = X
    if X.shape[0] > max_points:
        sample = X.sample(n=max_points, random_state=42)
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        inter = explainer.shap_interaction_values(sample)
    except Exception:
        return
    inter_arr = np.array(inter)
    if inter_arr.ndim == 3:
        inter_arr = inter_arr.mean(axis=0)
    strength = np.abs(inter_arr).mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    sns.heatmap(
        strength,
        xticklabels=sample.columns,
        yticklabels=sample.columns,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Mean |interaction SHAP|"},
    )
    ax.set_title("SHAP interaction strength")
    plt.tight_layout()
    _save_figures(fig, stem, output_dir=output_dir)
    plt.close(fig)


def plot_pdp_1d(
    model: object,
    X: pd.DataFrame,
    features: Sequence[str],
    stem: str,
    output_dir: str | Path = "figures",
    grid_resolution: int = 40,
) -> None:
    """
    Plot 1D partial dependence curves for selected features.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    features_to_plot = [f for f in features if f in X.columns]
    if not features_to_plot:
        return
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features_to_plot,
        ax=ax,
        kind="average",
        n_jobs=4,
        grid_resolution=grid_resolution,
    )
    ax.set_title("Partial dependence (1D)")
    plt.tight_layout()
    _save_figures(fig, stem, output_dir=output_dir)
    plt.close(fig)


def plot_pdp_2d(
    model: object,
    X: pd.DataFrame,
    feature_pairs: Sequence[Tuple[str, str]],
    stem: str,
    output_dir: str | Path = "figures",
    grid_resolution: int = 30,
) -> None:
    """
    Plot 2D partial dependence surfaces for feature pairs.
    """
    valid_pairs = [(a, b) for a, b in feature_pairs if a in X.columns and b in X.columns]
    if not valid_pairs:
        return
    n_cols = len(valid_pairs)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    for ax, pair in zip(axes.ravel(), valid_pairs):
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            [pair],
            ax=ax,
            kind="average",
            n_jobs=4,
            grid_resolution=grid_resolution,
        )
        ax.set_title(f"PDP: {pair[0]} vs {pair[1]}")
    plt.tight_layout()
    _save_figures(fig, stem, output_dir=output_dir)
    plt.close(fig)
