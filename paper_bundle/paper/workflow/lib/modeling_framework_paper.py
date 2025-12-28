from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from scipy.stats import zscore
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from skopt.plots import plot_objective

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, CHECKPOINT_DIR, FIGURE_DIR, INTERMEDIATE_DIR, TABLES_DIR
from paper.workflow.lib.plot_style_helvetica import register_helvetica_fonts
from src.workflow import core
from src.workflow.config import WorkflowConfig, default_config

CACHE_VERSION = "cache_v1"

# References: Gardiner (1985) Handbook of Stochastic Methods; Kolmogorov forward equation for BVOCs SDE.
# Equations: mean(T) = 0.5*a*T^2 + v0*T (intercept removed per mechanism reconstruction); var(T) = (k^2*T^3)/3 + k*sigma0*T^2 + sigma0^2*T.
# Parameters: a quadratic curvature, v0 linear slope vs temperature, k diffusion sensitivity, sigma0 diffusion intercept.
def _sde_mean_func(temp: np.ndarray, a: float, v0: float) -> np.ndarray:
    return 0.5 * a * temp * temp + v0 * temp


def _sde_var_func(temp: np.ndarray, k_val: float, sigma0: float) -> np.ndarray:
    return (k_val * k_val * temp * temp * temp) / 3.0 + k_val * sigma0 * temp * temp + sigma0 * sigma0 * temp


# References: Fuchs and Sutugin (1971) for condensation sink; CS = sum(4*pi*D_v*r*F(Kn)*N).
# Parameters: D_v vapor diffusivity (m^2 s^-1), r particle radius (m), Kn Knudsen number with mean free path, N number concentration (m^-3).
def compute_cs(df: pd.DataFrame, cfg: WorkflowConfig) -> pd.Series:
    number_cols = [c for c in df.columns if c.startswith("C") and c.endswith("um")]
    if not number_cols:
        raise ValueError("No number concentration columns (C*um) found for CS calculation.")
    cs = core.compute_condensation_sink(
        df[number_cols],
        df["temperature_c"],
        cfg.cs_diffusivity_m2_s,
        cfg.mean_free_path_nm,
        cfg.accommodation_coefficient,
    )
    cs = cs.replace([np.inf, -np.inf], np.nan)
    return cs


def set_plot_style() -> None:
    """
    Sets Matplotlib style parameters to meet Nature journal requirements:
    - Font: Helvetica Neue LT Pro (sans-serif)
    - Font sizes: 7-9pt for readability at print size
    - Linewidths: 0.5-1.0pt
    - Layout: Clean, minimal chartjunk
    """
    register_helvetica_fonts(BUNDLE_ROOT.parent)
    plt.rcParams.update(
        {
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue LT Pro", "Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            # Keep math glyphs consistent with Helvetica.
            "mathtext.rm": "Helvetica Neue LT Pro",
            "mathtext.it": "Helvetica Neue LT Pro:italic",
            "mathtext.bf": "Helvetica Neue LT Pro:bold",
             
            # Font sizes (Nature: 5-7pt for ticks/legends, 7-9pt for labels)
            "font.size": 10, # Slightly larger base for Helvetica readability
            "axes.titlesize": 10,
            "axes.titleweight": "bold", 
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 11,

            # Axes and Spines
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,

            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            
            # Grid and Lines
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            
            # Saving
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.dpi": 300,
        }
    )
    # Consistent Palette
    sns.set_palette(["#1b9e77", "#d95f02", "#7570b3", "#4d4d4d", "#66a61e"])


def load_base_data(cfg: WorkflowConfig | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    df = pd.read_parquet(INTERMEDIATE_DIR / "step01_clean.parquet")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")
    df = df.sort_index()
    iso_candidates = ["Isoprene", "isoprene", "isoprene_ppb", "isoprene_conc"]
    iso_candidates.extend([col for col in cfg.bvoc_columns if col not in iso_candidates])
    iso_col = next((col for col in iso_candidates if col in df.columns), None)
    if iso_col is not None:
        # References: mechanism reconstruction note using Isoprene as the BVOC proxy.
        # Equation: bvocs = Isoprene concentration; Parameters: bvocs target column equals observed isoprene surrogate.
        df["bvocs"] = pd.to_numeric(df[iso_col], errors="coerce")
    elif "bvocs" in df.columns:
        df["bvocs"] = pd.to_numeric(df["bvocs"], errors="coerce")
    else:
        raise KeyError("Isoprene proxy column not found; expected Isoprene/isoprene or cfg.bvoc_columns.")
    soa_cols = ["0.25um", "0.28um", "0.30um"]
    csoa_cols = ["C0.25um", "C0.28um", "C0.30um"]
    df["SOA"] = df[soa_cols].sum(axis=1, min_count=1)
    df["CSOA"] = df[csoa_cols].sum(axis=1, min_count=1)
    df["SOA"] = df["SOA"].clip(lower=0)
    df["CSOA"] = df["CSOA"].clip(lower=0)
    return df


def aggregate_by_hour_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by Hour:Minute within each site, following the BVOCs对SOA的拟合.py approach.
    This averages over all days sharing the same minute-of-day to create a climatological minute profile.
    """
    if "place" not in df.columns:
        return df
    work = df.copy()
    work["hour_min"] = work.index.strftime("%H:%M")
    numeric_cols = work.select_dtypes(include=[np.number]).columns
    grouped = (
        work.groupby(["place", "hour_min"])[numeric_cols]
        .mean()
        .reset_index()
    )
    # Rebuild a time index using a dummy date plus minute-of-day to preserve order; offset each place by +1 day to keep index unique.
    base_date = pd.to_datetime("2000-01-01")
    grouped["minutes_since_midnight"] = grouped["hour_min"].apply(
        lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
    )
    grouped = grouped.sort_values(["place", "minutes_since_midnight"])
    place_offsets = {p: i for i, p in enumerate(grouped["place"].unique())}
    grouped["Time"] = grouped.apply(
        lambda row: base_date
        + pd.Timedelta(days=place_offsets.get(row["place"], 0))
        + pd.to_timedelta(row["minutes_since_midnight"], unit="m"),
        axis=1,
    )
    grouped = grouped.drop(columns=["minutes_since_midnight"])
    grouped = grouped.set_index("Time")
    return grouped


def _clean_outliers(df: pd.DataFrame, cols: Iterable[str], threshold: float = 3.0) -> pd.DataFrame:
    clean = df.copy()
    for col in cols:
        if col not in clean.columns:
            continue
        series = pd.to_numeric(clean[col], errors="coerce")
        scores = zscore(series.to_numpy(), nan_policy="omit")
        mask = np.abs(scores) < threshold
        if mask.shape[0] != clean.shape[0]:
            mask = np.resize(mask, clean.shape[0])
        clean = clean.loc[mask]
    return clean


# References: Iglewicz and Hoaglin (1993) robust statistics; z = (x - mean(x)) / std(x).
# Equations: remove rows where |z| >= z_threshold for the provided columns.
# Parameters: columns target columns for screening; z_threshold absolute z-score cutoff controlling filtering strength.
def remove_extreme_rows(df: pd.DataFrame, columns: Iterable[str], z_threshold: float = 3.5) -> pd.DataFrame:
    return _clean_outliers(df, columns, threshold=z_threshold)


@dataclass
class SDEFitResult:
    place: str
    a: float
    v0: float
    k_val: float
    sigma0: float
    mean_r2: float
    var_r2: float
    total_r2: float
    cov_mean: List[List[float]]
    cov_var: List[List[float]]


def fit_sde_by_site(df: pd.DataFrame, cfg: WorkflowConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results: List[SDEFitResult] = []
    df_out = df.copy()
    bin_width = cfg.temperature_bin_c
    min_count = cfg.min_samples_per_temp_bin
    for place, sub in df.groupby("place"):
        sub = sub.dropna(subset=["bvocs", "temperature_c"]).copy()
        if sub.empty:
            continue
        sub["temp_bin"] = (sub["temperature_c"] / bin_width).round() * bin_width
        grouped = sub.groupby("temp_bin")["bvocs"]
        mu_series = grouped.mean()
        var_series = grouped.var(ddof=1).fillna(0)
        count_series = grouped.count()
        valid_mask = count_series >= min_count
        mu_series = mu_series[valid_mask]
        var_series = var_series[valid_mask]
        if mu_series.empty or var_series.empty:
            continue
        mu_series = mu_series[np.abs(zscore(mu_series)) < 3]
        var_series = var_series[np.abs(zscore(var_series.replace(0, np.nan).fillna(var_series.median()))) < 3]
        temp_grid_mu = mu_series.index.to_numpy(dtype=float)
        temp_grid_var = var_series.index.to_numpy(dtype=float)
        try:
            popt_mean, pcov_mean = curve_fit(
                _sde_mean_func,
                temp_grid_mu,
                mu_series.to_numpy(),
                maxfev=20000,
                p0=[0.01, 0.1],
            )
        except Exception:
            popt_mean = [0.0, 0.0]
            pcov_mean = np.full((2, 2), np.nan)
        try:
            popt_var, pcov_var = curve_fit(
                _sde_var_func,
                temp_grid_var,
                var_series.to_numpy(),
                bounds=(0, np.inf),
                maxfev=20000,
                p0=[0.01, max(var_series.median(), 1e-3)],
            )
        except Exception:
            popt_var = [0.0, max(var_series.median(), 1e-3)]
            pcov_var = np.full((2, 2), np.nan)
        mu_pred_binned = _sde_mean_func(temp_grid_mu, *popt_mean)
        var_pred_binned = _sde_var_func(temp_grid_var, *popt_var)
        mean_r2 = r2_score(mu_series.to_numpy(), mu_pred_binned) if mu_series.size > 1 else np.nan
        var_r2 = r2_score(var_series.to_numpy(), var_pred_binned) if var_series.size > 1 else np.nan
        temp_vals = sub["temperature_c"].to_numpy()
        full_mu = _sde_mean_func(temp_vals, *popt_mean)
        full_var = _sde_var_func(temp_vals, *popt_var)
        if full_mu.shape[0] != sub.shape[0]:
            full_mu = np.resize(full_mu, sub.shape[0])
        if full_var.shape[0] != sub.shape[0]:
            full_var = np.resize(full_var, sub.shape[0])
        var_target = (sub["bvocs"] - full_mu) ** 2
        var_r2_full = r2_score(var_target, full_var) if var_target.size > 1 else np.nan
        total_r2 = np.nanmean([mean_r2, var_r2_full])
        df_out.loc[sub.index, "bvoc_mu_hat"] = full_mu
        df_out.loc[sub.index, "bvoc_var_hat"] = np.clip(full_var, 1e-6, None)
        results.append(
            SDEFitResult(
                place=place,
                a=float(popt_mean[0]),
                v0=float(popt_mean[1]),
                k_val=float(popt_var[0]),
                sigma0=float(popt_var[1]),
                mean_r2=float(mean_r2),
                var_r2=float(var_r2),
                total_r2=float(total_r2),
                cov_mean=pcov_mean.tolist(),
                cov_var=pcov_var.tolist(),
            )
        )
    if not results:
        raise RuntimeError("SDE fitting failed for all sites.")
    summary = pd.DataFrame(
        [
            {
                "Category": "I",
                "ModelID": "SDE",
                "Place": r.place,
                "a": r.a,
                "v0": r.v0,
                "k": r.k_val,
                "sigma0": r.sigma0,
                "R2_mean": r.mean_r2,
                "R2_var": r.var_r2,
                "R2_total": r.total_r2,
                "cov_mean": json.dumps(r.cov_mean),
                "cov_var": json.dumps(r.cov_var),
            }
            for r in results
        ]
    )
    return summary, df_out


def _build_env_features(df: pd.DataFrame) -> pd.DataFrame:
    env = pd.DataFrame(index=df.index)
    rh = df["rh_pct"]
    env["HNO3"] = rh * df["NOx"]
    env["H2SO4"] = rh * df["SO2"]
    env["H2SO4O3"] = rh * df["SO2"] * df["O3"]
    env["HNO3O3"] = rh * df["NOx"] * df["O3"]
    env["O3hv"] = df["O3"] * df["rad_w_m2"]
    env["K"] = 1.0
    env["hv"] = df["rad_w_m2"]
    return env


def significance_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int, uncentered: bool = False) -> Dict[str, float]:
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum(y_true**2) if uncentered else np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - rss / tss if tss else np.nan
    if uncentered:
        adj_r2 = 1.0 - (1.0 - r2) * len(y_true) / max(len(y_true) - n_params, 1)
    else:
        adj_r2 = 1.0 - (1.0 - r2) * (len(y_true) - 1) / max(len(y_true) - n_params - 1, 1)
    rmse = math.sqrt(np.mean(residuals**2))
    bic = len(y_true) * math.log(rss / len(y_true) + 1e-12) + n_params * math.log(len(y_true))
    return {"R2": r2, "R2_uncentered": _r2_uncentered(y_true, y_pred), "Adj_R2": adj_r2, "RMSE": rmse, "BIC": bic}


def _r2_uncentered(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    denom = np.sum(y_true_arr**2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((y_true_arr - y_pred_arr) ** 2) / denom


def save_dual(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=500, bbox_inches="tight")


def save_table(df: pd.DataFrame, stem: str, sheet: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{stem}.csv"
    xlsx_path = TABLES_DIR / f"{stem}.xlsx"
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.book[sheet]
        from openpyxl.styles import Alignment, Font

        for cell in ws[1]:
            cell.font = Font(name="Times New Roman", bold=True)
            cell.alignment = Alignment(wrap_text=False)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name="Times New Roman", bold=False)
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)


def _default_plot_labels() -> Dict[str, object]:
    return {
        "palette_places": {"JH": "#1b9e77", "CM": "#d95f02"},
        "palette_extra": {
            "line": "#4e79a7",
            "cs": "#d95f02",
            "temp": "#7570b3",
            "cs_m1": "#7b3294",
            "cs_m2": "#e79f00",
            "cs_m3": "#3182bd",
        },
        "marker_map": {"1_cs": "s", "2_cs": "o", "3_cs": "^"},
        "param_labels": {
            "HNO3": "HNO3",
            "H2SO4": "H2SO4",
            "H2SO4O3": "H2SO4*O3",
            "HNO3O3": "HNO3*O3",
            "O3hv": "O3*hv",
            "hv": "hv",
            "K": "K",
            "C_T_hat": "T-driven BVOC",
        },
    }


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _config_snapshot(cfg: WorkflowConfig) -> Dict[str, object]:
    cfg_dict = asdict(cfg)
    for key, value in list(cfg_dict.items()):
        if isinstance(value, Path):
            cfg_dict[key] = str(value)
        elif isinstance(value, dict):
            cfg_dict[key] = {k: str(v) if isinstance(v, Path) else v for k, v in value.items()}
    cfg_dict["cache_version"] = CACHE_VERSION
    return cfg_dict


def _serialize_prediction_dict(pred_map: Dict[Tuple[str, str], pd.Series], value_col: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for (place, mid), series in pred_map.items():
        if series is None:
            continue
        ser = pd.Series(series)
        ser.index = pd.to_datetime(ser.index)
        frame = ser.reset_index()
        frame.columns = ["Time", value_col]
        frame["place"] = place
        frame["model_id"] = str(mid)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["Time", "place", "model_id", value_col])
    out = pd.concat(rows, ignore_index=True)
    out["Time"] = pd.to_datetime(out["Time"])
    return out.sort_values(["place", "model_id", "Time"])


def _serialize_cat2_diag(cat2_outputs: Dict[str, object]) -> pd.DataFrame:
    diag = cat2_outputs.get("diag", {}) if isinstance(cat2_outputs, dict) else {}
    frames: List[pd.DataFrame] = []
    entries: List[Tuple[str, str, Dict[str, pd.Series]]] = []
    for key, payload in diag.items():
        if payload is None:
            continue
        if isinstance(payload, dict) and any(isinstance(v, dict) for v in payload.values()):
            for nested_key, nested_payload in payload.items():
                place = key[0] if isinstance(key, tuple) else key
                entries.append((place, str(nested_key), nested_payload or {}))
        else:
            place, model_id = (key if isinstance(key, tuple) and len(key) == 2 else (key, "2_cs"))
            entries.append((place, str(model_id), payload))
    for place, model_id, payload in entries:
        cs = pd.Series(payload.get("cs"))
        ratio = pd.Series(payload.get("ratio"))
        lin_pred = pd.Series(payload.get("lin_pred"))
        soa_pred_cs = pd.Series(payload.get("soa_pred_cs"))
        obs = pd.Series(payload.get("obs"))
        cs.index = pd.to_datetime(cs.index)
        ratio.index = pd.to_datetime(ratio.index)
        lin_pred.index = pd.to_datetime(lin_pred.index)
        soa_pred_cs.index = pd.to_datetime(soa_pred_cs.index)
        obs.index = pd.to_datetime(obs.index)
        frame = pd.DataFrame(
            {
                "Time": cs.index,
                "place": place,
                "model_id": model_id,
                "cs": cs.to_numpy(),
                "ratio": ratio.reindex(cs.index).to_numpy(),
                "lin_pred": lin_pred.reindex(cs.index).to_numpy(),
                "soa_pred_cs": soa_pred_cs.reindex(cs.index).to_numpy(),
                "obs": obs.reindex(cs.index).to_numpy(),
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Time", "place", "model_id", "cs", "ratio", "lin_pred", "soa_pred_cs", "obs"])
    out = pd.concat(frames, ignore_index=True)
    out["Time"] = pd.to_datetime(out["Time"])
    return out.sort_values(["place", "model_id", "Time"])


def _serialize_ml_best_predictions(best_predictions: Dict[Tuple[str, str], Dict[str, object]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for (place, target), payload in best_predictions.items():
        soa_pred = pd.Series(payload.get("soa_pred"))
        y_true = pd.Series(payload.get("y_true"))
        y_pred_norm = pd.Series(payload.get("y_pred"))
        soa_pred.index = pd.to_datetime(soa_pred.index)
        y_true.index = pd.to_datetime(y_true.index)
        y_pred_norm.index = pd.to_datetime(y_pred_norm.index)
        frame = pd.DataFrame(
            {
                "Time": soa_pred.index,
                "place": place,
                "target": target,
                "soa_pred": soa_pred.to_numpy(),
                "y_true": y_true.reindex(soa_pred.index).to_numpy(),
                "y_pred_norm": y_pred_norm.reindex(soa_pred.index).to_numpy(),
                "model_name": payload.get("model_name", ""),
                "R2_test": payload.get("R2_test", np.nan),
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Time", "place", "target", "soa_pred", "y_true", "y_pred_norm", "model_name", "R2_test"])
    out = pd.concat(frames, ignore_index=True)
    out["Time"] = pd.to_datetime(out["Time"])
    return out.sort_values(["place", "target", "Time"])


def _serialize_ml_all_predictions(all_predictions: Dict[Tuple[str, str, str], Dict[str, object]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for (place, target, model_name), payload in all_predictions.items():
        y_true = pd.Series(payload.get("y_true"))
        y_pred = pd.Series(payload.get("y_pred"))
        y_true.index = pd.to_datetime(y_true.index)
        y_pred.index = pd.to_datetime(y_pred.index)
        frame = pd.DataFrame(
            {
                "Time": y_true.index,
                "place": place,
                "target": target,
                "model_name": model_name,
                "soa_true": y_true.to_numpy(),
                "soa_pred": y_pred.reindex(y_true.index).to_numpy(),
                "R2_test": payload.get("R2_test", np.nan),
                "R2_test_uncentered": payload.get("R2_test_uncentered", np.nan),
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Time", "place", "target", "model_name", "soa_true", "soa_pred", "R2_test", "R2_test_uncentered"])
    out = pd.concat(frames, ignore_index=True)
    out["Time"] = pd.to_datetime(out["Time"])
    return out.sort_values(["place", "target", "model_name", "Time"])


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")


def persist_cache_outputs(
    df_sde: pd.DataFrame,
    cfg: WorkflowConfig,
    sde_summary: pd.DataFrame,
    cat1: Dict[str, object],
    cat2: Dict[str, object],
    ml_outputs: Dict[str, object],
    cache_dir: Path | str = CHECKPOINT_DIR,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_to_save = df_sde.reset_index().rename(columns={"index": "Time"})
    df_to_save["Time"] = pd.to_datetime(df_to_save["Time"])
    _save_parquet(df_to_save, cache_dir / "df_sde.parquet")
    metadata = {"cache_version": CACHE_VERSION, "config": _config_snapshot(cfg)}
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))
    if not sde_summary.empty:
        sde_summary.to_json(cache_dir / "sde_params.json", orient="records")
    cat1_preds_df = _serialize_prediction_dict(cat1.get("predictions", {}), "pred")
    _save_parquet(cat1_preds_df, cache_dir / "cat1_predictions.parquet")
    cat1.get("metrics", pd.DataFrame()).to_json(cache_dir / "cat1_metrics.json", orient="records")
    cat1.get("params", pd.DataFrame()).to_json(cache_dir / "cat1_params.json", orient="records")
    cs_series = cat2.get("cs", pd.Series(dtype=float))
    cs_diag_df = _serialize_cat2_diag(cat2)
    _save_parquet(cs_diag_df, cache_dir / "cs_diag.parquet")
    cs_preds = cat2.get("predictions", {})
    cs_preds_df = _serialize_prediction_dict(cs_preds, "pred_cs")
    if not cs_preds_df.empty:
        cs_preds_df["model_id"] = cs_preds_df["model_id"].fillna("2_cs").astype(str)
        if not cs_diag_df.empty:
            merge_cols = ["Time", "place", "model_id"]
            diag_subset = cs_diag_df[merge_cols + ["lin_pred", "cs"]] if "model_id" in cs_diag_df.columns else cs_diag_df
            diag_subset = diag_subset.rename(columns={"model_id": "model_id"})
            cs_preds_df = cs_preds_df.merge(diag_subset, on=merge_cols, how="left")
    else:
        cs_preds_df = pd.DataFrame(columns=["Time", "place", "model_id", "pred_cs", "lin_pred", "cs"])
    _save_parquet(cs_preds_df, cache_dir / "cs_predictions.parquet")
    if isinstance(cs_series, pd.Series) and not df_to_save.empty:
        cs_values = pd.Series(cs_series)
        cs_values = cs_values.reset_index(drop=True)
        if len(cs_values) == len(df_to_save):
            cs_frame = df_to_save[["Time", "place"]].copy()
            cs_frame["cs"] = cs_values.to_numpy()
            _save_parquet(cs_frame, cache_dir / "cs_series.parquet")
    cat2.get("metrics", pd.DataFrame()).to_json(cache_dir / "cs_metrics.json", orient="records")
    cat2.get("params", pd.DataFrame()).to_json(cache_dir / "cs_params.json", orient="records")
    ml_metrics = ml_outputs.get("metrics", pd.DataFrame())
    ml_features = ml_outputs.get("features", pd.DataFrame())
    if isinstance(ml_metrics, pd.DataFrame):
        ml_metrics.to_json(cache_dir / "ml_metrics.json", orient="records")
    if isinstance(ml_features, pd.DataFrame):
        ml_features.to_json(cache_dir / "ml_features.json", orient="records")
    best_pred_df = _serialize_ml_best_predictions(ml_outputs.get("best_predictions", {}))
    _save_parquet(best_pred_df, cache_dir / "ml_best_predictions.parquet")
    all_pred_df = _serialize_ml_all_predictions(ml_outputs.get("all_predictions", {}))
    if not all_pred_df.empty:
        _save_parquet(all_pred_df, cache_dir / "ml_all_predictions.parquet")
    ml_metrics_cs = ml_outputs.get("metrics_cs", pd.DataFrame())
    if isinstance(ml_metrics_cs, pd.DataFrame):
        ml_metrics_cs.to_json(cache_dir / "ml_metrics_cs.json", orient="records")
    best_pred_cs_df = _serialize_ml_best_predictions(ml_outputs.get("best_predictions_cs", {}))
    _save_parquet(best_pred_cs_df, cache_dir / "ml_best_predictions_cs.parquet")
    all_pred_cs_df = _serialize_ml_all_predictions(ml_outputs.get("all_predictions_cs", {}))
    if not all_pred_cs_df.empty:
        _save_parquet(all_pred_cs_df, cache_dir / "ml_all_predictions_cs.parquet")
    labels_path = cache_dir / "plot_labels.json"
    if not labels_path.exists():
        labels_path.write_text(json.dumps(_default_plot_labels(), indent=2))


def _load_table_or_cache(cache_path: Path, table_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_json(cache_path)
    if table_path.exists():
        return pd.read_csv(table_path)
    raise FileNotFoundError(f"Missing required file: {cache_path} (cache) and {table_path} (tables)")


def load_cached_results(
    cache_dir: Path | str = CHECKPOINT_DIR,
    tables_dir: Path | str = TABLES_DIR,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    cache_dir = Path(cache_dir)
    tables_dir = Path(tables_dir)
    df_path = cache_dir / "df_sde.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"Cached dataframe not found at {df_path}")
    df_sde = pd.read_parquet(df_path)
    if "Time" not in df_sde.columns:
        raise ValueError("df_sde cache must include Time column.")
    df_sde["Time"] = pd.to_datetime(df_sde["Time"])
    df_sde = df_sde.set_index("Time").sort_index()
    cat1_pred_path = cache_dir / "cat1_predictions.parquet"
    if not cat1_pred_path.exists():
        raise FileNotFoundError(f"Cached cat1 predictions not found at {cat1_pred_path}")
    cat1_pred_df = pd.read_parquet(cat1_pred_path)
    required_cat1_cols = {"Time", "place", "model_id", "pred"}
    if missing := required_cat1_cols - set(cat1_pred_df.columns):
        raise ValueError(f"cat1_predictions.parquet missing columns: {missing}")
    cat1_pred_df["Time"] = pd.to_datetime(cat1_pred_df["Time"])
    cat1_preds: Dict[Tuple[str, str], pd.Series] = {}
    for (place, mid), sub in cat1_pred_df.groupby(["place", "model_id"]):
        ser = pd.Series(sub["pred"].to_numpy(), index=sub["Time"])
        cat1_preds[(place, str(mid))] = ser.sort_index()
    cat1_metrics = _load_table_or_cache(cache_dir / "cat1_metrics.json", tables_dir / "results_linear_models.csv")
    cat1_params = _load_table_or_cache(cache_dir / "cat1_params.json", tables_dir / "results_linear_params.csv")
    cat1_outputs = {"metrics": cat1_metrics, "params": cat1_params, "predictions": cat1_preds}
    cs_pred_path = cache_dir / "cs_predictions.parquet"
    cs_diag_path = cache_dir / "cs_diag.parquet"
    cat2_preds: Dict[Tuple[str, str], pd.Series] = {}
    cs_series_combined = pd.Series(dtype=float)
    diag_map: Dict[Tuple[str, str], Dict[str, pd.Series]] = {}
    if cs_pred_path.exists():
        cs_pred_df = pd.read_parquet(cs_pred_path)
        required_cs_cols = {"Time", "place", "pred_cs"}
        if missing_cs := required_cs_cols - set(cs_pred_df.columns):
            raise ValueError(f"cs_predictions.parquet missing columns: {missing_cs}")
        cs_pred_df["Time"] = pd.to_datetime(cs_pred_df["Time"])
        if "model_id" not in cs_pred_df.columns:
            cs_pred_df["model_id"] = "2_cs"
        cs_pred_df["model_id"] = cs_pred_df["model_id"].fillna("2_cs").astype(str)
        for (place, mid), sub in cs_pred_df.groupby(["place", "model_id"]):
            cat2_preds[(place, str(mid))] = pd.Series(sub["pred_cs"].to_numpy(), index=sub["Time"]).sort_index()
    if cs_diag_path.exists():
        cs_diag_df = pd.read_parquet(cs_diag_path)
        cs_diag_df["Time"] = pd.to_datetime(cs_diag_df["Time"])
        if "model_id" not in cs_diag_df.columns:
            cs_diag_df["model_id"] = "2_cs"
        cs_diag_df["model_id"] = cs_diag_df["model_id"].fillna("2_cs").astype(str)
        for (place, mid), sub in cs_diag_df.groupby(["place", "model_id"]):
            diag_map[(place, str(mid))] = {
                "cs": pd.Series(sub["cs"].to_numpy(), index=sub["Time"]).sort_index(),
                "ratio": pd.Series(sub["ratio"].to_numpy(), index=sub["Time"]).sort_index(),
                "lin_pred": pd.Series(sub["lin_pred"].to_numpy(), index=sub["Time"]).sort_index(),
                "soa_pred_cs": pd.Series(sub["soa_pred_cs"].to_numpy(), index=sub["Time"]).sort_index(),
                "obs": pd.Series(sub["obs"].to_numpy(), index=sub["Time"]).sort_index(),
            }
    cs_series_path = cache_dir / "cs_series.parquet"
    if cs_series_path.exists():
        cs_series_df = pd.read_parquet(cs_series_path)
        cs_series_df["Time"] = pd.to_datetime(cs_series_df["Time"])
        merged = (
            df_sde.reset_index()
            .rename(columns={"index": "Time"})
            .merge(cs_series_df, on=["Time", "place"], how="left")
        )
        cs_series_combined = pd.Series(merged["cs"].to_numpy(), index=merged["Time"])
    cat2_metrics = _load_table_or_cache(cache_dir / "cs_metrics.json", tables_dir / "results_cs_models.csv")
    cat2_params = _load_table_or_cache(cache_dir / "cs_params.json", tables_dir / "results_cs_params.csv")
    cat2_outputs = {
        "metrics": cat2_metrics,
        "params": cat2_params,
        "predictions": cat2_preds,
        "cs": cs_series_combined,
        "diag": diag_map,
    }
    ml_metrics_path = cache_dir / "ml_metrics.json"
    ml_metrics_cs_path = cache_dir / "ml_metrics_cs.json"
    ml_features_path = cache_dir / "ml_features.json"
    ml_metrics_df = pd.read_json(ml_metrics_path) if ml_metrics_path.exists() else _load_table_or_cache(ml_metrics_path, tables_dir / "results_ML_models.csv")
    if ml_metrics_cs_path.exists():
        ml_metrics_cs_df = pd.read_json(ml_metrics_cs_path)
    else:
        metrics_cs_table = tables_dir / "results_ML_models_cs.csv"
        ml_metrics_cs_df = pd.read_csv(metrics_cs_table) if metrics_cs_table.exists() else pd.DataFrame()
    ml_features_df = pd.read_json(ml_features_path) if ml_features_path.exists() else _load_table_or_cache(ml_features_path, tables_dir / "results_ML_features.csv")
    ml_best_path = cache_dir / "ml_best_predictions.parquet"
    ml_all_path = cache_dir / "ml_all_predictions.parquet"
    ml_best_path_cs = cache_dir / "ml_best_predictions_cs.parquet"
    ml_all_path_cs = cache_dir / "ml_all_predictions_cs.parquet"
    best_predictions: Dict[Tuple[str, str], Dict[str, object]] = {}
    all_predictions: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    best_predictions_cs: Dict[Tuple[str, str], Dict[str, object]] = {}
    all_predictions_cs: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    if ml_best_path.exists():
        best_df = pd.read_parquet(ml_best_path)
        best_df["Time"] = pd.to_datetime(best_df["Time"])
        for (place, target), sub in best_df.groupby(["place", "target"]):
            ser = pd.Series(sub["soa_pred"].to_numpy(), index=sub["Time"]).sort_index()
            best_predictions[(place, target)] = {
                "soa_pred": ser,
                "model_name": sub["model_name"].iloc[0] if "model_name" in sub else "",
                "R2_test": float(sub["R2_test"].iloc[0]) if "R2_test" in sub else np.nan,
            }
    if ml_all_path.exists():
        all_df = pd.read_parquet(ml_all_path)
        all_df["Time"] = pd.to_datetime(all_df["Time"])
        for (place, target, model_name), sub in all_df.groupby(["place", "target", "model_name"]):
            y_true = pd.Series(sub["soa_true"].to_numpy(), index=sub["Time"]).sort_index()
            y_pred = pd.Series(sub["soa_pred"].to_numpy(), index=sub["Time"]).sort_index()
            all_predictions[(place, target, model_name)] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "R2_test": float(sub["R2_test"].iloc[0]) if "R2_test" in sub else np.nan,
                "R2_test_uncentered": float(sub["R2_test_uncentered"].iloc[0]) if "R2_test_uncentered" in sub else np.nan,
            }
    if ml_best_path_cs.exists():
        best_df_cs = pd.read_parquet(ml_best_path_cs)
        best_df_cs["Time"] = pd.to_datetime(best_df_cs["Time"])
        for (place, target), sub in best_df_cs.groupby(["place", "target"]):
            ser = pd.Series(sub["soa_pred"].to_numpy(), index=sub["Time"]).sort_index()
            best_predictions_cs[(place, target)] = {
                "soa_pred": ser,
                "model_name": sub["model_name"].iloc[0] if "model_name" in sub else "",
                "R2_test": float(sub["R2_test"].iloc[0]) if "R2_test" in sub else np.nan,
            }
    if ml_all_path_cs.exists():
        all_df_cs = pd.read_parquet(ml_all_path_cs)
        all_df_cs["Time"] = pd.to_datetime(all_df_cs["Time"])
        for (place, target, model_name), sub in all_df_cs.groupby(["place", "target", "model_name"]):
            y_true = pd.Series(sub["soa_true"].to_numpy(), index=sub["Time"]).sort_index()
            y_pred = pd.Series(sub["soa_pred"].to_numpy(), index=sub["Time"]).sort_index()
            all_predictions_cs[(place, target, model_name)] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "R2_test": float(sub["R2_test"].iloc[0]) if "R2_test" in sub else np.nan,
                "R2_test_uncentered": float(sub["R2_test_uncentered"].iloc[0]) if "R2_test_uncentered" in sub else np.nan,
            }
    ml_outputs = {
        "metrics": ml_metrics_df,
        "metrics_cs": ml_metrics_cs_df,
        "features": ml_features_df,
        "best_predictions": best_predictions,
        "best_predictions_cs": best_predictions_cs,
        "all_predictions": all_predictions,
        "all_predictions_cs": all_predictions_cs,
    }
    labels_path = cache_dir / "plot_labels.json"
    labels_cfg = _default_plot_labels()
    if labels_path.exists():
        try:
            labels_cfg = json.loads(labels_path.read_text())
        except Exception:
            labels_cfg = _default_plot_labels()
    return df_sde, cat1_outputs, cat2_outputs, ml_outputs, labels_cfg


def run_category_i(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    metrics_rows: List[Dict[str, object]] = []
    param_rows: List[Dict[str, object]] = []
    predictions: Dict[Tuple[str, str], pd.Series] = {}
    for place, sub in df.groupby("place"):
        env = _build_env_features(sub)
        y = sub["SOA"]
        X1 = env.mul(sub["bvocs"], axis=0)
        mask1 = (~X1.isna().any(axis=1)) & y.notna()
        model1 = sm.WLS(y[mask1], X1[mask1]).fit(cov_type="HC3")
        pred1 = model1.predict(X1)
        predictions[(place, "1")] = pred1
        metrics1 = _regression_metrics(y[mask1].to_numpy(), pred1[mask1].to_numpy(), X1.shape[1], uncentered=True)
        metrics_rows.append({"Category": "I", "ModelID": "1", "Place": place, **metrics1})
        for name, coef, se, pval in zip(model1.params.index, model1.params.values, model1.bse.values, model1.pvalues):
            param_rows.append(
                {
                    "Category": "I",
                    "ModelID": "1",
                    "Place": place,
                    "Parameter": name,
                    "Estimate": coef,
                    "StdErr": se,
                    "p_value": pval,
                    "Significance": significance_stars(pval),
                }
            )
        X2 = env.mul(sub["bvoc_mu_hat"], axis=0)
        weights2 = 1.0 / np.clip(sub["bvoc_var_hat"], 1e-6, None)
        mask2 = (~X2.isna().any(axis=1)) & y.notna() & weights2.notna()
        model2 = sm.WLS(y[mask2], X2[mask2], weights=weights2[mask2]).fit(cov_type="HC3")
        pred2 = model2.predict(X2)
        predictions[(place, "2")] = pred2
        metrics2 = _regression_metrics(y[mask2].to_numpy(), pred2[mask2].to_numpy(), X2.shape[1], uncentered=True)
        metrics_rows.append({"Category": "I", "ModelID": "2", "Place": place, **metrics2})
        for name, coef, se, pval in zip(model2.params.index, model2.params.values, model2.bse.values, model2.pvalues):
            param_rows.append(
                {
                    "Category": "I",
                    "ModelID": "2",
                    "Place": place,
                    "Parameter": name,
                    "Estimate": coef,
                    "StdErr": se,
                    "p_value": pval,
                    "Significance": significance_stars(pval),
                }
            )
        X3 = pd.DataFrame({"C_T_hat": sub["bvoc_mu_hat"]}, index=sub.index)
        mask3 = X3["C_T_hat"].notna() & y.notna() & weights2.notna()
        model3 = sm.WLS(y[mask3], X3[mask3], weights=weights2[mask3]).fit(cov_type="HC3")
        pred3 = pd.Series(model3.predict(X3), index=X3.index)
        predictions[(place, "3")] = pred3.reindex(sub.index, fill_value=np.nan)
        metrics3 = _regression_metrics(y[mask3].to_numpy(), pred3[mask3].to_numpy(), X3.shape[1], uncentered=True)
        metrics_rows.append({"Category": "I", "ModelID": "3", "Place": place, **metrics3})
        for name, coef, se, pval in zip(model3.params.index, model3.params.values, model3.bse.values, model3.pvalues):
            param_rows.append(
                {
                    "Category": "I",
                    "ModelID": "3",
                    "Place": place,
                    "Parameter": name,
                    "Estimate": coef,
                    "StdErr": se,
                    "p_value": pval,
                    "Significance": significance_stars(pval),
                }
            )
    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(param_rows)
    if not metrics_df.empty:
        save_table(metrics_df, "results_linear_models", "LinearModels")
    if not params_df.empty:
        save_table(params_df, "results_linear_params", "LinearParams")
    return {"metrics": metrics_df, "params": params_df, "predictions": predictions}


def _parameter_effects(params_df: pd.DataFrame, env: pd.DataFrame, place: str) -> pd.DataFrame:
    mean_env = env.mean()
    rows = []
    for _, row in params_df[params_df["Place"] == place].iterrows():
        if row["Parameter"] not in mean_env:
            continue
        scaled = row["Estimate"] * mean_env[row["Parameter"]]
        rows.append({"Parameter": row["Parameter"], "Scaled": scaled, "Significance": row["Significance"]})
    return pd.DataFrame(rows)


def plot_category_i(df: pd.DataFrame, outputs: Dict[str, pd.DataFrame]) -> None:
    preds = outputs["predictions"]
    params_df = outputs["params"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    places = sorted(df["place"].dropna().unique())
    model_labels = {"1": "(1) Observed BVOCs", "2": "(2) Temp-fit BVOCs", "3": "(3) Temp-only"}
    for r, place in enumerate(places[:2]):
        y_true = df.loc[df["place"] == place, "SOA"]
        for c, mid in enumerate(["1", "2", "3"]):
            ax = axes[r, c]
            pred = preds.get((place, mid))
            if pred is None:
                ax.axis("off")
                continue
            mask = y_true.notna() & pred.notna()
            sns.kdeplot(x=y_true[mask], y=pred[mask], fill=True, cmap="Blues", ax=ax, thresh=0.05)
            ax.scatter(y_true[mask], pred[mask], s=8, alpha=0.25, color="#1b9e77")
            lim = max(y_true[mask].max(), pred[mask].max())
            ax.plot([0, lim], [0, lim], linestyle="--", color="#d95f02", linewidth=1.2)
            ax.set_title(f"{place} {model_labels[mid]}")
            ax.set_xlabel("Observed SOA")
            ax.set_ylabel("Predicted SOA")
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(6))
    env_map = {place: _build_env_features(df[df["place"] == place]) for place in places}
    for idx, mid in enumerate(["1", "2", "3"]):
        ax = axes[2, idx]
        effects = []
        for place in places:
            sub_params = params_df[(params_df["ModelID"] == mid) & (params_df["Place"] == place)]
            eff = _parameter_effects(sub_params, env_map.get(place, pd.DataFrame()), place)
            eff["Place"] = place
            effects.append(eff)
        if not effects:
            ax.axis("off")
            continue
        eff_df = pd.concat(effects)
        if eff_df.empty or "Scaled" not in eff_df:
            ax.axis("off")
            continue
        eff_df = eff_df.sort_values("Scaled")
        colors = {"JH": "#1b9e77", "CM": "#d95f02"}
        y_pos = np.arange(eff_df.shape[0])
        ax.barh(y_pos, eff_df["Scaled"], color=[colors.get(p, "#7570b3") for p in eff_df["Place"]])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(eff_df["Place"] + " " + eff_df["Parameter"])
        for yloc, sig, val in zip(y_pos, eff_df["Significance"], eff_df["Scaled"]):
            ax.text(val, yloc, f" {sig}", va="center", ha="left")
        ax.set_title(f"Parameter effects Model {mid}")
        ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_dual(fig, "Fig_linear_models_SOA")
    plt.close(fig)


# References: Saturation response k_env(CS) = beta_max * CS / (CS + CS0); Parameters beta_max max scaling, CS0 half-saturation condensation sink.
def _fit_cs_scaling(cs: pd.Series, ratio: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    cs_clean = cs.replace([np.inf, -np.inf], np.nan)
    mask = cs_clean.notna() & ratio.notna() & (cs_clean > 0) & (ratio > 0)
    x = cs_clean[mask].to_numpy()
    y = ratio[mask].to_numpy()
    if x.size < 5:
        return np.array([np.nan, np.nan]), np.full((2, 2), np.nan)
    p0 = [np.nanpercentile(y, 90), np.nanmedian(x)]
    popt, pcov = curve_fit(lambda c, bmax, cs0: bmax * c / (c + cs0), x, y, p0=p0, bounds=(0, np.inf), maxfev=20000)
    return popt, pcov


def run_category_ii(df: pd.DataFrame, cat1_outputs: Dict[str, pd.DataFrame], cfg: WorkflowConfig) -> Dict[str, pd.DataFrame]:
    cs_series = compute_cs(df, cfg)
    metrics_rows: List[Dict[str, object]] = []
    param_rows: List[Dict[str, object]] = []
    preds_with_cs: Dict[Tuple[str, str], pd.Series] = {}
    diagnostics: Dict[Tuple[str, str], Dict[str, pd.Series]] = {}
    baseline = cat1_outputs["predictions"]
    for place, sub in df.groupby("place"):
        obs = sub["SOA"]
        lin_pred_base = baseline.get((place, "2"))
        if lin_pred_base is None:
            continue
        cs_place = cs_series.loc[sub.index]
        lin_pred_base = lin_pred_base.reindex(sub.index)
        mask = obs.notna() & lin_pred_base.notna() & cs_place.notna()
        ratio = (obs[mask] / lin_pred_base[mask]).replace([np.inf, -np.inf], np.nan)
        popt, pcov = _fit_cs_scaling(cs_place, ratio)
        beta_max, cs0 = popt
        # References: Kulmala et al. (2012) condensation sink scaling; Equation: k_env = beta_max * CS / (CS + CS0); Parameters: beta_max fitted maximum scaling, CS0 half-saturation CS, CS condensational sink series.
        fitted = beta_max * cs_place / (cs_place + cs0)
        soa_pred_cs = lin_pred_base * fitted
        preds_with_cs[(place, "2_cs")] = soa_pred_cs
        base_metrics = _regression_metrics(obs[mask].to_numpy(), lin_pred_base[mask].to_numpy(), 1, uncentered=True)
        cs_metrics = _regression_metrics(obs[mask].to_numpy(), soa_pred_cs[mask].to_numpy(), 2, uncentered=True)
        metrics_rows.append(
            {
                "Category": "II",
                "ModelID": "CS",
                "Place": place,
                "beta_max": beta_max,
                "CS0": cs0,
                "R2_before": base_metrics["R2"],
                "RMSE_before": base_metrics["RMSE"],
                "R2_after": cs_metrics["R2"],
                "RMSE_after": cs_metrics["RMSE"],
            }
        )
        param_rows.append(
            {
                "Category": "II",
                "ModelID": "CS",
                "Place": place,
                "Parameter": "beta_max",
                "Estimate": beta_max,
                "StdErr": math.sqrt(pcov[0][0]) if not np.isnan(pcov).all() else np.nan,
                "p_value": np.nan,
                "Significance": "",
            }
        )
        param_rows.append(
            {
                "Category": "II",
                "ModelID": "CS",
                "Place": place,
                "Parameter": "CS0",
                "Estimate": cs0,
                "StdErr": math.sqrt(pcov[1][1]) if not np.isnan(pcov).all() else np.nan,
                "p_value": np.nan,
                "Significance": "",
            }
        )
        diagnostics[(place, "2_cs")] = {
            "cs": cs_place,
            "ratio": (obs / lin_pred_base).replace([np.inf, -np.inf], np.nan),
            "lin_pred": lin_pred_base,
            "soa_pred_cs": soa_pred_cs,
            "obs": obs,
        }
        for extra_mid in ["1", "3"]:
            lin_pred_extra = baseline.get((place, extra_mid))
            if lin_pred_extra is None:
                continue
            lin_pred_extra = lin_pred_extra.reindex(sub.index)
            ratio_extra = (obs / lin_pred_extra).replace([np.inf, -np.inf], np.nan)
            soa_pred_extra = lin_pred_extra * fitted
            preds_with_cs[(place, f"{extra_mid}_cs")] = soa_pred_extra
            diagnostics[(place, f"{extra_mid}_cs")] = {
                "cs": cs_place,
                "ratio": ratio_extra,
                "lin_pred": lin_pred_extra,
                "soa_pred_cs": soa_pred_extra,
                "obs": obs,
            }
    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(param_rows)
    if not metrics_df.empty:
        save_table(metrics_df, "results_cs_models", "CSModels")
    if not params_df.empty:
        save_table(params_df, "results_cs_params", "CSParams")
    return {"metrics": metrics_df, "params": params_df, "predictions": preds_with_cs, "cs": cs_series, "diag": diagnostics}


def _prepare_ml_features(
    df: pd.DataFrame, ct_hat: pd.Series, cs: pd.Series, beta_max: float | None, cs0: float | None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    ct_hat_clean = pd.to_numeric(ct_hat, errors="coerce").clip(lower=1e-3)
    env = pd.DataFrame(
        {
            "O3": pd.to_numeric(df["O3"], errors="coerce"),
            "NOx": pd.to_numeric(df["NOx"], errors="coerce"),
            "SO2": pd.to_numeric(df["SO2"], errors="coerce"),
            "Radiation": pd.to_numeric(df["rad_w_m2"], errors="coerce"),
            "Temperature": pd.to_numeric(df["temperature_c"], errors="coerce"),
            "Humidity": pd.to_numeric(df["rh_pct"], errors="coerce"),
        },
        index=df.index,
    )
    env["HNO3"] = env["Humidity"] * env["NOx"]
    env["H2SO4"] = env["Humidity"] * env["SO2"]
    env["HNO3O3"] = env["Humidity"] * env["NOx"] * env["O3"]
    env["H2SO4O3"] = env["Humidity"] * env["SO2"] * env["O3"]
    env["O3hv"] = env["O3"] * env["Radiation"]
    env["C_T_hat"] = ct_hat_clean
    # Time encoders to capture seasonal/diurnal drift (helps when train/test are time-separated)
    dt_index = df.index
    if isinstance(dt_index, pd.DatetimeIndex):
        doy = dt_index.dayofyear.to_numpy()
        hour = dt_index.hour.to_numpy()
        env["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
        env["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
        env["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        env["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    if beta_max is not None and cs0 is not None:
        cs_clean = pd.to_numeric(cs, errors="coerce").clip(lower=1e-6)
        k_env = beta_max * cs_clean / (cs_clean + cs0)
    else:
        k_env = pd.Series(1.0, index=df.index)
    soa_clean = pd.to_numeric(df["SOA"], errors="coerce")
    y1 = soa_clean / ct_hat_clean.replace(0, np.nan)
    y2 = soa_clean / (ct_hat_clean.replace(0, np.nan) * k_env.replace(0, np.nan))
    y1 = y1.replace([np.inf, -np.inf], np.nan)
    y2 = y2.replace([np.inf, -np.inf], np.nan)
    return env.astype(float), y1.astype(float), y2.astype(float), k_env.astype(float), ct_hat_clean.astype(float)


def build_ml_training_set(
    sub: pd.DataFrame,
    cs_series: pd.Series,
    beta_max: float | None,
    cs0: float | None,
    target_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, str, pd.Series, pd.Series]:
    """
    Prepare ML training data for a given place/target using the same preprocessing
    steps as run_category_iii.

    Returns
    -------
    X_target : pd.DataFrame
        Feature matrix after masking and z-score filtering.
    y_target : pd.Series
        Target series aligned with X_target.
    multiplier : pd.Series
        Multipliers (C_T_hat or C_T_hat * k_env) aligned with X_target.
    variant_label : str
        Either "base" for Y1 or "cs" for Y2 (CS-adjusted target).
    ct_series : pd.Series
        Temperature-driven BVOC baseline, aligned to the original index.
    k_env_series : pd.Series
        Environmental scaling series; constant 1.0 for base variant.
    """

    features_base, y1, y2, k_env_series, ct_series = _prepare_ml_features(sub, sub["bvoc_mu_hat"], cs_series, beta_max, cs0)
    variant_label = "base"
    if target_name == "Y1":
        features = features_base
        target_series = y1
        multiplier = ct_series
    elif target_name in ("Y2", "Y1_cs"):
        features = features_base.copy()
        features["k_env_cs"] = k_env_series
        target_series = y2
        multiplier = ct_series * k_env_series
        variant_label = "cs"
    else:
        raise ValueError(f"Unsupported target_name={target_name} for ML training set.")

    feature_mask = features.notna().all(axis=1)
    target_mask = target_series.notna() & feature_mask
    multiplier = multiplier.reindex(target_series.index)
    target_mask = target_mask & multiplier.notna()
    X_target = features[target_mask]
    y_target = target_series[target_mask]
    multiplier = multiplier.loc[y_target.index]

    z_scores = zscore(y_target.to_numpy(), nan_policy="omit")
    if z_scores.shape[0] != len(y_target):
        z_scores = np.resize(z_scores, len(y_target))
    keep_mask = np.abs(z_scores) < 3.5
    X_target = X_target.loc[y_target.index[keep_mask]]
    y_target = y_target.loc[y_target.index[keep_mask]]
    multiplier = multiplier.loc[y_target.index]

    return X_target, y_target, multiplier, variant_label, ct_series, k_env_series


def _bayes_model_space(model_name: str):
    if model_name == "RF":
        est = RandomForestRegressor(random_state=42, n_jobs=-1)
        space = {
            "n_estimators": Integer(200, 400),
            "max_depth": Integer(4, 20),
            "min_samples_leaf": Integer(1, 12),
            "max_features": Categorical(["sqrt", "log2", None, 0.5]),
        }
    elif model_name == "ADB":
        est = AdaBoostRegressor(random_state=42)
        space = {
            "n_estimators": Integer(50, 600),
            "learning_rate": Real(0.01, 0.8, prior="log-uniform"),
            "loss": Categorical(["linear", "square", "exponential"]),
        }
    elif model_name == "XGB":
        if XGBRegressor is None:
            raise ImportError("XGBRegressor not available; install xgboost.")
        est = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
        space = {
            "n_estimators": Integer(200, 500),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(3, 6),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "gamma": Real(0.0, 0.5),
            "min_child_weight": Real(0.5, 5.0),
        }
    elif model_name == "SVR":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]
        )
        space = {
            "model__C": Real(1e-2, 5e2, prior="log-uniform"),
            "model__epsilon": Real(1e-3, 1.0, prior="log-uniform"),
            "model__gamma": Real(1e-4, 1e1, prior="log-uniform"),
        }
    elif model_name == "KNN":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor()),
            ]
        )
        space = {
            "model__n_neighbors": Integer(3, 80),
            "model__weights": Categorical(["uniform", "distance"]),
            "model__p": Integer(1, 2),
        }
    elif model_name == "Ridge":
        est = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        space = {
            "model__alpha": Real(1e-4, 1e3, prior="log-uniform"),
            "model__fit_intercept": Categorical([False]),
        }
    elif model_name == "MLP":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(max_iter=600, random_state=42)),
            ]
        )
        space = None  # Bayes search skipped for MLP to avoid categorical tuple casting errors
    elif model_name == "GBDT":
        est = GradientBoostingRegressor(random_state=42)
        space = {
            "n_estimators": Integer(80, 400),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "max_depth": Integer(2, 6),
            "min_samples_leaf": Integer(1, 8),
            "subsample": Real(0.6, 1.0),
        }
    else:
        raise ValueError(f"Unsupported model for Bayes search: {model_name}")
    return est, space


def _train_ml_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    use_bayes: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, object, object | None, dict | None, np.ndarray]:
    est, space = _bayes_model_space(model_name)
    search_obj = None
    # References: Gelman and Hill (2007) hierarchical models; Equation: y_center = y - mean(y_train); Parameters: target_offset is the training mean removed to reduce intercept drift.
    target_offset = float(np.nanmean(y_train)) if len(y_train) else 0.0
    y_train_fit = y_train - target_offset
    if use_bayes and space:
        cv = TimeSeriesSplit(n_splits=3)
        opt = BayesSearchCV(
            est,
            search_spaces=space,
            n_iter=10,
            cv=cv,
            scoring="r2",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
        opt.fit(X_train, y_train_fit)
        model = opt.best_estimator_
        search_obj = opt
    else:
        model = est
        model.fit(X_train, y_train_fit)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    # References: Draper and Smith (1998) Applied Regression Analysis; Equation: bias_train = mean(pred_train - y_train_fit); Parameters: pred_train model outputs, y_train_fit centered target values.
    bias_train = float(np.nanmean(pred_train - y_train_fit)) if len(pred_train) else 0.0
    pred_train = pred_train - bias_train + target_offset
    pred_test = pred_test - bias_train + target_offset
    metrics = {
        "R2_train": r2_score(y_train, pred_train),
        "R2_test": r2_score(y_test, pred_test),
        "RMSE_test": mean_squared_error(y_test, pred_test) ** 0.5,
        "MAE_test": mean_absolute_error(y_test, pred_test),
    }
    return metrics, pred_test, model, search_obj, space, pred_train


def run_category_iii(df: pd.DataFrame, cat1: Dict[str, pd.DataFrame], cat2: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    metrics_rows: List[Dict[str, object]] = []
    metrics_rows_cs: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    best_predictions: Dict[Tuple[str, str], Dict[str, object]] = {}
    best_predictions_cs: Dict[Tuple[str, str], Dict[str, object]] = {}
    all_predictions: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    all_predictions_cs: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    cs_series = cat2.get("cs")
    for place, sub in df.groupby("place"):
        ct_hat = sub["bvoc_mu_hat"]
        cs_place = cs_series.loc[sub.index] if cs_series is not None else pd.Series(np.nan, index=sub.index)
        beta_max = None
        cs0 = None
        cs_params = cat2.get("params", pd.DataFrame())
        if not cs_params.empty:
            row_cs = cs_params[(cs_params["Place"] == place) & (cs_params["Parameter"] == "beta_max")]
            row_cs0 = cs_params[(cs_params["Place"] == place) & (cs_params["Parameter"] == "CS0")]
            if not row_cs.empty and not row_cs0.empty:
                beta_max = row_cs["Estimate"].iloc[0]
                cs0 = row_cs0["Estimate"].iloc[0]
        features_base, y1, y2, k_env_series, ct_series = _prepare_ml_features(sub, ct_hat, cs_place, beta_max, cs0)
        base_mask = features_base.notna().all(axis=1)
        features_base = features_base[base_mask]
        y1 = y1[base_mask]
        y2 = y2[base_mask]
        k_env_series = k_env_series.reindex(features_base.index)
        ct_series = ct_series.reindex(features_base.index)
        # References: Fuchs and Sutugin (1971) condensation sink scaling applied to ML targets; Equation: SOA_pred_cs = y_pred * C_T_hat * k_env with k_env = beta_max * CS / (CS + CS0); Parameters: beta_max maximum environmental scaling, CS0 half-saturation sink, C_T_hat temperature-driven BVOC baseline.
        features_cs = features_base.copy()
        features_cs["k_env_cs"] = k_env_series
        if features_base.empty:
            continue
        target_configs = [
            {"name": "Y1", "series": y1, "features": features_base, "multiplier": ct_series, "variant": "base"},
            {"name": "Y2", "series": y2, "features": features_cs, "multiplier": ct_series * k_env_series, "variant": "cs", "alias": "Y1_cs"},
        ]
        for cfg_target in target_configs:
            target_name = cfg_target["name"]
            target_series = cfg_target["series"]
            features_set = cfg_target["features"]
            multiplier_series = cfg_target["multiplier"]
            variant_tag = cfg_target.get("variant", "base")
            alias_target = cfg_target.get("alias")
            feature_mask = features_set.notna().all(axis=1)
            target_mask = target_series.notna() & feature_mask
            multiplier_series = multiplier_series.reindex(target_series.index)
            target_mask = target_mask & multiplier_series.notna()
            X_target = features_set[target_mask]
            y_target = target_series[target_mask]
            # References: Iglewicz and Hoaglin (1993) z-score filter; Equation: z = (y - mean)/std, keep |z| < 3.5; Parameters: mean and std are computed on the current target slice.
            z_scores = zscore(y_target.to_numpy(), nan_policy="omit")
            if z_scores.shape[0] != len(y_target):
                z_scores = np.resize(z_scores, len(y_target))
            keep_mask = np.abs(z_scores) < 3.5
            X_target = X_target.loc[y_target.index[keep_mask]]
            y_target = y_target.loc[y_target.index[keep_mask]]
            if X_target.empty or len(y_target) < 5:
                continue
            X_train, X_test, y_train, y_test = train_test_split(
                X_target, y_target, test_size=0.2, random_state=42, shuffle=True
            )
            model_candidates = ["RF", "SVR", "KNN", "MLP", "GBDT", "ADB", "Ridge"]
            if XGBRegressor is not None:
                model_candidates.append("XGB")
            for model_name in model_candidates:
                metrics, pred_test, model, search_obj, space, pred_train = _train_ml_model(
                    X_train, X_test, y_train, y_test, model_name
                )
                if model_name in ["RF", "GBDT", "XGB"]:
                    importances = model.feature_importances_
                    for feat, imp in sorted(
                        zip(X_target.columns, importances), key=lambda tup: tup[1], reverse=True
                    )[:5]:
                        feature_rows.append(
                            {
                                "Place": place,
                                "Target": target_name,
                                "Model": model_name,
                                "Feature": feat,
                                "Importance": imp,
                            }
                        )
                else:
                    sample_size = min(300, len(X_test))
                    perm = permutation_importance(
                        model,
                        X_test.iloc[:sample_size],
                        y_test.iloc[:sample_size],
                        n_repeats=5,
                        random_state=42,
                        n_jobs=1,
                    )
                    order = np.argsort(perm.importances_mean)[::-1][:5]
                    for idx in order:
                        feature_rows.append(
                            {
                                "Place": place,
                                "Target": target_name,
                                "Model": model_name,
                                "Feature": X_target.columns[idx],
                                "Importance": perm.importances_mean[idx],
                            }
                        )
                key = (place, target_name)
                ct_test = ct_series.reindex(y_test.index)
                k_env_test = k_env_series.reindex(y_test.index)
                y_pred_series = pd.Series(pred_test, index=y_test.index)
                ct_train = ct_series.reindex(y_train.index)
                k_env_train = k_env_series.reindex(y_train.index)
                if target_name == "Y1":
                    soa_pred = y_pred_series * ct_test
                    soa_pred_train = pd.Series(pred_train, index=y_train.index) * ct_train
                else:
                    soa_pred = y_pred_series * multiplier_series.reindex(y_test.index)
                    soa_pred_train = pd.Series(pred_train, index=y_train.index) * multiplier_series.reindex(y_train.index)
                y_train_soa = sub["SOA"].reindex(y_train.index)
                y_test_soa = sub["SOA"].reindex(y_test.index)
                r2_train_real = r2_score(y_train_soa, soa_pred_train)
                r2_test_real = r2_score(y_test_soa, soa_pred)
                r2_train_uncentered = _r2_uncentered(y_train_soa, soa_pred_train)
                r2_test_uncentered = _r2_uncentered(y_test_soa, soa_pred)
                rmse_test_real = mean_squared_error(y_test_soa, soa_pred) ** 0.5
                mae_test_real = mean_absolute_error(y_test_soa, soa_pred)
                metrics_rows.append(
                    {
                        "Category": "III",
                        "Model": model_name,
                        "Place": place,
                        "Target": target_name,
                        "Variant": variant_tag,
                        "R2_train_norm": metrics["R2_train"],
                        "R2_test_norm": metrics["R2_test"],
                        "R2_train": r2_train_real,
                        "R2_test": r2_test_real,
                        "R2_train_uncentered": r2_train_uncentered,
                        "R2_test_uncentered": r2_test_uncentered,
                        "RMSE_test": rmse_test_real,
                        "MAE_test": mae_test_real,
                    }
                )
                all_predictions[(place, target_name, model_name)] = {
                    "y_true": y_test_soa,
                    "y_pred": soa_pred,
                    "R2_test": r2_test_real,
                    "R2_test_uncentered": r2_test_uncentered,
                }
                if key not in best_predictions or r2_test_real > best_predictions[key]["R2_test"]:
                    best_predictions[key] = {
                        "R2_test": r2_test_real,
                        "y_true": y_test,
                        "y_pred": y_pred_series,
                        "soa_pred": soa_pred,
                        "model_name": model_name,
                        "search": search_obj,
                        "space": space,
                    }
                if alias_target:
                    metrics_rows_cs.append(
                        {
                            "Category": "III",
                            "Model": model_name,
                            "Place": place,
                            "Target": alias_target,
                            "Variant": variant_tag,
                            "R2_train_norm": metrics["R2_train"],
                            "R2_test_norm": metrics["R2_test"],
                            "R2_train": r2_train_real,
                            "R2_test": r2_test_real,
                            "R2_train_uncentered": r2_train_uncentered,
                            "R2_test_uncentered": r2_test_uncentered,
                            "RMSE_test": rmse_test_real,
                            "MAE_test": mae_test_real,
                        }
                    )
                    all_predictions_cs[(place, alias_target, model_name)] = {
                        "y_true": y_test_soa,
                        "y_pred": soa_pred,
                        "R2_test": r2_test_real,
                        "R2_test_uncentered": r2_test_uncentered,
                    }
                    if (place, alias_target) not in best_predictions_cs or r2_test_real > best_predictions_cs[(place, alias_target)]["R2_test"]:
                        best_predictions_cs[(place, alias_target)] = {
                            "R2_test": r2_test_real,
                            "y_true": y_test,
                            "y_pred": y_pred_series,
                            "soa_pred": soa_pred,
                            "model_name": model_name,
                        }
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_cs_df = pd.DataFrame(metrics_rows_cs)
    feature_df = pd.DataFrame(feature_rows)
    if not metrics_df.empty:
        save_table(metrics_df, "results_ML_models", "MLModels")
    if not metrics_cs_df.empty:
        save_table(metrics_cs_df, "results_ML_models_cs", "MLModelsCS")
    if not feature_df.empty:
        save_table(feature_df, "results_ML_features", "MLFeatures")
    return {
        "metrics": metrics_df,
        "metrics_cs": metrics_cs_df,
        "features": feature_df,
        "best_predictions": best_predictions,
        "best_predictions_cs": best_predictions_cs,
        "all_predictions": all_predictions,
        "all_predictions_cs": all_predictions_cs,
    }


def plot_category_iii(df: pd.DataFrame, cat1: Dict[str, pd.DataFrame], ml_outputs: Dict[str, pd.DataFrame]) -> None:
    metrics_df = ml_outputs.get("metrics", pd.DataFrame())
    feature_df = ml_outputs.get("features", pd.DataFrame())
    best_predictions = ml_outputs.get("best_predictions", {})
    if metrics_df.empty:
        return
    # Ranking plot
    fig_rank, ax_rank = plt.subplots(figsize=(8, 5))
    metrics_sorted = metrics_df.sort_values("R2_test", ascending=False)
    sns.barplot(data=metrics_sorted, x="Model", y="R2_test", hue="Target", ax=ax_rank, palette="Set2")
    ax_rank.set_title("Category III: Test R2 ranking")
    ax_rank.grid(alpha=0.3, linewidth=0.8, axis="y")
    save_dual(fig_rank, "Fig_ML_performance_rank")
    plt.close(fig_rank)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="Model", y="R2_test", hue="Target", ax=ax, palette="Set2")
    ax.set_title("Category III: Test R2 across ML models")
    ax.grid(alpha=0.3, linewidth=0.8, axis="y")
    save_dual(fig, "Fig_ML_performance_comparison")
    plt.close(fig)

    for place in df["place"].unique():
        base_pred = cat1["predictions"].get((place, "2"))
        y_true = df.loc[df["place"] == place, "SOA"]
        best_entry = best_predictions.get((place, "Y1"))
        if best_entry is None or base_pred is None:
            continue
        fig_scat, ax_scat = plt.subplots(figsize=(6, 5))
        mask_base = y_true.notna() & base_pred.notna()
        ax_scat.scatter(y_true[mask_base], base_pred[mask_base], s=8, alpha=0.25, label="Linear model 2", color="#1b9e77")
        mask_ml = y_true.index.isin(best_entry["soa_pred"].index)
        ax_scat.scatter(y_true[mask_ml], best_entry["soa_pred"], s=8, alpha=0.25, label=f"ML best ({best_entry['model_name']})", color="#d95f02")
        lim = max(y_true[mask_ml].max(), best_entry["soa_pred"].max(), base_pred[mask_base].max())
        ax_scat.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1.0)
        ax_scat.set_xlabel("Observed SOA")
        ax_scat.set_ylabel("Predicted SOA")
        ax_scat.set_title(f"{place} ML vs linear predictions")
        ax_scat.legend(frameon=False)
        ax_scat.grid(alpha=0.3, linewidth=0.8)
        save_dual(fig_scat, f"Fig_ML_vs_linear_scatter_{place}")
        plt.close(fig_scat)

        fig_res, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        y_obs_ml = y_true[mask_ml]
        y_pred_ml = best_entry["soa_pred"].reindex(y_obs_ml.index)
        axs[0].scatter(y_obs_ml, y_pred_ml, s=10, alpha=0.5, color="#2c7bb6", edgecolors="w", linewidth=0.3)
        line_lim = [0, max(y_obs_ml.max(), y_pred_ml.max())]
        axs[0].plot(line_lim, line_lim, linestyle="--", color="black", linewidth=1.0)
        axs[0].set_ylabel("Predicted SOA")
        axs[0].set_title(f"{place} ML best ({best_entry['model_name']}) diag")
        resid = y_obs_ml - y_pred_ml
        axs[1].scatter(y_obs_ml, resid, s=10, alpha=0.5, color="#d7191c", edgecolors="w", linewidth=0.3)
        axs[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
        axs[1].set_xlabel("Observed SOA")
        axs[1].set_ylabel("Residual")
        for ax_sub in axs:
            ax_sub.grid(alpha=0.3, linewidth=0.8)
            ax_sub.xaxis.set_major_locator(MaxNLocator(6))
            ax_sub.yaxis.set_major_locator(MaxNLocator(6))
        fig_res.tight_layout()
        save_dual(fig_res, f"Fig_ML_residual_{place}")
        plt.close(fig_res)

        search_obj = best_entry.get("search")
        space = best_entry.get("space")
        if search_obj is not None and hasattr(search_obj, "optimizer_results_") and isinstance(space, dict):
            try:
                dims = list(space.keys())
                fig_obj = plot_objective(
                    result=search_obj.optimizer_results_[0],
                    dimensions=[k.replace("model__", "").replace("_", " ") for k in dims],
                    n_points=20,
                )
                plt.suptitle(f"Bayes objective {place} {best_entry['model_name']}", fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    FIGURE_DIR / f"Fig_ML_bayes_objective_{place}_{best_entry['model_name']}.png",
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception:
                plt.close("all")

    if not feature_df.empty:
        top_feats = (
            feature_df.sort_values("Importance", ascending=False)
            .groupby(["Place", "Target"])
            .head(5)
            .reset_index(drop=True)
        )
        if not top_feats.empty:
            fig_feat, ax_feat = plt.subplots(figsize=(8, 6))
            sns.barplot(data=top_feats, x="Importance", y="Feature", hue="Place", ax=ax_feat)
            ax_feat.set_title("Top feature importance (Category III)")
            ax_feat.grid(alpha=0.3, linewidth=0.8, axis="x")
            save_dual(fig_feat, "Fig_ML_feature_importance")
            plt.close(fig_feat)


def run_full_pipeline(plot: bool = True) -> None:
    from paper.workflow.lib import plotting_paper as plotting

    cfg = default_config()
    set_plot_style()
    df = load_base_data(cfg)
    df = aggregate_by_hour_min(df)
    df = _clean_outliers(df, ["SOA", "bvocs", "NOx", "O3", "SO2", "rh_pct", "temperature_c"])
    ensure_cols = ["bvocs", "NOx", "O3", "SO2", "rh_pct", "rad_w_m2", "temperature_c", "SOA", "place"]
    missing_cols = [c for c in ensure_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    sde_summary, df_sde = fit_sde_by_site(df, cfg)
    save_table(sde_summary, "results_sde_params", "SDE")
    cat1 = run_category_i(df_sde)
    cat2 = run_category_ii(df_sde, cat1, cfg)
    ml_outputs = run_category_iii(df_sde, cat1, cat2)
    persist_cache_outputs(df_sde, cfg, sde_summary, cat1, cat2, ml_outputs)
    if plot:
        plotting.plot_category_i(df_sde, cat1, cat2)
        plotting.plot_category_ii(cat2)
        plotting.plot_category_iii(df_sde, cat1, ml_outputs, cat2)


__all__ = [
    "load_base_data",
    "fit_sde_by_site",
    "run_category_i",
    "run_category_ii",
    "run_category_iii",
    "run_full_pipeline",
    "set_plot_style",
    "persist_cache_outputs",
    "load_cached_results",
    "remove_extreme_rows",
    "_default_plot_labels",
]


if __name__ == "__main__":
    # Friendly guidance when the module is executed directly.
    msg = (
        "This module is part of the src.workflow package. "
        "Run from project root with one of:\n"
        "  python soa_full_pipeline.py\n"
        "  python -m src.workflow.modeling_framework\n"
    )
    print(msg)
