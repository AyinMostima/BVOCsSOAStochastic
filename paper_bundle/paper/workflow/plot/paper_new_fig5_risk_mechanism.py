from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontEntry
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import logging
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, shapiro, zscore, norm
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

THIS_BUNDLE_ROOT = Path(__file__).resolve().parents[3]  # .../paper_bundle
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(THIS_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_BUNDLE_ROOT))

from paper.workflow.lib.paper_paths import FIGURE_DIR, PAPER_ROOT  # noqa: E402
from src.workflow.modeling_framework import (  # noqa: E402
    compute_cs,
    default_config,
    load_base_data,
    load_cached_results,
)


def _register_helvetica_fonts_strict() -> str:
    """
    Register local Helvetica-family font files and return the family name.

    This function is strict: it avoids silent fallback fonts to keep a
    Nature/Science-compatible look across machines.
    """
    candidates = [
        REPO_ROOT / "HelveticaNeueLTPro-Roman.otf",
        REPO_ROOT / "HelveticaNeueLTPro-Bd.otf",
        REPO_ROOT / "HelveticaNeueLTPro-It.otf",
        REPO_ROOT / "HelveticaNeueLTPro-BdIt.otf",
    ]
    for font_file in candidates:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))

    def _add_face(font_file: Path, style: str, weight: str) -> None:
        if not font_file.exists():
            return
        font_manager.fontManager.ttflist.append(
            FontEntry(
                fname=str(font_file),
                name="Helvetica",
                style=style,
                variant="normal",
                weight=weight,
                stretch="normal",
                size="scalable",
            )
        )

    try:
        fp = font_manager.FontProperties(family="Helvetica")
        font_manager.findfont(fp, fallback_to_default=False)
        return "Helvetica"
    except Exception:
        _add_face(REPO_ROOT / "HelveticaNeueLTPro-Roman.otf", style="normal", weight="normal")
        _add_face(REPO_ROOT / "HelveticaNeueLTPro-It.otf", style="italic", weight="normal")
        _add_face(REPO_ROOT / "HelveticaNeueLTPro-Bd.otf", style="normal", weight="bold")
        _add_face(REPO_ROOT / "HelveticaNeueLTPro-BdIt.otf", style="italic", weight="bold")
        try:
            font_manager.fontManager._findfont_cached.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        fp = font_manager.FontProperties(family="Helvetica")
        font_manager.findfont(fp, fallback_to_default=False)
        return "Helvetica"


def _set_nature_style() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MATPLOTLIBRC", str(PAPER_ROOT / "matplotlibrc"))
    _register_helvetica_fonts_strict()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 9.6,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.unicode_minus": False,
            "savefig.transparent": False,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Match mathtext appearance to Helvetica (avoid mixed fonts in equations).
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica:bold",
            "mathtext.sf": "Helvetica",
            "mathtext.default": "regular",
        }
    )
    # Silence verbose fontTools subset logs when exporting PDF.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


# References: Gardiner (2009) Stochastic Methods; Ito SDE and moment relations.
# Equations:
# - Stochastic BVOC proxy (temperature-driven): dC = mu(T) dt + sigma(T) dW.
# - Here the fitted moment forms follow the legacy scripts:
#   E[C|T] = Q0 + (a*T^2)/2 + v0*T
#   Var[C|T] = (k^2*T^3)/3 + k*T^2*sigma0 + T*sigma0^2
# Parameters:
# - T: temperature (deg C).
# - Q0, a, v0: mean-response parameters.
# - k, sigma0: volatility parameters controlling variance.
def mean_relation(T: np.ndarray, Q0: float, a: float, v0: float) -> np.ndarray:
    return Q0 + (a * T**2) / 2.0 + T * v0


def std_dev_relation(T: np.ndarray, k: float, sigma0: float) -> np.ndarray:
    return (k**2 * T**3) / 3.0 + k * T**2 * sigma0 + T * sigma0**2


def _remove_outliers(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    scores = zscore(values, nan_policy="omit")
    mask = np.isfinite(scores) & (np.abs(scores) < threshold)
    return values[mask]


def _hourmin_to_float(hour_min_str: str) -> float:
    hour, minute = map(int, hour_min_str.split(":"))
    return float(hour) + float(minute) / 60.0


def _normal_distribution_fit_by_hour(hour_groups: pd.core.groupby.SeriesGroupBy) -> pd.DataFrame:
    results = {}
    for hour, group in hour_groups:
        params = norm.fit(group)
        _, p_value = shapiro(group)
        results[int(hour)] = {
            "Mean": float(params[0]),
            "StandardDeviation": float(params[1]),
            "PValue": float(p_value),
        }
    out = pd.DataFrame.from_dict(results, orient="index").sort_index()
    out.index.name = "Hour"
    return out.reset_index()


def _compute_cs_hourmin_mean(
    df_base: pd.DataFrame, cat2_outputs: dict, expected_hourmins: pd.Series | None = None
) -> pd.DataFrame:
    # References: Fuchs and Sutugin (1971) condensation sink; CS = sum(4*pi*D_v*r*F(Kn)*N).
    # Equations: CS(t) is provided by the main workflow cache when available.
    # Parameters: CS is the condensation sink time series aggregated to Hour_Min climatology.
    def _cs_from_cache() -> pd.DataFrame | None:
        cs_series_cached = cat2_outputs.get("cs")
        if isinstance(cs_series_cached, pd.Series) and not cs_series_cached.empty:
            cs_frame = pd.DataFrame({"Time": cs_series_cached.index, "CS": cs_series_cached.values})
            cs_frame["Hour_Min"] = cs_frame["Time"].dt.strftime("%H:%M")
            return cs_frame.groupby("Hour_Min")["CS"].mean(numeric_only=True).reset_index()
        return None

    def _cs_from_base() -> pd.DataFrame:
        cfg_cs = default_config()
        number_cols = [c for c in df_base.columns if c.startswith("C") and c.endswith("um")]
        if not number_cols:
            raise ValueError("No C*um columns found for CS calculation.")
        df_hour = df_base.reset_index()
        df_hour["Hour_Min"] = df_hour["Time"].dt.strftime("%H:%M")
        grouped_num = df_hour.groupby(["place", "Hour_Min"])[number_cols].mean(numeric_only=True).reset_index()
        grouped_temp = (
            df_hour.groupby(["place", "Hour_Min"])["temperature_c"].mean(numeric_only=True).reset_index()
        )
        cs_input = grouped_num.merge(grouped_temp, on=["place", "Hour_Min"], how="inner")
        cs_values = compute_cs(cs_input[["temperature_c"] + number_cols], cfg_cs)
        cs_input["CS"] = cs_values
        return cs_input.groupby("Hour_Min")["CS"].mean(numeric_only=True).reset_index()

    cs_cache = _cs_from_cache()
    if expected_hourmins is None:
        if cs_cache is not None:
            return cs_cache
        return _cs_from_base()

    expected = pd.DataFrame({"Hour_Min": expected_hourmins.astype(str).unique()})
    expected = expected.sort_values("Hour_Min").reset_index(drop=True)

    cs_base = _cs_from_base()
    cs_base = expected.merge(cs_base, on="Hour_Min", how="left").rename(columns={"CS": "CS_base"})

    if cs_cache is None:
        out = cs_base.rename(columns={"CS_base": "CS"})[["Hour_Min", "CS"]]
        return out

    cs_cache = expected.merge(cs_cache, on="Hour_Min", how="left").rename(columns={"CS": "CS_cache"})
    merged = cs_cache.merge(cs_base, on="Hour_Min", how="left")
    merged["CS"] = merged["CS_cache"].where(merged["CS_cache"].notna(), merged["CS_base"])
    out = merged[["Hour_Min", "CS"]]
    return out


@dataclass(frozen=True)
class _KernelFit:
    params_mean: np.ndarray
    params_var: np.ndarray
    coef: np.ndarray
    feature_names: Tuple[str, ...]
    k_env_series: np.ndarray
    r2_unc: float
    r_c: float
    hourmin_frame: pd.DataFrame
    full_soa_mean: float
    full_soa_std: float
    full_temp_mean: float
    full_temp_std: float


def _fit_temp_bvoc_kernel_and_cs(df_base: pd.DataFrame, cat2_outputs: dict) -> _KernelFit:
    """
    Fit the temperature-driven BVOC stochastic proxy + linear kernel and CS modulation.

    This function mirrors the legacy linear CS trial script logic while keeping the
    implementation lightweight for figure assembly.
    """
    df_long = df_base.reset_index().rename(columns={"Time": "Datetime"})
    df_long["Time"] = df_long["Datetime"]
    df_long["Temperature"] = df_long["temperature_c"]
    df_long["Radiation"] = df_long["rad_w_m2"]
    df_long["Humidity"] = df_long["rh_pct"]
    df_long["Isoprene"] = df_long["bvocs"]
    df_long["Hour_Min"] = df_long["Datetime"].dt.strftime("%H:%M")
    df_long["Hour"] = df_long["Time"].dt.hour
    df_long["seconds"] = (df_long["Time"] - df_long["Time"].dt.normalize()).dt.total_seconds()

    dataall = df_long[
        [
            "Time",
            "Isoprene",
            "SOA",
            "SO2",
            "NOx",
            "O3",
            "Radiation",
            "Temperature",
            "Humidity",
            "Hour_Min",
            "Hour",
            "seconds",
            "place",
        ]
    ].copy()
    full_soa_mean = float(dataall["SOA"].mean(numeric_only=True))
    full_soa_std = float(dataall["SOA"].std(ddof=1))
    full_temp_mean = float(dataall["Temperature"].mean(numeric_only=True))
    full_temp_std = float(dataall["Temperature"].std(ddof=1))

    # Fit normal-by-hour statistics on Hour_Min climatology (legacy approach).
    data_grouped = dataall.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    grouped_by_hour = data_grouped.groupby("Hour")["Isoprene"]
    normality_df = _normal_distribution_fit_by_hour(grouped_by_hour)
    temp_by_hour = data_grouped.groupby("Hour")["Temperature"].mean(numeric_only=True).reset_index()
    normality_df = normality_df.merge(temp_by_hour, on="Hour", how="left")

    T = normality_df["Temperature"].to_numpy(dtype=float)
    mean_values = normality_df["Mean"].to_numpy(dtype=float)
    var_values = (normality_df["StandardDeviation"].to_numpy(dtype=float) ** 2).astype(float)

    mean_filtered = _remove_outliers(mean_values)
    var_filtered = _remove_outliers(var_values)
    T_mean = T[np.isin(mean_values, mean_filtered)]
    T_var = T[np.isin(var_values, var_filtered)]

    params_mean, _ = curve_fit(mean_relation, T_mean, mean_filtered, method="trf", maxfev=10000)
    params_var, _ = curve_fit(std_dev_relation, T_var, var_filtered, method="trf", maxfev=10000)

    # Build Hour_Min climatology for the linear kernel.
    hourmin = dataall.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    hourmin["Time"] = pd.to_datetime(hourmin["Hour_Min"], format="%H:%M")
    hourmin["Hour_Float"] = hourmin["Hour_Min"].map(_hourmin_to_float)

    hourmin["HNO3"] = hourmin["Humidity"] * hourmin["NOx"]
    hourmin["H2SO4"] = hourmin["Humidity"] * hourmin["SO2"]
    hourmin["H2SO4O3"] = hourmin["Humidity"] * hourmin["SO2"] * hourmin["O3"]
    hourmin["HNO3O3"] = hourmin["Humidity"] * hourmin["NOx"] * hourmin["O3"]
    hourmin["O3hv"] = hourmin["O3"] * hourmin["Radiation"]
    hourmin["K"] = 1.0

    variables = ("HNO3", "H2SO4", "H2SO4O3", "HNO3O3", "O3hv", "K", "Radiation")
    feature_names = tuple(f"{v}_BVOCs" for v in variables)

    # Use temperature-fitted BVOCs as the source term for the linear kernel (legacy approach).
    isoprene_fitted = mean_relation(hourmin["Temperature"].to_numpy(), *params_mean)
    X = np.column_stack([hourmin[v].to_numpy(dtype=float) * isoprene_fitted for v in variables])
    y = hourmin["SOA"].to_numpy(dtype=float)

    # Weighted least squares with weights = 1 / Var[BVOC|T] (legacy approach).
    # Note: The legacy script uses Var directly (no sqrt) for weights. Preserve that behavior.
    var_bvoc = std_dev_relation(hourmin["Temperature"].to_numpy(dtype=float), *params_var)
    weights = np.where(var_bvoc > 0, 1.0 / var_bvoc, 0.0)
    W = np.sqrt(weights)
    Xw = X * W[:, None]
    yw = y * W
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    y_base = X @ coef

    # References: Saturating CS modulation used in the paper (Michaelis-Menten-like).
    # Equations: k_env(CS) = beta_max * CS / (CS + CS0); SOA = y_base * k_env(CS).
    # Parameters: beta_max (maximum scaling), CS0 (half-saturation CS).
    cs_hourmin = _compute_cs_hourmin_mean(df_base, cat2_outputs, expected_hourmins=hourmin["Hour_Min"])
    hourmin = hourmin.merge(cs_hourmin, on="Hour_Min", how="left")

    ratio = np.where(y_base != 0, y / y_base, np.nan)
    ratio = pd.Series(ratio).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    cs_vals = hourmin["CS"].to_numpy(dtype=float)
    mask = np.isfinite(cs_vals) & (cs_vals > 0) & np.isfinite(ratio) & (ratio > 0)
    x = cs_vals[mask]
    y_ratio = ratio[mask]

    def _cs_kernel(c: np.ndarray, beta_max: float, cs0: float) -> np.ndarray:
        return beta_max * c / (c + cs0)

    beta0 = float(np.nanpercentile(y_ratio, 90)) if y_ratio.size else 1.0
    cs00 = float(np.nanmedian(x)) if x.size else 1.0
    beta_max, cs0 = curve_fit(
        _cs_kernel, x, y_ratio, p0=(beta0, cs00), bounds=(0.0, np.inf), maxfev=20000
    )[0]
    k_env = _cs_kernel(cs_vals, float(beta_max), float(cs0))
    y_pred = y_base * k_env

    y_true = y.astype(float)
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot_unc = np.nansum(y_true**2)
    r2_unc = float(1.0 - ss_res / ss_tot_unc) if ss_tot_unc > 0 else float("nan")
    mask_corr = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.any(mask_corr):
        y_true_c = y_true[mask_corr]
        y_pred_c = y_pred[mask_corr]
        corr_mat = np.corrcoef(y_true_c, y_pred_c)
        r_c = float(corr_mat[0, 1])
    else:
        r_c = float("nan")

    return _KernelFit(
        params_mean=np.asarray(params_mean, dtype=float),
        params_var=np.asarray(params_var, dtype=float),
        coef=np.asarray(coef, dtype=float),
        feature_names=feature_names,
        k_env_series=np.asarray(k_env, dtype=float),
        r2_unc=r2_unc,
        r_c=r_c,
        hourmin_frame=hourmin,
        full_soa_mean=full_soa_mean,
        full_soa_std=full_soa_std,
        full_temp_mean=full_temp_mean,
        full_temp_std=full_temp_std,
    )


@dataclass(frozen=True)
class _SynergyFit:
    params_mean: np.ndarray
    params_var: np.ndarray
    coef: np.ndarray
    k_env_series: np.ndarray
    hour_float: np.ndarray
    temperature: np.ndarray
    radiation: np.ndarray
    humidity: np.ndarray
    o3: np.ndarray
    nox: np.ndarray
    so2: np.ndarray
    soa_obs: np.ndarray
    o3_std: float
    nox_std: float


def _fit_anthropogenic_sensitivity_model(df_base: pd.DataFrame, cat2_outputs: dict) -> _SynergyFit:
    """
    Fit the pollutant-sensitivity WLS model used by the anthropogenic impact figure.

    References:
    - Legacy parameter sensitivity workflow (anthropogenic perturbation with WLS + CS scaling).
    - Gardiner (2009) Stochastic Methods (for the temperature-driven stochastic BVOC proxy).

    Equations:
    - BVOC stochastic proxy: dC = mu(T) dt + sigma(T) dW.
    - Hourly moment forms (legacy):
      E[C|T] = Q0 + (a*T^2)/2 + v0*T
      Var[C|T] = (k^2*T^3)/3 + k*T^2*sigma0 + T*sigma0^2
    - CS modulation: k_env(CS) = beta_max * CS / (CS + CS0)
    - SOA prediction: SOA = M_theta(X) * k_env(CS)

    Parameters:
    - T: temperature (deg C); mu(T), Var(T) from fitted moment relations.
    - O3, NOx: pollutant drivers (perturbed by +2*std in scenarios).
    - RH, SO2, hv: covariates to form interaction proxies.
    - beta_max, CS0: CS scaling parameters fit from observed/base ratio.
    """
    df_long = df_base.reset_index().rename(columns={"Time": "Datetime"})
    df_long["Time"] = df_long["Datetime"]
    df_long["Temperature"] = df_long["temperature_c"]
    df_long["Radiation"] = df_long["rad_w_m2"]
    df_long["Humidity"] = df_long["rh_pct"]
    df_long["Isoprene"] = df_long["bvocs"]
    df_long["Hour_Min"] = df_long["Datetime"].dt.strftime("%H:%M")
    df_long["Hour"] = df_long["Time"].dt.hour

    dataall = df_long[
        [
            "Time",
            "Isoprene",
            "SOA",
            "SO2",
            "NOx",
            "O3",
            "Radiation",
            "Temperature",
            "Humidity",
            "Hour_Min",
            "Hour",
        ]
    ].copy()

    # Fit moment relations on Hour_Min climatology aggregated to hours (legacy approach).
    data_grouped = dataall.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    grouped_by_hour = data_grouped.groupby("Hour")["Isoprene"]
    normality_df = _normal_distribution_fit_by_hour(grouped_by_hour)
    temp_by_hour = data_grouped.groupby("Hour")["Temperature"].mean(numeric_only=True).reset_index()
    normality_df = normality_df.merge(temp_by_hour, on="Hour", how="left")

    T_hour = normality_df["Temperature"].to_numpy(dtype=float)
    mean_values = normality_df["Mean"].to_numpy(dtype=float)
    var_values = (normality_df["StandardDeviation"].to_numpy(dtype=float) ** 2).astype(float)

    mean_filtered = _remove_outliers(mean_values)
    var_filtered = _remove_outliers(var_values)
    T_mean = T_hour[np.isin(mean_values, mean_filtered)]
    T_var = T_hour[np.isin(var_values, var_filtered)]

    params_mean, _ = curve_fit(mean_relation, T_mean, mean_filtered, method="trf", maxfev=10000)
    params_var, _ = curve_fit(std_dev_relation, T_var, var_filtered, method="trf", maxfev=10000)

    # Build Hour_Min climatology and fit WLS with the same reduced feature set as the legacy figure.
    hourmin = dataall.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    hour_float = hourmin["Hour_Min"].map(_hourmin_to_float).to_numpy(dtype=np.float64)

    temperature = hourmin["Temperature"].to_numpy(dtype=np.float64)
    radiation = hourmin["Radiation"].to_numpy(dtype=np.float64)
    humidity = hourmin["Humidity"].to_numpy(dtype=np.float64)
    o3 = hourmin["O3"].to_numpy(dtype=np.float64)
    nox = hourmin["NOx"].to_numpy(dtype=np.float64)
    so2 = hourmin["SO2"].to_numpy(dtype=np.float64)
    soa_obs = hourmin["SOA"].to_numpy(dtype=np.float64)

    hno3 = humidity * nox
    hno3o3 = humidity * nox * o3
    hv = radiation

    mu_bvoc = mean_relation(temperature, *params_mean).astype(np.float64)
    var_bvoc = std_dev_relation(temperature, *params_var).astype(np.float64)
    weights = np.where(np.isfinite(var_bvoc) & (var_bvoc > 0), 1.0 / var_bvoc, 0.0).astype(np.float64)

    X = np.column_stack([hno3 * mu_bvoc, hno3o3 * mu_bvoc, hv * mu_bvoc]).astype(np.float64)
    y = soa_obs.astype(np.float64)
    model = sm.WLS(y, X, weights=weights).fit(cov_type="HC3")
    coef = np.asarray(model.params, dtype=np.float64)

    y_base = X @ coef

    # Fit CS scaling on Hour_Min climatology (same two-step kernel as the paper).
    cs_hourmin = _compute_cs_hourmin_mean(df_base, cat2_outputs, expected_hourmins=hourmin["Hour_Min"])
    cs_vals = cs_hourmin["CS"].to_numpy(dtype=np.float64)
    ratio = np.where(y_base != 0, y / y_base, np.nan)
    ratio = pd.Series(ratio).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    mask = np.isfinite(cs_vals) & (cs_vals > 0) & np.isfinite(ratio) & (ratio > 0)
    x = cs_vals[mask]
    y_ratio = ratio[mask]

    def _cs_kernel(c: np.ndarray, beta_max: float, cs0: float) -> np.ndarray:
        return beta_max * c / (c + cs0)

    if x.size < 5:
        k_env_series = np.ones_like(cs_vals, dtype=np.float64)
    else:
        beta0 = float(np.nanpercentile(y_ratio, 90))
        cs00 = float(np.nanmedian(x))
        beta_max, cs0 = curve_fit(
            _cs_kernel, x, y_ratio, p0=(beta0, cs00), bounds=(0.0, np.inf), maxfev=20000
        )[0]
        k_env_series = _cs_kernel(cs_vals, float(beta_max), float(cs0)).astype(np.float64)
        k_env_series = np.where(np.isfinite(k_env_series), k_env_series, 1.0)

    o3_std = float(dataall["O3"].std(ddof=1))
    nox_std = float(dataall["NOx"].std(ddof=1))

    return _SynergyFit(
        params_mean=np.asarray(params_mean, dtype=np.float64),
        params_var=np.asarray(params_var, dtype=np.float64),
        coef=coef,
        k_env_series=k_env_series,
        hour_float=hour_float,
        temperature=temperature,
        radiation=radiation,
        humidity=humidity,
        o3=o3,
        nox=nox,
        so2=so2,
        soa_obs=soa_obs,
        o3_std=o3_std,
        nox_std=nox_std,
    )


def _simulate_anthropogenic_scenarios(
    fit: _SynergyFit,
    num_simulations: int = 1000,
    seed: int = 20231125,
) -> Dict[str, np.ndarray]:
    """
    Simulate baseline and pollutant-perturbed SOA ensembles for synergy panels.

    References:
    - Legacy anthropogenic perturbation setup: +2*std for O3 and/or NOx.

    Equations:
    - Random BVOCs ~ Normal(mu(T), sigma(T)). The legacy script uses the fitted Var(T) directly
      as the normal scale; preserve that behavior here for compatibility.
    - SOA = (X * BVOC) dot coef * k_env(CS).

    Parameters:
    - num_simulations: Monte Carlo ensemble size.
    - seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    temperature = fit.temperature.astype(np.float64)
    humidity = fit.humidity.astype(np.float64)
    hv = fit.radiation.astype(np.float64)
    o3_base = fit.o3.astype(np.float64)
    nox_base = fit.nox.astype(np.float64)

    mu_bvoc = mean_relation(temperature, *fit.params_mean).astype(np.float64)
    var_bvoc = std_dev_relation(temperature, *fit.params_var).astype(np.float64)
    scale = np.where(np.isfinite(var_bvoc) & (var_bvoc > 0), var_bvoc, 0.0).astype(np.float64)

    bvoc = rng.normal(loc=mu_bvoc[None, :], scale=scale[None, :], size=(num_simulations, temperature.size)).astype(
        np.float64
    )

    def _predict(ozone: np.ndarray, nox: np.ndarray) -> np.ndarray:
        hno3 = (humidity * nox).astype(np.float64)
        hno3o3 = (humidity * nox * ozone).astype(np.float64)
        X = np.stack([hno3, hno3o3, hv], axis=1).astype(np.float64)  # (T, 3)
        y_base = (bvoc[:, :, None] * X[None, :, :]).astype(np.float64) @ fit.coef.reshape(-1, 1)
        y_base = y_base[:, :, 0]
        return (y_base * fit.k_env_series[None, :]).astype(np.float64)

    o3_plus = (o3_base + 2.0 * float(fit.o3_std)).astype(np.float64)
    nox_plus = (nox_base + 2.0 * float(fit.nox_std)).astype(np.float64)

    return {
        "Baseline": _predict(o3_base, nox_base),
        "+O3": _predict(o3_plus, nox_base),
        "+NOx": _predict(o3_base, nox_plus),
        "Combined": _predict(o3_plus, nox_plus),
    }


def _simulate_warming_ensembles(
    fit: _KernelFit, num_simulations: int = 1000, seed: int = 20231125
) -> Dict[str, Dict[str, np.ndarray]]:
    # References: Monte Carlo propagation of temperature-driven precursor volatility.
    # Equations:
    # - BVOC_s ~ Normal(mean_relation(T+delta), std_dev_relation(T+delta))
    # - SOA_s(t) = (X(t) * BVOC_s(t)) dot coef * k_env(CS(t))
    # Parameters:
    # - num_simulations: ensemble size.
    # - seed: RNG seed to keep deterministic reproduction of the legacy ensemble.
    np.random.seed(seed)
    random.seed(seed)

    hourmin = fit.hourmin_frame
    variables = ("HNO3", "H2SO4", "H2SO4O3", "HNO3O3", "O3hv", "K", "Radiation")
    base_feats = np.column_stack([hourmin[v].to_numpy(dtype=float) for v in variables])
    k_env = fit.k_env_series.astype(float)

    T0 = hourmin["Temperature"].to_numpy(dtype=float)
    hour_float = hourmin["Hour_Float"].to_numpy(dtype=float)

    scenarios = {
        "Baseline": 0.0,
        "+1.5C": 1.5,
        "+2.0C": 2.0,
        "+3.0C": 3.0,
    }
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, delta in scenarios.items():
        T = T0 + float(delta)
        mu = mean_relation(T, *fit.params_mean)
        # Preserve legacy behavior: std_dev_relation is used directly as the Normal scale.
        scale = std_dev_relation(T, *fit.params_var)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 0.0)

        soa = np.empty((num_simulations, T.size), dtype=np.float32)
        for i in range(num_simulations):
            bvoc = np.random.normal(loc=mu, scale=scale, size=T.size).astype(np.float64)
            X = (base_feats * bvoc[:, None]).astype(np.float64)
            y_base = X @ fit.coef
            soa[i, :] = (y_base * k_env).astype(np.float32)

        out[name] = {
            "Temperature": T.astype(np.float64),
            "HourFloat": hour_float.astype(np.float64),
            "SOA": soa,
        }
    return out


def _compute_exceedance_landscape(
    fit: _KernelFit,
    delta_t_max: float = 3.0,
    sigma_max: float = 4.0,
    delta_t_steps: int = 31,
    sigma_steps: int = 41,
    num_simulations: int = 1000,
    simulation_size: int = 100,
    seed: int = 20231125,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # References: Threshold exceedance probability from Monte Carlo samples.
    # Equations:
    # - Draw BVOC ~ Normal(mu(T), sqrt(Var(T))) as in the legacy threshold script.
    # - Predict SOA = (X * BVOC) @ coef * mean(k_env(CS)).
    # - P_exceed(T, k) = mean( I(SOA > mean_base + k * std_base) ).
    # Parameters:
    # - delta_t_max: maximum warming relative to mean temperature (deg C).
    # - sigma_max: maximum threshold multiple (sigma units).
    # - simulation_size: number of draws per batch; num_simulations batches are pooled.
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    hourmin = fit.hourmin_frame
    delta_t = np.linspace(0.0, float(delta_t_max), int(delta_t_steps), dtype=np.float64)
    sigma_grid = np.linspace(0.0, float(sigma_max), int(sigma_steps), dtype=np.float64)
    thresholds = fit.full_soa_mean + sigma_grid * fit.full_soa_std

    T_mean = float(fit.full_temp_mean)
    T_values = T_mean + delta_t

    variables = ("HNO3", "H2SO4", "H2SO4O3", "HNO3O3", "O3hv", "K", "Radiation")
    base_row = pd.DataFrame(
        {
            "Temperature": [float(hourmin["Temperature"].mean(numeric_only=True))],
            "Radiation": [float(hourmin["Radiation"].mean(numeric_only=True))],
            "Humidity": [float(hourmin["Humidity"].mean(numeric_only=True))],
            "O3": [float(hourmin["O3"].mean(numeric_only=True))],
            "NOx": [float(hourmin["NOx"].mean(numeric_only=True))],
            "SO2": [float(hourmin["SO2"].mean(numeric_only=True))],
        }
    )
    base_row["HNO3"] = base_row["Humidity"] * base_row["NOx"]
    base_row["H2SO4"] = base_row["Humidity"] * base_row["SO2"]
    base_row["H2SO4O3"] = base_row["Humidity"] * base_row["SO2"] * base_row["O3"]
    base_row["HNO3O3"] = base_row["Humidity"] * base_row["NOx"] * base_row["O3"]
    base_row["O3hv"] = base_row["O3"] * base_row["Radiation"]
    base_row["K"] = 1.0

    base_feats = np.array([base_row[v].iloc[0] for v in variables], dtype=float)

    # Build Hour_Min climatology for CS scaling ratio, then use mean k_env as the threshold script does.
    X0 = np.column_stack([hourmin[v].to_numpy(dtype=float) for v in variables])
    bvoc0 = mean_relation(hourmin["Temperature"].to_numpy(dtype=float), *fit.params_mean)
    y_base_cs = (X0 * bvoc0[:, None]) @ fit.coef
    y_obs_cs = hourmin["SOA"].to_numpy(dtype=float)

    ratio = np.where(y_base_cs != 0, y_obs_cs / y_base_cs, np.nan)
    ratio = pd.Series(ratio).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    cs_vals = hourmin["CS"].to_numpy(dtype=float)
    mask = np.isfinite(cs_vals) & (cs_vals > 0) & np.isfinite(ratio) & (ratio > 0)
    x = cs_vals[mask]
    y_ratio = ratio[mask]

    def _cs_kernel(c: np.ndarray, beta_max: float, cs0: float) -> np.ndarray:
        return beta_max * c / (c + cs0)

    beta0 = float(np.nanpercentile(y_ratio, 90)) if y_ratio.size else 1.0
    cs00 = float(np.nanmedian(x)) if x.size else 1.0
    beta_max, cs0 = curve_fit(
        _cs_kernel, x, y_ratio, p0=(beta0, cs00), bounds=(0.0, np.inf), maxfev=20000
    )[0]
    k_env_series = _cs_kernel(cs_vals, float(beta_max), float(cs0))
    k_env_mean = float(np.nanmean(k_env_series)) if np.isfinite(k_env_series).any() else 1.0

    draw_count = int(num_simulations) * int(simulation_size)
    prob_grid = np.zeros((sigma_grid.size, delta_t.size), dtype=np.float64)

    for t_idx, T in enumerate(T_values):
        mu = float(mean_relation(np.array([T], dtype=float), *fit.params_mean)[0])
        var = float(std_dev_relation(np.array([T], dtype=float), *fit.params_var)[0])
        sigma = float(np.sqrt(max(var, 0.0))) if var > 0 else 0.0
        bvoc = np.random.normal(loc=mu, scale=sigma, size=draw_count).astype(np.float64)
        X = (base_feats[None, :] * bvoc[:, None]).astype(np.float64)
        y_base = X @ fit.coef
        y = y_base * k_env_mean
        prob_grid[:, t_idx] = np.mean(y[:, None] > thresholds[None, :], axis=0)

    return delta_t, sigma_grid, prob_grid


def _panel_label(ax: plt.Axes, label: str, x: float = 0.01, y: float = 0.99) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        clip_on=False,
    )


def main() -> None:
    _set_nature_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    df_base = load_base_data(cfg)
    _, _, cat2_outputs, _, _ = load_cached_results()

    fit = _fit_temp_bvoc_kernel_and_cs(df_base, cat2_outputs)
    ensembles = _simulate_warming_ensembles(fit, num_simulations=1000, seed=20231125)
    delta_t, sigma_grid, prob_grid = _compute_exceedance_landscape(
        fit,
        delta_t_max=3.0,
        sigma_max=4.0,
        delta_t_steps=121,
        sigma_steps=41,
        num_simulations=1000,
        simulation_size=100,
        seed=20231125,
    )
    synergy_fit = _fit_anthropogenic_sensitivity_model(df_base, cat2_outputs)
    synergy_ensembles = _simulate_anthropogenic_scenarios(synergy_fit, num_simulations=1000, seed=20231125)

    colors = {
        "Baseline": "#2F5597",  # Nature-friendly blue
        "+1.5C": "#E3BA22",  # muted gold (colorblind-friendly)
        "+2.0C": "#F28E2B",  # orange
        "+3.0C": "#B22222",  # firebrick red
    }

    fig = plt.figure(figsize=(7.35, 7.15))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.38, 1.38, 1.12, 0.58],
        height_ratios=[1.0, 1.05, 0.92],
        hspace=0.78,
        wspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_pen = fig.add_subplot(gs[0, 3])
    ax_c = fig.add_subplot(gs[1, 0:2])
    # Panel e: extreme risk landscape (heatmap).
    ax_e = fig.add_subplot(gs[1, 2:4])

    # Panel f/g/h: anthropogenic synergy evidence (time series, statistics, attribution).
    ax_f = fig.add_subplot(gs[2, 0:2])
    gs_sy = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 2:4], wspace=0.38, width_ratios=[1.35, 0.65])
    ax_g = fig.add_subplot(gs_sy[0, 0])
    ax_h = fig.add_subplot(gs_sy[0, 1])

    # Panel a: distributions (KDE)
    x_grid = np.linspace(0.0, 20.0, 300)
    for name, payload in ensembles.items():
        soa = payload["SOA"].astype(np.float64)
        sample = soa.ravel()
        if sample.size > 120000:
            sample = np.random.choice(sample, size=120000, replace=False)
        sample = sample[np.isfinite(sample) & (sample >= 0)]
        if sample.size < 200:
            continue
        kde = gaussian_kde(sample)
        y_kde = kde(x_grid)
        ax_a.fill_between(x_grid, y_kde, color=colors[name], alpha=0.30, linewidth=0.0)
        ax_a.plot(x_grid, y_kde, color=colors[name], linewidth=1.35)

    ax_a.set_xlim(0, 20)
    ax_a.set_xlabel(r"SOA ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("Distribution", pad=3, fontsize=8.2)
    eq_text = (
        r"$\mathrm{SOA}=k_{\mathrm{env}}(\mathrm{CS})\,C_{T}\,M_{\theta}(\mathbf{X})$"
        "\n"
        r"$k_{\mathrm{env}}(\mathrm{CS})=\beta_{\max}\,\frac{\mathrm{CS}}{\mathrm{CS}+\mathrm{CS}_{0}}$"
    )
    ax_a.text(
        0.50,
        0.88,
        eq_text,
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#111111",
        linespacing=1.1,
    )
    ax_a.legend(
        handles=[
            Line2D([0], [0], color=colors["Baseline"], lw=2.0, label="Baseline"),
            Line2D([0], [0], color=colors["+1.5C"], lw=2.0, label=r"+1.5$^\circ$C"),
            Line2D([0], [0], color=colors["+2.0C"], lw=2.0, label=r"+2.0$^\circ$C"),
            Line2D([0], [0], color=colors["+3.0C"], lw=2.0, label=r"+3.0$^\circ$C"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.2,
        handlelength=2.2,
        labelspacing=0.3,
        borderpad=0.2,
    )

    # Panel b: mean response with raw Monte Carlo trajectories + band (5-95%) + mean curve
    np.random.seed(20231125)
    for name, payload in ensembles.items():
        T = payload["Temperature"].astype(np.float64)
        soa = payload["SOA"].astype(np.float64)
        order = np.argsort(T)
        x = T[order]

        # Raw Monte Carlo trajectories (very low alpha, behind the mean curve)
        if soa.shape[0] >= 10:
            pick = np.random.choice(soa.shape[0], size=min(80, soa.shape[0]), replace=False)
            for idx in pick:
                y_sim = soa[idx, :].astype(np.float64)[order]
                ax_b.plot(x, y_sim, color=colors[name], alpha=0.003, linewidth=0.35, zorder=0)

        mean_y = np.nanmean(soa, axis=0)
        q05 = np.nanquantile(soa, 0.05, axis=0)
        q95 = np.nanquantile(soa, 0.95, axis=0)
        y_mean = mean_y[order]
        y_lo = q05[order]
        y_hi = q95[order]
        sm_mean = lowess(y_mean, x, frac=0.25, return_sorted=True)
        sm_lo = lowess(y_lo, x, frac=0.25, return_sorted=True)
        sm_hi = lowess(y_hi, x, frac=0.25, return_sorted=True)
        ax_b.fill_between(sm_mean[:, 0], sm_lo[:, 1], sm_hi[:, 1], color=colors[name], alpha=0.18)
        ax_b.plot(sm_mean[:, 0], sm_mean[:, 1], color=colors[name], linewidth=1.8)

    ax_b.set_xlabel(r"Temperature ($^\circ\mathrm{C}$)")
    ax_b.set_ylabel(r"Average SOA ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_b.set_title("Mean response", pad=3, fontsize=8.2)
    ax_b.text(
        0.02,
        0.66,
        rf"$R_{{uc}}^2={fit.r2_unc:.3f}$",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#111111",
    )

    # Panel penalty: variance penalty summary (mean, SD, P99 ratios vs Baseline).
    scenario_order = ["Baseline", "+1.5C", "+2.0C", "+3.0C"]
    delta_x = np.array([0.0, 1.5, 2.0, 3.0], dtype=float)
    stats = {}
    for name in scenario_order:
        soa_flat = ensembles[name]["SOA"].astype(np.float64).ravel()
        soa_flat = soa_flat[np.isfinite(soa_flat)]
        stats[name] = (float(np.mean(soa_flat)), float(np.std(soa_flat, ddof=1)))
    base_mean, base_sd = stats["Baseline"]
    mean_ratios = [stats[n][0] / base_mean if base_mean > 0 else np.nan for n in scenario_order]
    sd_ratios = [stats[n][1] / base_sd if base_sd > 0 else np.nan for n in scenario_order]
    q99_base = float(np.nanquantile(ensembles["Baseline"]["SOA"].astype(np.float64), 0.99))
    p99_ratios = []
    for name in scenario_order:
        q99 = float(np.nanquantile(ensembles[name]["SOA"].astype(np.float64), 0.99))
        p99_ratios.append(q99 / q99_base if q99_base > 0 else np.nan)
    ax_pen.plot(delta_x, mean_ratios, color="#111111", marker="o", lw=1.2, ms=3.0)
    ax_pen.plot(delta_x, sd_ratios, color="#6A6A6A", marker="o", lw=1.2, ms=3.0)
    ax_pen.plot(delta_x, p99_ratios, color="#8C1515", marker="o", lw=1.2, ms=3.0)
    ax_pen.set_title("Variance penalty", pad=3, fontsize=8.2)
    ax_pen.set_xlabel(r"$\Delta T$ ($^\circ\mathrm{C}$)", labelpad=6.0)
    ax_pen.set_ylabel("Ratio to baseline")
    ax_pen.set_ylim(0.8, max(float(np.nanmax(sd_ratios)), float(np.nanmax(p99_ratios)), 1.2) * 1.05)
    ax_pen.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_pen.tick_params(axis="both", which="major", labelsize=8.6, length=2.8, width=0.8)
    ax_pen.grid(False)
    ax_pen.set_xlim(-0.15, 3.35)
    ax_pen.set_xticks([0, 1.5, 2.0, 3.0])
    ax_pen.set_xticklabels(["0", "1.5", "2", "3"], rotation=45, ha="right")
    ax_pen.tick_params(axis="x", pad=0.6, labelsize=8.5)
    ax_pen.xaxis.set_label_coords(0.5, -0.32)
    pen_trans = ax_pen.get_yaxis_transform()
    ax_pen.text(
       1.1,
        float(mean_ratios[-1]) - 0.10,
        "Mean",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#111111",
        transform=pen_trans,
        clip_on=False,
    )
    ax_pen.text(
       1.1,
        float(sd_ratios[-1]) + 0.15,
        "SD",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#6A6A6A",
        transform=pen_trans,
        clip_on=False,
    )
    ax_pen.text(
        1.1,
        float(p99_ratios[-1]) + 0.08,
        "P99",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#8C1515",
        transform=pen_trans,
        clip_on=False,
    )

    # Panel d: diurnal cycle (focus Baseline vs +3.0C) with raw Monte Carlo trajectories + band + mean
    hour_float = ensembles["Baseline"]["HourFloat"].astype(np.float64)
    diurnal_names = ("Baseline", "+3.0C")
    ax_c.axvspan(6, 18, color="#FFF2CC", alpha=0.25, zorder=0)
    ax_c.axvspan(0, 6, color="#EAEAEA", alpha=0.25, zorder=0)
    ax_c.axvspan(18, 24, color="#EAEAEA", alpha=0.25, zorder=0)
    for name in diurnal_names:
        payload = ensembles[name]
        soa = payload["SOA"].astype(np.float64)

        if soa.shape[0] >= 10:
            pick = np.random.choice(soa.shape[0], size=min(80, soa.shape[0]), replace=False)
            for idx in pick:
                ax_c.plot(hour_float, soa[idx, :], color=colors[name], alpha=0.003, linewidth=0.35, zorder=0)

        mean_y = np.nanmean(soa, axis=0)
        q25 = np.nanquantile(soa, 0.25, axis=0)
        q75 = np.nanquantile(soa, 0.75, axis=0)
        sm_mean = lowess(mean_y, hour_float, frac=0.06, return_sorted=True)
        sm_q25 = lowess(q25, hour_float, frac=0.06, return_sorted=True)
        sm_q75 = lowess(q75, hour_float, frac=0.06, return_sorted=True)
        ax_c.fill_between(sm_mean[:, 0], sm_q25[:, 1], sm_q75[:, 1], color=colors[name], alpha=0.18, linewidth=0.0)
        ax_c.plot(sm_mean[:, 0], sm_mean[:, 1], color=colors[name], linewidth=1.8)

    mean_soa = float(fit.full_soa_mean)
    std_soa = float(fit.full_soa_std)
    thr_color = "#2B2B2B"
    ax_c.axhline(mean_soa, color=thr_color, linestyle="--", linewidth=1.1, alpha=0.85, zorder=4)
    ax_c.axhline(mean_soa + 1.0 * std_soa, color=thr_color, linestyle=":", linewidth=1.1, alpha=0.85, zorder=4)
    ax_c.axhline(mean_soa + 2.0 * std_soa, color=thr_color, linestyle="-.", linewidth=1.1, alpha=0.75, zorder=4)

    ax_c.set_xlim(0, 24)
    ax_c.set_xlabel("Local time (h)")
    ax_c.set_ylabel(r"SOA ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_c.set_title("Diurnal cycle", pad=3, fontsize=8.2)
    ax_c.set_xticks([0, 6, 12, 18, 24])
    y_min_c, y_max_c = ax_c.get_ylim()
    y_label_c = y_min_c + 0.08 * (y_max_c - y_min_c)
    ax_c.text(2.0, y_label_c, "Nighttime", ha="left", va="bottom", fontsize=7.0, color="#5A5A5A")
    ax_c.text(12.0, y_label_c, "Daytime", ha="center", va="bottom", fontsize=7.0, color="#5A5A5A")
    ax_c.text(22.0, y_label_c, "Nighttime", ha="right", va="bottom", fontsize=7.0, color="#5A5A5A")

    # Table: scenario mean and SD over the full ensemble (focus Baseline vs +3.0C).
    rows = []
    label_map = {"Baseline": "Base", "+1.5C": "+1.5", "+2.0C": "+2.0", "+3.0C": "+3.0"}
    for name in ("Baseline", "+3.0C"):
        m, s = stats[name]
        rows.append((label_map[name], m, s))
    table = ax_c.table(
        cellText=[[r[0], f"{r[1]:.2f}", f"{r[2]:.2f}"] for r in rows],
        colLabels=["Scenario", "Mean", "SD"],
        cellLoc="center",
        colLoc="center",
        bbox=(0.02, 0.70, 0.56, 0.28),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#B8B8B8")
        cell.set_linewidth(0.30)
        if r == 0:
            cell.set_facecolor("#F2F2F2")
        else:
            cell.set_facecolor("white")
        if r == 0:
            cell.get_text().set_fontweight("bold")
        else:
            cell.get_text().set_fontweight("normal")

    legend_handles = [
        Line2D([0], [0], color=colors["Baseline"], lw=1.8, label="Base"),
        Line2D([0], [0], color=colors["+3.0C"], lw=1.8, label="+3.0C"),
        Line2D([0], [0], color=thr_color, lw=1.1, ls="--", label="Mean"),
        Line2D([0], [0], color=thr_color, lw=1.1, ls=":", label="+1 SD"),
        Line2D([0], [0], color=thr_color, lw=1.1, ls="-.", label="+2 SD"),
    ]
    ax_c.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.96),
        frameon=False,
        fontsize=7.0,
        handlelength=2.2,
        labelspacing=0.3,
        borderpad=0.2,
    )

    # Panel e: extreme risk landscape (heatmap + contours).
    x_grid, y_grid = np.meshgrid(delta_t, sigma_grid)
    cmap = LinearSegmentedColormap.from_list("risk_red", ["#F7F7F7", "#FDE0DD", "#A50026"])
    levels = np.linspace(0.0, 1.0, 15)
    cf = ax_e.contourf(x_grid, y_grid, prob_grid, levels=levels, cmap=cmap, vmin=0.0, vmax=1.0)
    contour_color = "#4A4A4A"
    contour_001 = ax_e.contour(
        x_grid,
        y_grid,
        prob_grid,
        levels=[0.01],
        colors=contour_color,
        linewidths=0.5,
        linestyles="--",
        alpha=0.9,
    )
    contour_005 = ax_e.contour(
        x_grid,
        y_grid,
        prob_grid,
        levels=[0.05],
        colors=contour_color,
        linewidths=0.6,
        linestyles="--",
        alpha=0.9,
    )
    contour_01 = ax_e.contour(
        x_grid,
        y_grid,
        prob_grid,
        levels=[0.1],
        colors=contour_color,
        linewidths=0.8,
        linestyles="--",
        alpha=0.9,
    )
    contour_05 = ax_e.contour(
        x_grid,
        y_grid,
        prob_grid,
        levels=[0.5],
        colors=contour_color,
        linewidths=1.5,
        linestyles="-",
        alpha=0.9,
    )
    contour_09 = ax_e.contour(
        x_grid,
        y_grid,
        prob_grid,
        levels=[0.9],
        colors=contour_color,
        linewidths=0.8,
        linestyles="-",
        alpha=0.9,
    )
    label_001 = ax_e.clabel(
        contour_001,
        fmt={0.01: "P=0.01"},
        inline=True,
        fontsize=6.0,
        colors=contour_color,
        manual=[(0.30, 3.8)],
    )
    label_005 = ax_e.clabel(
        contour_005,
        fmt={0.05: "P=0.05"},
        inline=True,
        fontsize=6.2,
        colors=contour_color,
        manual=[(0.55, 3.4)],
    )
    label_01 = ax_e.clabel(
        contour_01,
        fmt={0.1: "P=0.1"},
        inline=True,
        fontsize=6.4,
        colors=contour_color,
        manual=[(0.95, 2.7)],
    )
    label_05 = ax_e.clabel(
        contour_05,
        fmt={0.5: "P=0.5"},
        inline=True,
        fontsize=6.8,
        colors=contour_color,
        manual=[(1.75, 1.15)],
    )
    label_09 = ax_e.clabel(
        contour_09,
        fmt={0.9: "P=0.9"},
        inline=True,
        fontsize=6.4,
        colors=contour_color,
        manual=[(2.65, 0.5)],
    )
    for labels in (label_001, label_005, label_01, label_05, label_09):
        for txt in labels:
            txt.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground="white")])
    ax_e.axvline(2.0, color="#111111", linestyle="--", linewidth=1.0, alpha=0.85)
    ax_e.axhline(3.0, color="#111111", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_e.text(
        1.92,
        3.95,
        "Volatility\nTipping Point",
        rotation=90,
        ha="right",
        va="top",
        fontsize=6.6,
        color="#111111",
    )
    ax_e.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_e.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax_e.set_xlim(0.0, 3.0)
    ax_e.set_ylim(0.0, 4.0)
    ax_e.set_yticks([0, 1, 2, 3, 4])
    ax_e.set_yticklabels(["0", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$", r"$4\sigma$"])
    ax_e.set_xlabel(r"Warming ($\Delta T$, $^\circ\mathrm{C}$)")
    ax_e.set_ylabel(r"Extreme threshold (in $\sigma_{base}$)")
    ax_e.set_title("Extreme risk landscape", pad=3, fontsize=8.2)
    ax_e.tick_params(axis="both", which="major", labelsize=8.6, length=2.8, width=0.8)
    ax_e.grid(False)

    cax = inset_axes(
        ax_e,
        width="3.5%",
        height="72%",
        loc="center left",
        bbox_to_anchor=(1.05, 0.05, 0.90, 0.90),
        bbox_transform=ax_e.transAxes,
        borderpad=0.2,
    )
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("Exceedance probability", fontsize=7.4, labelpad=2.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=6.4, length=1.2, width=0.6, pad=0.4)

    # Panel f/g/h: anthropogenic synergy evidence (ref: SOA_stochastic_anthropogenic_impact.png).
    poll_colors = {
        "Baseline": "#9E9E9E",  # baseline (pollutants) grey
        "+O3": colors["+1.5C"],  # ozone increased (gold)
        "+NOx": colors["Baseline"],  # NOx increased (blue)
        "Combined": colors["+3.0C"],  # combined perturbation (red)
        "Additive": "#111111",  # additive expectation (black dashed)
        "Synergy": "#00A087",  # synergy highlight (teal)
    }

    hour_float = synergy_fit.hour_float.astype(np.float64)
    mask_zoom = (hour_float >= 10.0) & (hour_float <= 16.0)
    x_zoom = hour_float[mask_zoom]

    base_mean = np.nanmean(synergy_ensembles["Baseline"], axis=0)
    o3_mean = np.nanmean(synergy_ensembles["+O3"], axis=0)
    nox_mean = np.nanmean(synergy_ensembles["+NOx"], axis=0)
    comb_mean = np.nanmean(synergy_ensembles["Combined"], axis=0)
    add_mean = o3_mean + nox_mean - base_mean

    base_zoom = base_mean[mask_zoom]
    o3_zoom = o3_mean[mask_zoom]
    nox_zoom = nox_mean[mask_zoom]
    comb_zoom = comb_mean[mask_zoom]
    add_zoom = add_mean[mask_zoom]

    mean_excess = float(np.nanmean(comb_zoom - add_zoom)) if comb_zoom.size else float("nan")
    mean_add = float(np.nanmean(add_zoom)) if add_zoom.size else float("nan")
    excess_pct = mean_excess / mean_add * 100.0 if mean_add > 0 else float("nan")

    # Panel f: synergistic gap time series (10:00-16:00).
    def _smooth(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 8 or not np.isfinite(y).any():
            return y
        sm_xy = lowess(y, x, frac=frac, return_sorted=True)
        return np.interp(x, sm_xy[:, 0], sm_xy[:, 1])

    base_zoom_s = _smooth(x_zoom, base_zoom, frac=0.08)
    o3_zoom_s = _smooth(x_zoom, o3_zoom, frac=0.08)
    nox_zoom_s = _smooth(x_zoom, nox_zoom, frac=0.08)
    comb_zoom_s = _smooth(x_zoom, comb_zoom, frac=0.08)
    add_zoom_s = _smooth(x_zoom, add_zoom, frac=0.08)

    ax_f.fill_between(
        x_zoom,
        add_zoom_s,
        comb_zoom_s,
        where=(comb_zoom_s > add_zoom_s),
        color=poll_colors["Synergy"],
        alpha=0.18,
        linewidth=0.0,
    )
    ax_f.plot(x_zoom, base_zoom_s, color=poll_colors["Baseline"], lw=1.0, alpha=0.90)
    ax_f.plot(x_zoom, o3_zoom_s, color=poll_colors["+O3"], lw=1.0, alpha=0.95)
    ax_f.plot(x_zoom, nox_zoom_s, color=poll_colors["+NOx"], lw=1.0, alpha=0.95)
    ax_f.plot(x_zoom, add_zoom_s, color=poll_colors["Additive"], lw=1.4, ls="--")
    ax_f.plot(x_zoom, comb_zoom_s, color=poll_colors["Combined"], lw=1.8)


    ax_f.set_title("Synergistic gap (10:00-16:00)", pad=3, fontsize=8.2)
    ax_f.set_xlabel("Local time (h)")
    ax_f.set_ylabel(r"SOA ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_f.set_xlim(10, 16)
    ax_f.set_xticks([10, 12, 14, 16])
    ax_f.grid(False)
    ax_f.legend(
        handles=[
            Line2D([0], [0], color=poll_colors["Baseline"], lw=1.0, label="Baseline"),
            Line2D([0], [0], color=poll_colors["+O3"], lw=1.0, label="+O3"),
            Line2D([0], [0], color=poll_colors["+NOx"], lw=1.0, label="+NOx"),
            Line2D([0], [0], color=poll_colors["Additive"], lw=1.4, ls="--", label="Additive expectation"),
            Line2D([0], [0], color=poll_colors["Combined"], lw=1.8, label="Actual combined"),
            Patch(facecolor=poll_colors["Synergy"], alpha=0.18, edgecolor="none", label="Synergistic excess"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.05),
        frameon=False,
        fontsize=7.6,
        ncol=3,
        columnspacing=0.9,
        handlelength=1.8,
    )

    # Panel g: non-linear scaling (linear expectation vs actual combined).
    # References: anthropogenic perturbation ensembles from _simulate_anthropogenic_scenarios.
    # Equations:
    #   SOA_linear = SOA_base + (SOA_O3 - SOA_base) + (SOA_NOx - SOA_base) = SOA_O3 + SOA_NOx - SOA_base
    #   SOA_actual = SOA_combined
    # Parameters: peak-hour window 10:00-16:00; temperature from synergy_fit.temperature; N = num_simulations.
    base_win = synergy_ensembles["Baseline"][:, mask_zoom]
    o3_win = synergy_ensembles["+O3"][:, mask_zoom]
    nox_win = synergy_ensembles["+NOx"][:, mask_zoom]
    comb_win = synergy_ensembles["Combined"][:, mask_zoom]

    base_m = np.nanmean(base_win, axis=1)
    o3_m = np.nanmean(o3_win, axis=1)
    nox_m = np.nanmean(nox_win, axis=1)
    comb_m = np.nanmean(comb_win, axis=1)
    add_m = (o3_m + nox_m - base_m).astype(np.float64)
    delta_syn = (comb_m - add_m).astype(np.float64)

    valid_m = np.isfinite(base_m) & np.isfinite(o3_m) & np.isfinite(nox_m) & np.isfinite(comb_m)
    base_m = base_m[valid_m]
    o3_m = o3_m[valid_m]
    nox_m = nox_m[valid_m]
    comb_m = comb_m[valid_m]
    add_m = add_m[valid_m]
    delta_syn = delta_syn[valid_m]

    x_sc = add_mean[mask_zoom].astype(np.float64)
    y_sc = comb_mean[mask_zoom].astype(np.float64)
    t_sc = synergy_fit.temperature.astype(np.float64)[mask_zoom]
    mask_sc = np.isfinite(x_sc) & np.isfinite(y_sc) & np.isfinite(t_sc)
    x_sc = x_sc[mask_sc]
    y_sc = y_sc[mask_sc]
    t_sc = t_sc[mask_sc]

    lim_lo = 0.0
    lim_hi = float(np.nanpercentile(np.concatenate([x_sc, y_sc]), 99.5))
    lim_hi = max(25.0, lim_hi * 1.05)

    residuals = y_sc - x_sc
    slope = float("nan")
    intercept = float("nan")
    r2_fit = float("nan")
    if x_sc.size >= 2 and np.isfinite(x_sc).all() and np.isfinite(y_sc).all():
        slope, intercept = np.polyfit(x_sc, y_sc, 1)
        y_fit = slope * x_sc + intercept
        ss_res = float(np.sum((y_sc - y_fit) ** 2))
        ss_tot = float(np.sum((y_sc - np.mean(y_sc)) ** 2))
        if ss_tot > 0:
            r2_fit = 1.0 - ss_res / ss_tot
    bias_mean = float(np.nanmean(residuals)) if residuals.size else float("nan")
    mean_expected = float(np.nanmean(x_sc)) if x_sc.size else float("nan")
    uplift_pct = bias_mean / mean_expected * 100.0 if mean_expected > 0 else float("nan")
    pct_above = float(np.nanmean(residuals > 0) * 100.0) if residuals.size else float("nan")
    norm_temp = Normalize(vmin=float(np.nanmin(t_sc)), vmax=float(np.nanmax(t_sc)))
    ax_g.scatter(
        x_sc,
        y_sc,
        c=t_sc,
        cmap=plt.cm.coolwarm,
        norm=norm_temp,
        s=20,
        alpha=0.6,
        linewidths=0.0,
        zorder=2,
    )
    if x_sc.size >= 12:
        num_bins = 6
        bin_edges = np.linspace(lim_lo, lim_hi, num_bins + 1)
        bin_idx = np.digitize(x_sc, bin_edges) - 1
        bin_x = []
        bin_y = []
        bin_y_std = []
        for i in range(num_bins):
            mask = bin_idx == i
            if not np.any(mask):
                continue
            bin_x.append(float(np.nanmean(x_sc[mask])))
            bin_y.append(float(np.nanmean(y_sc[mask])))
            bin_y_std.append(float(np.nanstd(y_sc[mask], ddof=0)))
        if bin_x:
            ax_g.errorbar(
                bin_x,
                bin_y,
                yerr=bin_y_std,
                fmt="o",
                ms=2.8,
                color="#111111",
                ecolor="#111111",
                elinewidth=0.7,
                capsize=1.5,
                zorder=5,
            )
    ax_g.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#111111", lw=1.0, ls="--", zorder=1)
    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.array([lim_lo, lim_hi], dtype=float)
        ax_g.plot(x_line, slope * x_line + intercept, color="#8C1515", lw=1.2, zorder=4)

    uplift_mean_text = f"{bias_mean:+.2f}" if np.isfinite(bias_mean) else "NA"
    uplift_pct_text = f"{uplift_pct:+.1f}%" if np.isfinite(uplift_pct) else "NA"
    pct_above_text = f"{pct_above:.0f}%" if np.isfinite(pct_above) else "NA"
    ax_g.legend(
        handles=[
            Line2D([0], [0], color="#111111", marker="o", lw=0.8, ms=3.0, label="Bin mean +/- 1 SD"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        fontsize=6.2,
        handlelength=1.2,
        labelspacing=0.3,
        borderpad=0.2,
    )
    ax_g.text(
        0.03,
        0.82,
        f"Combined uplift:\n{uplift_mean_text} ({uplift_pct_text})\nP(y>x)={pct_above_text}",
        transform=ax_g.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#111111",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.8),
    )

    cax = inset_axes(
        ax_g,
        width="50%",
        height="10%",
        loc="lower right",
        bbox_to_anchor=(0.05, 0.06, 0.90, 0.90),
        bbox_transform=ax_g.transAxes,
        borderpad=0.5,
    )
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_temp, cmap=plt.cm.coolwarm), cax=cax, orientation="horizontal")
    cbar.set_label(r"Temperature ($^\circ$C)", fontsize=8, labelpad=0.5)
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_coords(0.5, -0.2)
    tick_vals = np.linspace(float(norm_temp.vmin), float(norm_temp.vmax), 4)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{v:.1f}" for v in tick_vals])
    cbar.ax.tick_params(labelsize=6, length=1.1, width=0.6, pad=0.2)


    ax_g.set_xlim(lim_lo, lim_hi)
    ax_g.set_ylim(lim_lo, lim_hi)
    ax_g.set_xlabel(r"Linear Expectation ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_g.set_ylabel(r"Actual Combined ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_g.set_title("Non-linear scaling", pad=3, fontsize=8.2)

    # Panel h: contribution breakdown (additive vs combined).
    base_mean_win = float(np.nanmean(base_m)) if base_m.size else 0.0
    delta_o3 = float(np.nanmean(o3_m - base_m)) if base_m.size else 0.0
    delta_nox = float(np.nanmean(nox_m - base_m)) if base_m.size else 0.0
    delta_syn_mean = float(np.nanmean(delta_syn)) if delta_syn.size else 0.0

    x_pos = np.array([0.0, 1.0], dtype=float)
    width = 0.62
    base_col = "#C7C7C7"

    # Additive expectation bar.
    b0 = base_mean_win
    b1 = b0 + delta_o3
    b2 = b1 + delta_nox
    ax_h.bar(x_pos[0], b0, width=width, color=base_col, edgecolor="#333333", linewidth=0.6)
    ax_h.bar(x_pos[0], delta_o3, width=width, bottom=b0, color=poll_colors["+O3"], edgecolor="#333333", linewidth=0.6)
    ax_h.bar(x_pos[0], delta_nox, width=width, bottom=b1, color=poll_colors["+NOx"], edgecolor="#333333", linewidth=0.6)

    # Combined response bar (includes synergy on top).
    ax_h.bar(x_pos[1], b0, width=width, color=base_col, edgecolor="#333333", linewidth=0.6)
    ax_h.bar(x_pos[1], delta_o3, width=width, bottom=b0, color=poll_colors["+O3"], edgecolor="#333333", linewidth=0.6)
    ax_h.bar(x_pos[1], delta_nox, width=width, bottom=b1, color=poll_colors["+NOx"], edgecolor="#333333", linewidth=0.6)
    ax_h.bar(
        x_pos[1],
        delta_syn_mean,
        width=width,
        bottom=b2,
        color=poll_colors["Synergy"],
        edgecolor="#333333",
        linewidth=0.6,
        hatch="//",
    )
    total_resp = b2 + delta_syn_mean
    synergy_pct = delta_syn_mean / total_resp * 100.0 if total_resp > 0 else float("nan")

    ax_h.set_title("Contribution", pad=3, fontsize=8.2)
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels(["Input", "Response"], rotation=45, ha="right", rotation_mode="anchor")
    ax_h.tick_params(axis="x", pad=2.0)
    ax_h.set_ylabel(r"SOA ($\mu\mathrm{g}\,\mathrm{m}^{-3}$)")
    ax_h.tick_params(axis="x", labelsize=8.4)
    ax_h.tick_params(axis="y", labelsize=8.4, length=2.8, width=0.8)
    ax_h.grid(False)
    ax_h.legend(
        handles=[
            Patch(facecolor=base_col, edgecolor="#333333", label="Base"),
            Patch(facecolor=poll_colors["+O3"], edgecolor="#333333", label=r"$\Delta O_3$"),
            Patch(facecolor=poll_colors["+NOx"], edgecolor="#333333", label=r"$\Delta NO_x$"),
            Patch(facecolor=poll_colors["Synergy"], edgecolor="#333333", hatch="//", label=r"$\Delta_{syn}$"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.20, 1.0),
        frameon=False,
        fontsize=7.2,
        handlelength=1.2,
        borderpad=0.2,
        labelspacing=0.3,
    )
    y_max = float(np.nanmax([b2 + delta_syn_mean, b2, base_mean_win])) if np.isfinite(b2) else 1.0
    ax_h.set_ylim(0.0, max(1.0, y_max * 1.12))

    for ax in (ax_a, ax_b, ax_pen, ax_c, ax_e, ax_f, ax_g, ax_h):
        ax.tick_params(axis="both", which="major", length=3.2, width=0.9)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.10)

    out_png = FIGURE_DIR / "Fig5_warming_risks.png"
    out_pdf = FIGURE_DIR / "Fig5_warming_risks.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("[OK] NEW FIG 5 saved:", out_png.name, "and", out_pdf.name)


if __name__ == "__main__":
    main()
