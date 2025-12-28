from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm, zscore
from statsmodels.nonparametric.smoothers_lowess import lowess

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT  # noqa: E402

@dataclass(frozen=True)
class FitResult:
    place: str
    mean_params: np.ndarray
    mean_cov: np.ndarray
    var_params: np.ndarray
    var_cov: np.ndarray
    r2_mean: float
    r2_var: float
    hourly_df: pd.DataFrame
    hourmin_mean_df: pd.DataFrame
    hourmin_var_df: pd.DataFrame


def _display_name(var_col: str) -> str:
    if var_col == "Isoprene":
        return "BVOCs (Isoprene)"
    if var_col in ("正辛烷浓度", "n-Octane"):
        return "AVOCs (n-Octane)"
    if var_col in ("正十三烷浓度", "n-Tridecane"):
        return "AVOCs (n-Tridecane)"
    if var_col == "1,1-Dichloroethylene":
        return "AVOCs (1,1-Dichloroethylene)"
    return var_col


def _smooth_1d(values: np.ndarray, window: int = 11) -> np.ndarray:
    """
    Smooth a 1D curve for publication-ready rendering.
    """
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        return values
    try:
        from scipy.signal import savgol_filter  # type: ignore

        # Savitzky-Golay keeps curve shape while reducing jaggedness.
        return savgol_filter(values, window_length=window, polyorder=3, mode="interp")
    except Exception:
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(values, kernel, mode="same")


def _register_helvetica() -> None:
    """
    Register Helvetica (or Helvetica-compatible) fonts if local font files exist.

    This avoids relying on OS-wide font installation and keeps outputs reproducible.
    """
    candidates = [
        BUNDLE_ROOT.parent / "HelveticaNeueLTPro-Roman.otf",
        BUNDLE_ROOT.parent / "HelveticaNeueLTPro-Bd.otf",
        BUNDLE_ROOT.parent / "HelveticaNeueLTPro-It.otf",
        BUNDLE_ROOT.parent / "HelveticaNeueLTPro-BdIt.otf",
        BUNDLE_ROOT.parent / "Helvetica.ttf",
        BUNDLE_ROOT.parent / "helvetica.ttf",
    ]
    for font_file in candidates:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    # Helvetica-like fallback with better symbol coverage for Greek letters and degree-C sign.
    texgyre_dir = BUNDLE_ROOT / "paper" / "workflow" / "lib" / "fonts" / "texgyreheros"
    for name in (
        "texgyreheros-regular.otf",
        "texgyreheros-bold.otf",
        "texgyreheros-italic.otf",
        "texgyreheros-bolditalic.otf",
    ):
        font_file = texgyre_dir / name
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))


def _set_style() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _register_helvetica()
    primary_font = "Helvetica Neue LT Pro"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Use Helvetica with local fallbacks for symbol coverage.
            "font.sans-serif": [primary_font, "Helvetica", "TeX Gyre Heros", "Arial", "DejaVu Sans"],
            "font.weight": "bold",
            "mathtext.fontset": "custom",
            "mathtext.rm": primary_font,
            "mathtext.it": f"{primary_font}:italic",
            "mathtext.bf": f"{primary_font}:bold",
            "mathtext.default": "bf",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#333333",
            "axes.unicode_minus": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _set_xlabel_celsius(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$T$ ($^\circ$C)", fontweight="bold")


def _apply_bold_ticks(ax: plt.Axes) -> None:
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def _load_site_csv(place: str) -> pd.DataFrame:
    path_map = {
        "CM": BUNDLE_ROOT / "groupedcmS.csv",
        "JH": BUNDLE_ROOT / "groupedjhS.csv",
    }
    path = path_map.get(place)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Grouped CSV for place={place} not found.")
    df = pd.read_csv(path)
    df["place"] = place
    rename_map = {
        "地面层温度": "Temperature",
        "异戊二烯浓度": "Isoprene",
        "1,1-二氯乙烯浓度": "1,1-Dichloroethylene",
    }
    df = df.rename(columns=rename_map)
    required = ["Temperature", "Hour_Min", "Hour", "Isoprene", "1,1-Dichloroethylene"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for place={place}: {missing}")
    return df


def _hour_min_to_float(hour_min: str) -> float:
    # References: standard time-of-day conversion.
    # Equation: hour_float = HH + MM/60.
    # Parameters: HH hour (0-23), MM minute (0-59).
    parts = str(hour_min).split(":")
    if len(parts) < 2:
        return float("nan")
    hour = float(parts[0])
    minute = float(parts[1])
    return hour + minute / 60.0


def _remove_outliers_iqr(df: pd.DataFrame, columns: Tuple[str, ...], iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    IQR filter used in VOC_temperature_influence_mode.png.
    """
    out = df.copy()
    for col in columns:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - iqr_factor * iqr
        hi = q3 + iqr_factor * iqr
        out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out


def _compute_kl_divergence_legacy(series: pd.Series) -> float:
    """
    Legacy KLD implementation migrated from VOC_temperature_influence_mode.png.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.size <= 1:
        return float("nan")
    empirical_kde = gaussian_kde(s.to_numpy(dtype=float))
    xs = np.linspace(float(s.min()), float(s.max()), 1000)
    empirical_pdf = empirical_kde(xs)
    empirical_pdf = empirical_pdf / np.sum(empirical_pdf)

    mean = float(s.mean())
    std = float(s.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return float("nan")
    normal_pdf = norm.pdf(xs, mean, std)
    normal_pdf = normal_pdf / np.sum(normal_pdf)
    from scipy.stats import entropy  # local import to keep dependencies aligned

    return float(entropy(empirical_pdf, normal_pdf))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if y.size < 3:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _remove_outliers_z(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    z = zscore(vals.to_numpy(dtype=float), nan_policy="omit")
    out = vals.copy()
    out.loc[np.abs(z) >= z_thresh] = np.nan
    return out


def mean_relation(temp: np.ndarray, q0: float, a: float, v0: float) -> np.ndarray:
    # References: Gardiner (1985) Handbook of Stochastic Methods; moment closure used in isoprene_temperature_response.png.
    # Equation: mu(T) = Q0 + (a*T^2)/2 + v0*T.
    # Parameters: Q0 intercept (ug m^-3), a curvature (ug m^-3 degC^-2), v0 linear sensitivity (ug m^-3 degC^-1).
    return q0 + 0.5 * a * temp * temp + v0 * temp


def var_relation(temp: np.ndarray, k_val: float, sigma0: float) -> np.ndarray:
    # References: Gardiner (1985) Handbook of Stochastic Methods; moment equation used in isoprene_temperature_response.png.
    # Equation: sigma^2(T) = (k^2*T^3)/3 + k*sigma0*T^2 + sigma0^2*T.
    # Parameters: k diffusion sensitivity (ug m^-3 degC^-1), sigma0 diffusion intercept (ug m^-3).
    return (k_val * k_val * temp * temp * temp) / 3.0 + k_val * sigma0 * temp * temp + sigma0 * sigma0 * temp


def _hourly_moments_from_hourmin(data_hourmin: pd.DataFrame, var_col: str) -> pd.DataFrame:
    """
    Replicates the hourly "normal fit" summary in BVOCs的拟合效果.py:
    compute mean and standard deviation within each hour using minute-of-day aggregated values.
    """
    rows = []
    for hour, block in data_hourmin.groupby("Hour"):
        vals = pd.to_numeric(block[var_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size < 5:
            continue
        mean, std = [float(x) for x in norm.fit(vals)]
        temp_h = float(pd.to_numeric(block["Temperature"], errors="coerce").mean())
        rows.append({"Hour": int(hour), "Temperature": temp_h, "Mean": mean, "Std": std, "Var": std * std})
    out = pd.DataFrame(rows).sort_values("Temperature").reset_index(drop=True)
    return out


def _fit_site(place_df: pd.DataFrame, var_col: str) -> FitResult:
    """
    Build all inputs and fits needed for Panels A/B/D for one site,
    following isoprene_temperature_response.png logic.
    """
    data_hourmin = place_df.groupby("Hour_Min").mean(numeric_only=True).reset_index()
    data_hourmin["Hour_Float"] = data_hourmin["Hour_Min"].apply(_hour_min_to_float)

    data_hourmin_var = place_df.groupby("Hour_Min").var(numeric_only=True).reset_index()
    data_hourmin_var["Hour_Float"] = data_hourmin_var["Hour_Min"].apply(_hour_min_to_float)
    data_hourmin_var[var_col] = pd.to_numeric(data_hourmin_var[var_col], errors="coerce") / 60.0

    hourly = _hourly_moments_from_hourmin(data_hourmin, var_col=var_col)
    if hourly.empty:
        raise RuntimeError(f"Hourly moments empty for place={place_df['place'].iloc[0]}.")

    temp = hourly["Temperature"].to_numpy(dtype=float)
    mean_vals = hourly["Mean"].to_numpy(dtype=float)
    var_vals = hourly["Var"].to_numpy(dtype=float)

    # Outlier screening for the fitted points (same Z threshold used in the legacy script).
    mean_keep = np.isfinite(_remove_outliers_z(pd.Series(mean_vals), z_thresh=3.0).to_numpy())
    var_keep = np.isfinite(_remove_outliers_z(pd.Series(var_vals), z_thresh=3.0).to_numpy())

    temp_mean_fit = temp[mean_keep]
    mean_fit = mean_vals[mean_keep]
    temp_var_fit = temp[var_keep]
    var_fit = var_vals[var_keep]

    mean_p0 = [float(np.nanmean(mean_fit)), 0.01, 0.0]
    mean_params, mean_cov = curve_fit(mean_relation, temp_mean_fit, mean_fit, p0=mean_p0, maxfev=20000)
    var_p0 = [0.006, -0.10]
    var_params, var_cov = curve_fit(
        var_relation,
        temp_var_fit,
        var_fit,
        p0=var_p0,
        bounds=([0.0, -np.inf], [np.inf, np.inf]),
        maxfev=20000,
    )

    r2_mean = _r2(mean_fit, mean_relation(temp_mean_fit, *mean_params))
    r2_var = _r2(var_fit, var_relation(temp_var_fit, *var_params))

    return FitResult(
        place=str(place_df["place"].iloc[0]),
        mean_params=mean_params,
        mean_cov=mean_cov,
        var_params=var_params,
        var_cov=var_cov,
        r2_mean=r2_mean,
        r2_var=r2_var,
        hourly_df=hourly,
        hourmin_mean_df=data_hourmin,
        hourmin_var_df=data_hourmin_var,
    )


def _conf_int_95(params: np.ndarray, cov: np.ndarray, index: int, *, lower_clip: float | None = None) -> Tuple[float, float]:
    # References: asymptotic normal approximation for nonlinear least squares.
    # Equation: CI = theta_i +/- 1.96*sqrt(cov_ii).
    # Parameters: params fitted parameters; cov covariance matrix from curve_fit.
    se = float(np.sqrt(max(float(cov[index, index]), 0.0)))
    lo = float(params[index] - 1.96 * se)
    hi = float(params[index] + 1.96 * se)
    if lower_clip is not None:
        lo = max(float(lower_clip), lo)
    return lo, hi


def _plot_mean_panel(ax: plt.Axes, fit: FitResult, var_col: str) -> None:
    df_hm = fit.hourmin_mean_df
    # Cloud points use hour-min aggregated values (Temperature vs Isoprene), with Z-screening.
    cloud = df_hm[["Temperature", var_col]].copy()
    cloud[var_col] = _remove_outliers_z(cloud[var_col], z_thresh=3.0)
    cloud = cloud.dropna()
    ax.scatter(cloud["Temperature"], cloud[var_col], color="gray", alpha=0.10, s=28, marker="o", linewidths=0)

    hourly = fit.hourly_df
    mean_pts = pd.Series(hourly["Mean"])
    keep = mean_pts.notna() & _remove_outliers_z(mean_pts, z_thresh=3.0).notna()
    hourly_keep = hourly.loc[keep]
    ax.scatter(
        hourly_keep["Temperature"],
        hourly_keep["Mean"],
        color="#D8383A",
        alpha=0.85,
        marker="x",
        s=90,
        linewidths=2.6,
        zorder=4,
    )

    t_plot = np.linspace(float(hourly["Temperature"].min()), float(hourly["Temperature"].max()), 200)
    y_plot = mean_relation(t_plot, *fit.mean_params)
    ax.plot(t_plot, y_plot, color="#F27970", lw=3.2, zorder=5)

    q0, a, v0 = [float(x) for x in fit.mean_params]
    ax.text(
        0.05,
        0.93,
        rf"{fit.place}"
        "\n"
        rf"$Q_0$={q0:.3f}, $a$={a:.3f}, $v_0$={v0:.2f}"
        "\n"
        rf"$R^2$={fit.r2_mean:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    _set_xlabel_celsius(ax)
    ax.set_ylabel(r"$\mu(T)$ ($\mu$g m$^{-3}$)")
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.8)
    _apply_bold_ticks(ax)


def _plot_var_panel(ax: plt.Axes, fit: FitResult, var_col: str) -> None:
    # NOTE: The legacy isoprene_temperature_response plot uses mean Temperature (from hour-min mean)
    # against the variance estimate (from hour-min variance). Using Temperature variance would be incorrect.
    df_mean = fit.hourmin_mean_df[["Hour_Min", "Temperature"]].copy()
    df_var = fit.hourmin_var_df[["Hour_Min", var_col]].copy()
    cloud = df_mean.merge(df_var, on="Hour_Min", how="inner")
    cloud["Temperature"] = pd.to_numeric(cloud["Temperature"], errors="coerce")
    cloud[var_col] = _remove_outliers_z(cloud[var_col], z_thresh=3.0)
    cloud = cloud.dropna()
    ax.scatter(cloud["Temperature"], cloud[var_col], color="gray", alpha=0.10, s=28, marker="o", linewidths=0)

    hourly = fit.hourly_df
    var_pts = pd.Series(hourly["Var"])
    keep = var_pts.notna() & _remove_outliers_z(var_pts, z_thresh=3.0).notna()
    hourly_keep = hourly.loc[keep]
    ax.scatter(
        hourly_keep["Temperature"],
        hourly_keep["Var"],
        color="#14517C",
        alpha=0.85,
        marker="x",
        s=90,
        linewidths=2.6,
        zorder=4,
    )

    t_plot = np.linspace(float(hourly["Temperature"].min()), float(hourly["Temperature"].max()), 200)
    y_plot = var_relation(t_plot, *fit.var_params)
    ax.plot(t_plot, y_plot, color="#05B9E2", lw=3.2, zorder=5)

    k_val, sigma0 = [float(x) for x in fit.var_params]
    k_lo, k_hi = _conf_int_95(fit.var_params, fit.var_cov, index=0, lower_clip=0.0)
    ax.text(
        0.05,
        0.93,
        rf"{fit.place}"
        "\n"
        rf"$k$={k_val:.3f} [{k_lo:.3f},{k_hi:.3f}]"
        "\n"
        rf"$R^2$={fit.r2_var:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    _set_xlabel_celsius(ax)
    ax.set_ylabel(r"$\sigma^2(T)$ ($\mu$g$^2$ m$^{-6}$)")
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.8)
    _apply_bold_ticks(ax)


def _plot_mean_panel_combined(ax: plt.Axes, fits: Tuple[FitResult, ...], var_col: str) -> None:
    """
    Combined mean panel with site-specific colors and parameter labels.
    """
    color_map = {
        "JH": "#B02020",
        "CM": "#0070A3",
    }

    fits_seq = list(fits)

    for fit in fits_seq:
        df_hm = fit.hourmin_mean_df
        cloud = df_hm[["Temperature", var_col]].copy()
        cloud[var_col] = _remove_outliers_z(cloud[var_col], z_thresh=3.0)
        cloud = cloud.dropna()
        place = fit.place
        cloud_color = color_map.get(place, "gray")
        if cloud.shape[0] > 0:
            ax.scatter(
                cloud["Temperature"],
                cloud[var_col],
                color=cloud_color,
                alpha=0.10,
                s=28,
                marker="o",
                linewidths=0,
            )

    t_min = np.inf
    t_max = -np.inf
    for fit in fits_seq:
        hourly = fit.hourly_df
        if hourly.empty:
            continue
        t_min = min(t_min, float(hourly["Temperature"].min()))
        t_max = max(t_max, float(hourly["Temperature"].max()))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min >= t_max:
        return
    t_plot = np.linspace(t_min, t_max, 200)

    handles = []
    for idx, fit in enumerate(fits_seq):
        hourly = fit.hourly_df
        mean_pts = pd.Series(hourly["Mean"])
        keep = mean_pts.notna() & _remove_outliers_z(mean_pts, z_thresh=3.0).notna()
        hourly_keep = hourly.loc[keep]

        place = fit.place
        line_color = color_map.get(place, "#333333")

        if hourly_keep.shape[0] > 0:
            ax.scatter(
                hourly_keep["Temperature"],
                hourly_keep["Mean"],
                color=line_color,
                alpha=0.85,
                marker="x",
                s=90,
                linewidths=2.6,
                zorder=4,
            )

        y_plot = mean_relation(t_plot, *fit.mean_params)
        ax.plot(t_plot, y_plot, color=line_color, lw=3.2, zorder=5)

        q0, a, v0 = [float(x) for x in fit.mean_params]
        y_text = 0.97 - idx * 0.16
        ax.text(
            0.03,
            y_text,
            rf"{place}: $Q_0$={q0:.3f}, $a$={a:.3f}, $v_0$={v0:.2f}, $R^2$={fit.r2_mean:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            fontweight="bold",
            color=line_color,
        )

        handles.append(plt.Line2D([0], [0], color=line_color, lw=3.0, label=place))

    if handles:
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=11)

    _set_xlabel_celsius(ax)
    ax.set_ylabel(r"$\mu(T)$ ($\mu$g m$^{-3}$)")
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.8)
    _apply_bold_ticks(ax)


def _plot_var_panel_combined(ax: plt.Axes, fits: Tuple[FitResult, ...], var_col: str) -> None:
    """
    Combined variance panel with site-specific colors and parameter labels.
    """
    color_map = {
        "JH": "#B02020",
        "CM": "#0070A3",
    }

    fits_seq = list(fits)

    for fit in fits_seq:
        df_mean = fit.hourmin_mean_df[["Hour_Min", "Temperature"]].copy()
        df_var = fit.hourmin_var_df[["Hour_Min", var_col]].copy()
        cloud = df_mean.merge(df_var, on="Hour_Min", how="inner")
        cloud["Temperature"] = pd.to_numeric(cloud["Temperature"], errors="coerce")
        cloud[var_col] = _remove_outliers_z(cloud[var_col], z_thresh=3.0)
        cloud = cloud.dropna()
        place = fit.place
        cloud_color = color_map.get(place, "gray")
        if cloud.shape[0] > 0:
            ax.scatter(
                cloud["Temperature"],
                cloud[var_col],
                color=cloud_color,
                alpha=0.10,
                s=28,
                marker="o",
                linewidths=0,
            )

    t_min = np.inf
    t_max = -np.inf
    for fit in fits_seq:
        hourly = fit.hourly_df
        if hourly.empty:
            continue
        t_min = min(t_min, float(hourly["Temperature"].min()))
        t_max = max(t_max, float(hourly["Temperature"].max()))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min >= t_max:
        return
    t_plot = np.linspace(t_min, t_max, 200)

    handles = []
    for idx, fit in enumerate(fits_seq):
        hourly = fit.hourly_df
        var_pts = pd.Series(hourly["Var"])
        keep = var_pts.notna() & _remove_outliers_z(var_pts, z_thresh=3.0).notna()
        hourly_keep = hourly.loc[keep]

        place = fit.place
        line_color = color_map.get(place, "#333333")

        if hourly_keep.shape[0] > 0:
            ax.scatter(
                hourly_keep["Temperature"],
                hourly_keep["Var"],
                color=line_color,
                alpha=0.85,
                marker="x",
                s=90,
                linewidths=2.6,
                zorder=4,
            )

        y_plot = var_relation(t_plot, *fit.var_params)
        ax.plot(t_plot, y_plot, color=line_color, lw=3.2, zorder=5)

        k_val, sigma0 = [float(x) for x in fit.var_params]
        y_text = 0.97 - idx * 0.16
        ax.text(
            0.03,
            y_text,
            rf"{place}: $k$={k_val:.3f}, $\sigma_0$={sigma0:.3f}, $R^2$={fit.r2_var:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            fontweight="bold",
            color=line_color,
        )

        handles.append(plt.Line2D([0], [0], color=line_color, lw=3.0, label=place))

    if handles:
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=11)

    _set_xlabel_celsius(ax)
    ax.set_ylabel(r"$\sigma^2(T)$ ($\mu$g$^2$ m$^{-6}$)")
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.8)
    _apply_bold_ticks(ax)


def _crps_normal_analytical(mu_arr: np.ndarray, sigma_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    # References: Gneiting and Raftery (2007, JASA); Hersbach (2000, Weather and Forecasting).
    # Equation: CRPS(N(mu,sigma^2), y) = sigma * [ w*(2*Phi(w)-1) + 2*phi(w) - 1/sqrt(pi) ], w=(y-mu)/sigma.
    # Parameters: mu_arr predictive means; sigma_arr predictive standard deviations; y_arr observations.
    mu_arr = np.asarray(mu_arr, dtype=float)
    sigma_arr = np.asarray(sigma_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    sigma_arr = np.maximum(sigma_arr, 1e-12)
    w = (y_arr - mu_arr) / sigma_arr
    return sigma_arr * (w * (2.0 * norm.cdf(w) - 1.0) + 2.0 * norm.pdf(w) - 1.0 / np.sqrt(np.pi))


def _crps_skill_score_legacy(T: np.ndarray, mean_params: np.ndarray, var_params: np.ndarray, real_data: np.ndarray, hours: np.ndarray) -> Tuple[float, float]:
    """
    References: same as _crps_normal_analytical, using CRPS-based skill score.
    Equation: CRPSS = 1 - CRPS_model / CRPS_ref.
    Parameters: T Hour_Min temperatures; mean_params and var_params from moment fits;
    real_data observed concentrations; hours Hour identifiers for equal-weight averaging.
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(real_data, dtype=float).ravel()
    if T.shape[0] != y.shape[0]:
        raise ValueError("T and real_data must have the same length.")

    mean_pred = mean_relation(T, *mean_params)
    var_pred = var_relation(T, *var_params)
    var_pred = np.maximum(var_pred, 1e-12)
    std_pred = np.sqrt(var_pred)
    crps_model_values = _crps_normal_analytical(mean_pred, std_pred, y)

    mu_ref = float(np.mean(y))
    sigma_ref = float(np.std(y))
    if sigma_ref <= 0:
        return float(np.nanmean(crps_model_values)), float("nan")
    crps_ref_values = _crps_normal_analytical(mu_ref, sigma_ref, y)

    hours_arr = np.asarray(hours)
    unique_hours = np.unique(hours_arr)
    per_hour_model = []
    per_hour_ref = []
    for h in unique_hours:
        mask_h = hours_arr == h
        if not np.any(mask_h):
            continue
        m_h = float(np.nanmean(crps_model_values[mask_h]))
        r_h = float(np.nanmean(crps_ref_values[mask_h]))
        if np.isfinite(m_h) and np.isfinite(r_h) and r_h > 0:
            per_hour_model.append(m_h)
            per_hour_ref.append(r_h)

    if len(per_hour_model) == 0:
        crps_model = float(np.nanmean(crps_model_values))
        crps_ref = float(np.nanmean(crps_ref_values))
    else:
        crps_model = float(np.mean(per_hour_model))
        crps_ref = float(np.mean(per_hour_ref))

    if crps_ref <= 0:
        return crps_model, float("nan")
    crpss = 1.0 - crps_model / crps_ref
    return crps_model, crpss


def _plot_temperature_influence_mode(ax: plt.Axes, place_df: pd.DataFrame, var_col: str, title: str, color_points: str) -> None:
    """
    Matplotlib re-implementation of VOC_temperature_influence_mode.png for a single variable and site.
    """
    # Legacy pipeline (VOC_temperature_influence_mode.png): Hour_Min mean + IQR outlier removal.
    datp = place_df.groupby("Hour_Min").mean(numeric_only=True).reset_index()
    dat = pd.DataFrame({"x": datp["Temperature"].copy(), "y": datp[var_col].copy()})
    dat["x"] = pd.to_numeric(dat["x"], errors="coerce")
    dat["y"] = pd.to_numeric(dat["y"], errors="coerce")
    dat = dat.dropna()
    dat = _remove_outliers_iqr(dat, columns=("x", "y"))
    if dat.empty:
        return

    n_breaks = 7
    breaks = np.linspace(float(dat["x"].min()), float(dat["x"].max()), n_breaks)
    dat["section"] = pd.cut(dat["x"], breaks)

    ax.scatter(dat["x"], dat["y"], color=color_points, alpha=0.22, s=18, linewidths=0, zorder=1)
    smoothed = lowess(dat["y"], dat["x"], frac=0.55, return_sorted=True)
    ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=color_points,
        lw=4.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    # Match the original VOC_temperature_influence_mode.png scaling:
    # Isoprene and 1,1-Dichloroethylene use rat=0.7, n-Tridecane uses rat=0.3.
    if var_col in ("正十三烷浓度", "n-Tridecane"):
        rat = 0.3
    else:
        rat = 0.7
    kld_per_section: list[float] = []
    for _, group in dat.groupby("section", observed=False):
        if group.shape[0] < 10:
            continue
        g = pd.to_numeric(group["y"], errors="coerce").dropna().to_numpy(dtype=float)
        if g.size <= 1:
            continue
        xs = np.linspace(float(np.min(g)), float(np.max(g)), 1000)
        kde_pdf = gaussian_kde(g)(xs)
        mu = float(np.mean(g))
        sd = float(np.std(g, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        nor_pdf = norm.pdf(xs, mu, sd)
        x_anchor = float(np.nanmax(pd.to_numeric(group["x"], errors="coerce")))
        ax.plot(
            x_anchor - kde_pdf * rat,
            xs,
            color="#D8383A",
            lw=4.0,
            ls="-.",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
        ax.plot(
            x_anchor - nor_pdf * rat,
            xs,
            color="#05B9E2",
            lw=4.0,
            ls="-.",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )

        kld_val = _compute_kl_divergence_legacy(pd.Series(g))
        if np.isfinite(kld_val):
            kld_per_section.append(float(kld_val))

    for b in breaks:
        ax.axvline(float(b), color="#333333", ls="--", lw=1.1, alpha=0.75)

    _set_xlabel_celsius(ax)
    ax.set_ylabel(rf"{_display_name(var_col)} ($\mu$g m$^{{-3}}$)")
    y_label_obj = ax.yaxis.get_label()
    try:
        # One size smaller than default for BVOCs, two sizes smaller for AVOCs.
        base_size = 13
        if _display_name(var_col).startswith("AVOCs"):
            base_size = 12
        # Additional downscale requested for paper typography balance.
        base_size = max(8, base_size - 1)
        y_label_obj.set_fontsize(base_size)
    except Exception:
        pass
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.8)
    _apply_bold_ticks(ax)

    x_min = float(dat["x"].min())
    x_max = float(dat["x"].max())
    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        pad_x = 0.01 * (x_max - x_min)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)

    y_min = float(dat["y"].min())
    y_max = float(dat["y"].max())
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        pad_y = 0.02 * (y_max - y_min)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    if kld_per_section:
        kld_arr = np.asarray(kld_per_section, dtype=float)
        kld_arr = kld_arr[np.isfinite(kld_arr)]
        if kld_arr.size:
            kld_mean = float(np.mean(kld_arr))
            kld_min = float(np.min(kld_arr))
            kld_max = float(np.max(kld_arr))
            text_label = rf"$D_{{KL}}$={kld_mean:.2f}" + "\n" + rf"[{kld_min:.2f},{kld_max:.2f}]"
            ax.text(
                0.98,
                0.05,
                text_label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                color="#333333",
                linespacing=1.1,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.3),
            )

    handles = [
        plt.Line2D([0], [0], color="#D8383A", lw=4.0, ls="-.", label="Empirical"),
        plt.Line2D([0], [0], color="#05B9E2", lw=4.0, ls="-.", label="Normal"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=22,
        prop={"weight": "bold"},
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.0,
        labelspacing=0.2,
    )


def _select_warming_window(hourmin_mean_df: pd.DataFrame, minutes: int = 360) -> pd.DataFrame:
    """
    Select a 6-hour window on the diurnal profile with the largest temperature increase.
    """
    df = hourmin_mean_df[["Hour_Float", "Temperature"]].dropna().sort_values("Hour_Float").reset_index(drop=True)
    temps = df["Temperature"].to_numpy(dtype=float)
    if df.shape[0] <= minutes:
        return df
    delta = temps[minutes - 1 :] - temps[: -(minutes - 1)]
    start = int(np.nanargmax(delta))
    end = start + minutes
    return df.iloc[start:end].reset_index(drop=True)


def _plot_monte_carlo_panel(ax: plt.Axes, fit: FitResult, var_col: str) -> None:
    df = fit.hourmin_mean_df.copy()
    df["Hour_Float"] = df["Hour_Min"].apply(_hour_min_to_float)
    df = df.sort_values("Hour_Float")

    window = _select_warming_window(df[["Hour_Float", "Temperature"]].assign(Temperature=df["Temperature"]), minutes=360)
    x = window["Hour_Float"].to_numpy(dtype=float)
    temp_series = window["Temperature"].to_numpy(dtype=float)

    mu = mean_relation(temp_series, *fit.mean_params)
    var = var_relation(temp_series, *fit.var_params)
    sd = np.sqrt(np.maximum(var, 1e-12))

    # References: Gardiner (1985) Handbook of Stochastic Methods; sampling model in isoprene_temperature_response.png.
    # Equation: X_i ~ Normal(mu(T_i), sigma^2(T_i)); i indexes minute-of-day within a short window.
    # Parameters: mu(T) and sigma^2(T) from fitted moment relations; n_paths Monte Carlo ensemble size.
    rng = np.random.default_rng(42)
    n_paths = 400
    samples = rng.normal(loc=mu, scale=sd, size=(n_paths, x.size))
    samples = np.clip(samples, a_min=0.0, a_max=None)
    q10, q50, q90 = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

    # Overlay a few sample paths (matching the visual cue in the reference figure).
    ns = 5
    for j in range(ns):
        path = rng.normal(loc=mu, scale=sd, size=x.size)
        path = np.clip(path, a_min=0.0, a_max=None)
        ax.plot(x, path, color="#7f7f7f", alpha=0.55, lw=1.8, zorder=2)

    # Observations in the same window (hour-min values).
    obs = fit.hourmin_mean_df.copy()
    obs["Hour_Float"] = obs["Hour_Min"].apply(_hour_min_to_float)
    obs = obs[(obs["Hour_Float"] >= float(x.min())) & (obs["Hour_Float"] <= float(x.max()))]
    obs_y = pd.to_numeric(obs[var_col], errors="coerce")
    ax.scatter(
        obs["Hour_Float"],
        obs_y,
        color="#F1D77E",
        alpha=0.98,
        s=40,
        linewidths=0.25,
        edgecolors="#6B5A12",
        label="Observed",
        zorder=30,
    )

    # Skill metrics: compute on the full Hour_Min series to match isoprene_temperature_response.png.
    df_full = fit.hourmin_mean_df.copy()
    for c in ("Temperature", var_col, "Hour"):
        df_full[c] = pd.to_numeric(df_full[c], errors="coerce")
    df_full = df_full.dropna(subset=["Temperature", var_col, "Hour"]).copy()
    df_full = df_full[np.abs(zscore(df_full["Temperature"])) < 3].copy()
    df_full = df_full[np.abs(zscore(df_full[var_col])) < 3].copy()
    df_full = df_full.reset_index(drop=True)

    crps_value = float("nan")
    crpss = float("nan")
    bias_abs = float("nan")
    r_spread = float("nan")
    c90 = float("nan")
    if df_full.shape[0] >= 10:
        crps_value, crpss = _crps_skill_score_legacy(
            df_full["Temperature"].to_numpy(dtype=float),
            fit.mean_params,
            fit.var_params,
            df_full[var_col].to_numpy(dtype=float),
            df_full["Hour"].to_numpy(),
        )

        mu_full = mean_relation(df_full["Temperature"].to_numpy(dtype=float), *fit.mean_params)
        var_full = var_relation(df_full["Temperature"].to_numpy(dtype=float), *fit.var_params)
        sigma_full = np.sqrt(np.maximum(var_full, 1e-12))
        y_full = df_full[var_col].to_numpy(dtype=float)
        bias_abs = float(np.mean(np.abs(mu_full - y_full)))
        lower_90 = mu_full - 1.645 * sigma_full
        upper_90 = mu_full + 1.645 * sigma_full
        c90 = float(np.mean((y_full >= lower_90) & (y_full <= upper_90)))

        spread_ratios = []
        for h_val in sorted(df_full["Hour"].unique()):
            block = df_full[df_full["Hour"] == h_val]
            y_h = pd.to_numeric(block[var_col], errors="coerce").dropna().to_numpy(dtype=float)
            if y_h.size < 2:
                continue
            obs_sd_h = float(np.std(y_h, ddof=1))
            if obs_sd_h <= 0:
                continue
            t_h = pd.to_numeric(block["Temperature"], errors="coerce").dropna().to_numpy(dtype=float)
            if t_h.size < 1:
                continue
            t_mean_h = float(np.mean(t_h))
            var_h = float(var_relation(np.asarray([t_mean_h]), *fit.var_params)[0])
            sigma_h = float(np.sqrt(max(var_h, 1e-12)))
            spread_ratios.append(sigma_h / obs_sd_h)
        if spread_ratios:
            r_spread = float(np.median(spread_ratios))

    ax.text(
        0.98,
        0.06,
        "CRPS = "
        + f"{crps_value:.3f}, CRPSS = {crpss:.3f}"
        + "\n"
        + r"$|\mathrm{bias}|$"
        + f" = {bias_abs:.2f}, "
        + r"$R_{\mathrm{spread}}$"
        + f" = {r_spread:.2f}"
        + "\n"
        + r"$C_{90}$"
        + f" = {c90:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.25),
        zorder=2500,
    )

    # Use robust limits to avoid rare spikes dominating the y-scale.
    y_all = np.concatenate([obs_y.dropna().to_numpy(dtype=float), q90])
    if y_all.size:
        y_hi = float(np.nanquantile(y_all, 0.995))
        ax.set_ylim(bottom=0.0, top=max(1.0, y_hi * 1.05))

    ax.set_xlabel("Hour")
    ax.set_ylabel(rf"{_display_name(var_col)} ($\mu$g m$^{{-3}}$)")
    ax.set_xticks([6, 8, 10, 12])
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.8)

    handles = [
        plt.Line2D([0], [0], color="#7f7f7f", lw=2.2, label="Monte-Carlo Simulation"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#6B5A12",
            markerfacecolor="#F1D77E",
            markersize=7.5,
            alpha=0.95,
            linewidth=0,
            label="Data Point",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, prop={"weight": "bold"}, fontsize=13)
    _apply_bold_ticks(ax)


def _raise_all_text_to_top(fig: plt.Figure, *, text_zorder: float = 1000.0, legend_zorder: float = 2000.0) -> None:
    for ax in fig.axes:
        for txt in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + list(ax.get_xticklabels())
            + list(ax.get_yticklabels())
            + list(getattr(ax, "texts", []))
        ):
            try:
                txt.set_zorder(text_zorder)
            except Exception:
                pass

        leg = ax.get_legend()
        if leg is not None:
            try:
                leg.set_zorder(legend_zorder)
            except Exception:
                pass
            for t in leg.get_texts():
                try:
                    t.set_zorder(legend_zorder + 1.0)
                except Exception:
                    pass
            if leg.get_title() is not None:
                try:
                    leg.get_title().set_zorder(legend_zorder + 1.0)
                except Exception:
                    pass


def main() -> None:
    env = os.environ.copy()
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    os.environ.update(env)
    _set_style()

    df_cm = _load_site_csv("CM")
    df_jh = _load_site_csv("JH")

    species_main = "Isoprene"
    species_ctrl = "正十三烷浓度"

    fit_cm = _fit_site(df_cm, var_col=species_main)
    fit_jh = _fit_site(df_jh, var_col=species_main)

    fig = plt.figure(figsize=(16.5, 10.2))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.15, 1.0, 0.9],
        wspace=0.30,
        hspace=0.36,
    )

    # Row 1: distribution (JH/CM; BVOC vs AVOC control).
    ax_jh_iso = fig.add_subplot(gs[0, 0])
    ax_jh_avoc = fig.add_subplot(gs[0, 1])
    ax_cm_iso = fig.add_subplot(gs[0, 2])
    ax_cm_avoc = fig.add_subplot(gs[0, 3])
    _plot_temperature_influence_mode(
        ax_jh_iso,
        df_jh,
        var_col=species_main,
        title="",
        color_points="#96C37D",
    )
    _plot_temperature_influence_mode(
        ax_jh_avoc,
        df_jh,
        var_col=species_ctrl,
        title="",
        color_points="#C497B2",
    )
    _plot_temperature_influence_mode(
        ax_cm_iso,
        df_cm,
        var_col=species_main,
        title="",
        color_points="#4C9FDE",
    )
    _plot_temperature_influence_mode(
        ax_cm_avoc,
        df_cm,
        var_col=species_ctrl,
        title="",
        color_points="#E67E22",
    )

    # Row 2: combined fits (two sites) - mean and variance in two panels.
    ax_mean = fig.add_subplot(gs[1, 0:2])
    ax_var = fig.add_subplot(gs[1, 2:4])
    ax_mean.set_title(r"Mean relation: $\mu(T)$")
    ax_var.set_title(r"Diffusion term: $\sigma^2(T)$")
    _plot_mean_panel_combined(ax_mean, (fit_jh, fit_cm), var_col=species_main)
    _plot_var_panel_combined(ax_var, (fit_jh, fit_cm), var_col=species_main)

    # Row 3: Monte Carlo (two sites) - migrated from isoprene_temperature_response.png.
    ax_d_jh = fig.add_subplot(gs[2, 0:2])
    ax_d_cm = fig.add_subplot(gs[2, 2:4])
    ax_d_jh.set_title("Monte Carlo ensemble (6 hours): JH")
    ax_d_cm.set_title("Monte Carlo ensemble (6 hours): CM")
    _plot_monte_carlo_panel(ax_d_jh, fit_jh, var_col=species_main)
    _plot_monte_carlo_panel(ax_d_cm, fit_cm, var_col=species_main)

    for text in fig.findobj(match=plt.Text):
        text.set_fontweight("bold")
    _raise_all_text_to_top(fig)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "Fig2_temperature_amplifies_bvocs.png"
    fig.savefig(out_path)
    plt.close(fig)
    print("[OK] Fig2 written: Fig2_temperature_amplifies_bvocs.png")


if __name__ == "__main__":
    main()
