import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import CHECKPOINT_DIR, FIGURE_DIR  # noqa: E402
from paper.workflow.plot import paper_japp_surv_plots as japp_surv  # noqa: E402
from src.workflow.config import default_config  # noqa: E402
from src.workflow import core as workflow_core  # noqa: E402


TARGET_NAME = "Fig3_mechanism_validation.png"
DT_SECONDS = 10.0
HIGH_I_QUANTILE = 0.8
SURV_OBS_D1_UM = 0.25
SURV_OBS_D2_UM = 0.40
SURV_OBS_WINDOW_MIN = 60
SURV_OBS_SMOOTH_STEPS = 5
SURV_OBS_SIGNAL_Q = 0.6
SURV_OBS_MAX_LAG_MIN = 360

COLORS = {
    "JH": "#0b8043",
    "CM": "#b35900",
    "theory": "#222222",
    "accent": "#E64B35",
    "grid": "#d9d9d9",
    "spine": "#333333",
}


@dataclass(frozen=True)
class FitStats:
    slope: float
    intercept: float
    r: float
    r2_uncentered: float


@dataclass(frozen=True)
class MassClosureStats:
    fit: FitStats
    fit_hi: FitStats
    n_hi: int
    alpha_theory: float


def _register_helvetica_like_fonts() -> None:
    """
    Register Helvetica (or Helvetica-compatible) fonts from local files if present.

    This avoids relying on OS-wide font installation and keeps outputs reproducible.
    """
    candidates = [
        THIS_ROOT.parent / "HelveticaNeueLTPro-Roman.otf",
        THIS_ROOT.parent / "HelveticaNeueLTPro-Bd.otf",
        THIS_ROOT.parent / "HelveticaNeueLTPro-It.otf",
        THIS_ROOT.parent / "HelveticaNeueLTPro-BdIt.otf",
        THIS_ROOT.parent / "Helvetica.ttf",
        THIS_ROOT.parent / "helvetica.ttf",
        THIS_ROOT / "paper" / "workflow" / "lib" / "fonts" / "texgyreheros" / "texgyreheros-regular.otf",
        THIS_ROOT / "paper" / "workflow" / "lib" / "fonts" / "texgyreheros" / "texgyreheros-bold.otf",
        THIS_ROOT / "paper" / "workflow" / "lib" / "fonts" / "texgyreheros" / "texgyreheros-italic.otf",
        THIS_ROOT / "paper" / "workflow" / "lib" / "fonts" / "texgyreheros" / "texgyreheros-bolditalic.otf",
    ]
    for font_file in candidates:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))


def _set_figure_style() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _register_helvetica_like_fonts()
    # Prefer Helvetica if bundled; keep local fallbacks for symbol coverage.
    primary_font = "Helvetica Neue LT Pro"
    math_font = primary_font
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [primary_font, "Helvetica", "TeX Gyre Heros", "Arial", "DejaVu Sans"],
            "font.weight": "bold",
            "mathtext.fontset": "custom",
            "mathtext.rm": math_font,
            "mathtext.it": f"{math_font}:italic",
            "mathtext.bf": f"{math_font}:bold",
            "mathtext.default": "bf",
            "text.usetex": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "axes.edgecolor": COLORS["spine"],
            "axes.unicode_minus": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22, color=COLORS["grid"], linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(COLORS["spine"])
        spine.set_linewidth(1.0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def _select_ratio_decade_levels(
    log_ratio: np.ndarray, *, max_levels: int = 4
) -> tuple[list[float], list[int]]:
    if log_ratio.size == 0:
        return [1.0, 10.0, 100.0, 1000.0], [0, 1, 2, 3]
    q_low, q_high = np.nanquantile(log_ratio, [0.05, 0.95])
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        return [1.0, 10.0, 100.0, 1000.0], [0, 1, 2, 3]
    decade_low = int(np.floor(q_low))
    decade_high = int(np.ceil(q_high))
    decades = list(range(decade_low, decade_high + 1))
    if len(decades) > max_levels:
        idx = np.linspace(0, len(decades) - 1, max_levels)
        decades = sorted({decades[int(round(i))] for i in idx})
    levels = [10.0 ** d for d in decades]
    return levels, decades


def _median_implied_eta(survival: pd.Series, ratio: pd.Series) -> float:
    implied_eta = ((1.0 / survival) - 1.0) / ratio
    implied_eta = implied_eta.replace([np.inf, -np.inf], np.nan)
    implied_eta = implied_eta[(implied_eta > 0) & implied_eta.notna()]
    if implied_eta.empty:
        return float("nan")
    return float(implied_eta.median())


def _compute_eta_by_place(output: pd.DataFrame) -> dict[str, float]:
    if not {"S_surv", "CS_star", "G_abs"}.issubset(output.columns):
        return {}
    if "place" in output.columns and output["place"].notna().any():
        places = sorted(output["place"].dropna().unique())
    else:
        places = ["ALL"]
    eta_by_place: dict[str, float] = {}
    for place in places:
        sub = output[output["place"] == place] if place != "ALL" else output
        ratio = (sub["CS_star"] / sub["G_abs"]).replace([np.inf, -np.inf], np.nan)
        ratio = ratio[ratio > 0]
        surv = sub.loc[ratio.index, "S_surv"].replace([np.inf, -np.inf], np.nan)
        # References: Kerminen and Kulmala (2002); Kulmala et al. (2007).
        # Equation: eta = (1 / S_surv - 1) / (CS_star / G_abs).
        # Parameters: eta is dimensionless; S_surv is survival probability; ratio is CS_star over G_abs.
        eta_val = _median_implied_eta(surv, ratio)
        if np.isfinite(eta_val):
            eta_by_place[str(place)] = eta_val
    return eta_by_place


def _resample_numeric_with_interpolation(
    df: pd.DataFrame, *, rule: str, limit: int
) -> pd.DataFrame:
    resampled = df.resample(rule).mean()
    for col in resampled.columns:
        resampled[col] = resampled[col].interpolate(limit=limit, limit_direction="both")
    return resampled


def _select_nearest_number_bins(
    number_cols: list[str], target_um: tuple[float, float]
) -> tuple[str, str, np.ndarray, list[str]]:
    diam_um = np.array([workflow_core._column_to_um(col) for col in number_cols], dtype=float)
    order = np.argsort(diam_um)
    diam_um_sorted = diam_um[order]
    cols_sorted = [number_cols[idx] for idx in order]
    idx1 = int(np.argmin(np.abs(diam_um_sorted - target_um[0])))
    idx2 = int(np.argmin(np.abs(diam_um_sorted - target_um[1])))
    if idx1 == idx2:
        idx2 = min(idx1 + 1, len(cols_sorted) - 1) if idx1 < len(cols_sorted) - 1 else max(idx1 - 1, 0)
    return cols_sorted[idx1], cols_sorted[idx2], diam_um_sorted, cols_sorted


def _estimate_bin_width_nm(diam_um_sorted: np.ndarray, idx: int) -> float:
    if diam_um_sorted.size < 2:
        return float("nan")
    if idx >= diam_um_sorted.size - 1:
        width_um = diam_um_sorted[idx] - diam_um_sorted[idx - 1]
    else:
        width_um = diam_um_sorted[idx + 1] - diam_um_sorted[idx]
    return max(float(width_um), 1e-6) * 1000.0


def _compute_empirical_survival(
    output: pd.DataFrame,
    cfg,
    *,
    d1_um: float,
    d2_um: float,
    window_minutes: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not {"CS_star", "G_abs"}.issubset(output.columns):
        return pd.DataFrame(), {}
    dt_seconds = float(pd.to_timedelta(cfg.resample_rule).total_seconds())
    window_steps = max(3, int(round(window_minutes * 60.0 / dt_seconds)))
    min_steps = max(3, window_steps // 2)
    frames: list[pd.DataFrame] = []
    meta: dict[str, float] = {}
    if "place" in output.columns and output["place"].notna().any():
        places = sorted(output["place"].dropna().unique())
    else:
        places = [None]

    for place in places:
        if place is None:
            size_path = cfg.size_distribution_csv
            out_place = output
            place_label = "ALL"
        else:
            size_path = cfg.size_distribution_csv_sites.get(place)
            out_place = output[output["place"] == place]
            place_label = str(place)
        if size_path is None or not Path(size_path).exists():
            continue

        size_df = workflow_core.load_size_distribution(Path(size_path))
        _, number_cols = workflow_core._split_size_columns(size_df.columns)
        if not number_cols:
            continue
        col1, col2, diam_um_sorted, cols_sorted = _select_nearest_number_bins(
            number_cols, (d1_um, d2_um)
        )
        idx1 = cols_sorted.index(col1)
        idx2 = cols_sorted.index(col2)
        d1_used = float(diam_um_sorted[idx1])
        d2_used = float(diam_um_sorted[idx2])
        delta1_nm = _estimate_bin_width_nm(diam_um_sorted, idx1)
        delta2_nm = _estimate_bin_width_nm(diam_um_sorted, idx2)
        if not np.isfinite(delta1_nm) or not np.isfinite(delta2_nm):
            continue

        size_sub = size_df[[col1, col2]].copy()
        size_sub = _resample_numeric_with_interpolation(
            size_sub, rule=cfg.resample_rule, limit=cfg.short_gap_limit
        )
        out_sub = out_place[["CS_star", "G_abs"]].copy()
        out_sub = _resample_numeric_with_interpolation(
            out_sub, rule=cfg.resample_rule, limit=cfg.short_gap_limit
        )
        joined = size_sub.join(out_sub, how="inner").dropna()
        if joined.empty:
            continue

        n1 = pd.to_numeric(joined[col1], errors="coerce")
        n2 = pd.to_numeric(joined[col2], errors="coerce")
        if SURV_OBS_SMOOTH_STEPS > 1:
            n1 = n1.rolling(SURV_OBS_SMOOTH_STEPS, center=True, min_periods=1).mean()
            n2 = n2.rolling(SURV_OBS_SMOOTH_STEPS, center=True, min_periods=1).mean()
        dn1_dt = n1.diff() / dt_seconds
        dn2_dt = n2.diff() / dt_seconds
        g_abs = pd.to_numeric(joined["G_abs"], errors="coerce").abs().clip(lower=1e-6)
        cs_star = pd.to_numeric(joined["CS_star"], errors="coerce").clip(lower=0.0)

        # References: Kerminen and Kulmala 2002; Kulmala et al. 2007.
        # Equation: J(D,t) = dN_dt + K_loss * N + (G / delta_D) * N.
        # Parameters: J is formation rate; N is number conc in bin; K_loss uses CS_star as loss rate (s^-1);
        # G is growth rate (nm/s); delta_D is bin width (nm).
        j1 = (dn1_dt + cs_star * n1 + (g_abs / delta1_nm) * n1).clip(lower=0.0)
        j2 = (dn2_dt + cs_star * n2 + (g_abs / delta2_nm) * n2).clip(lower=0.0)

        # References: Kerminen and Kulmala 2002; Kulmala et al. 2007.
        # Equation: tau = (D2 - D1) / G_med; J2_aligned(t) = J2(t + tau).
        # Parameters: D1 and D2 are diameters (nm); G_med is median growth rate (nm/s); tau is lag (s).
        d_nm = (d2_used - d1_used) * 1000.0
        g_med = float(g_abs.median())
        tau_seconds = d_nm / max(g_med, 1e-6)
        max_shift_steps = int(round(SURV_OBS_MAX_LAG_MIN * 60.0 / dt_seconds))
        shift_steps = int(round(tau_seconds / dt_seconds))
        shift_steps = min(max(shift_steps, 0), max_shift_steps)
        if shift_steps > 0:
            j2 = j2.shift(-shift_steps)

        # References: size-space continuity survival estimate.
        # Equation: S_surv_obs = sum_window(J2_aligned) / sum_window(J1).
        # Parameters: J1 and J2 are formation rates at D1 and D2; window_minutes sets the integration window.
        roll_j1 = j1.rolling(window_steps, min_periods=min_steps).sum()
        roll_j2 = j2.rolling(window_steps, min_periods=min_steps).sum()
        s_obs = (roll_j2 / roll_j1).replace([np.inf, -np.inf], np.nan)
        s_obs = s_obs.clip(lower=0.0, upper=1.0)
        ratio = (cs_star / g_abs).replace([np.inf, -np.inf], np.nan)
        ratio_window = ratio.rolling(window_steps, min_periods=min_steps).median()
        # References: robust event filtering for noisy formation rates.
        # Equation: keep samples where sum_window(J1) exceeds the SURV_OBS_SIGNAL_Q quantile.
        # Parameters: SURV_OBS_SIGNAL_Q controls the retained high-signal fraction.
        mask_signal = roll_j1 > roll_j1.quantile(SURV_OBS_SIGNAL_Q)
        df_obs = pd.DataFrame({"S_surv_obs": s_obs, "ratio": ratio_window}).dropna()
        if not df_obs.empty:
            df_obs = df_obs[mask_signal.loc[df_obs.index]]
        if df_obs.empty:
            continue
        df_obs = df_obs[df_obs["ratio"] > 0]
        df_obs["log_ratio"] = np.log10(df_obs["ratio"])
        df_obs["place"] = place_label
        frames.append(df_obs)
        meta["d1_um"] = d1_used
        meta["d2_um"] = d2_used

    if not frames:
        return pd.DataFrame(), meta
    combined = pd.concat(frames, axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    return combined, meta


def _legend_no_background(legend: Legend | None) -> None:
    if legend is None:
        return
    legend.set_frame_on(False)
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_edgecolor("none")
    frame.set_alpha(0.0)


def _raise_all_text_to_front(fig: plt.Figure, z_text: float = 50.0, z_legend: float = 60.0) -> None:
    """
    Ensure all figure text is drawn above points/lines.

    This is used to guarantee that annotations/labels are not visually occluded by dense scatters.
    """
    for t in fig.findobj(Text):
        try:
            t.set_zorder(z_text)
        except Exception:
            continue

    # Figure-level legends (fig.legend) live in fig.legends.
    for leg in getattr(fig, "legends", []):
        try:
            leg.set_zorder(z_legend)
        except Exception:
            pass
        try:
            for t in leg.get_texts():
                t.set_zorder(z_legend + 1.0)
        except Exception:
            pass

    # Axes-level legends (ax.legend) are attached to each axes.
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        try:
            leg.set_zorder(z_legend)
        except Exception:
            pass
        try:
            for t in leg.get_texts():
                t.set_zorder(z_legend + 1.0)
        except Exception:
            pass


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    # Panel letters are disabled for the final paper figure (requested: remove all numbering).
    return
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def _compute_mass_per_particle_ug(d_soA_nm: float, d_nuc_nm: float, rho_g_cm3: float) -> float:
    """
    References
    ----------
    - Kulmala, M. et al. (2007) Atmos. Chem. Phys.

    Mathematical expression
    -----------------------
    m_p = (pi/6) * rho * (d_f^3 - d_nuc^3).

    Parameter meanings
    ------------------
    - d_f: effective final diameter (nm).
    - d_nuc: nucleation diameter (nm).
    - rho: particle density (g cm^-3).

    Returns
    -------
    float
        Particle mass in micrograms (ug) per particle.
    """
    d_f_cm = d_soA_nm * 1e-7
    d_nuc_cm = d_nuc_nm * 1e-7
    volume_cm3 = (np.pi / 6.0) * (d_f_cm**3 - d_nuc_cm**3)
    mass_g = volume_cm3 * rho_g_cm3
    return float(mass_g * 1e6)


def _alpha_theory_ug_m3_per_cm3(
    growth_metrics: pd.DataFrame | None,
    *,
    place: str,
    d_soA_nm_default: float = 200.0,
    d_nuc_nm: float = 2.0,
    rho_g_cm3: float = 1.3,
) -> float:
    """
    References
    ----------
    - Kulmala, M. et al. (2007) Atmos. Chem. Phys.

    Mathematical expression
    -----------------------
    DeltaM = alpha * I_delta, where alpha = m_p * 1e6.

    Parameter meanings
    ------------------
    - I_delta = int J_app S_surv dt (cm^-3).
    - DeltaM: mass increment (ug m^-3).
    - m_p: mass per particle (ug particle^-1).
    - 1e6: conversion from per cm^3 to per m^3.
    """
    d_target_nm = float(d_soA_nm_default)
    if growth_metrics is not None and {"place", "d_eff_nm"}.issubset(growth_metrics.columns):
        # Keep consistent with scripts\\mass_closure.py (SOA_mass_closure_deltaM_vs_I.png):
        # use the 0.75 quantile of d_eff_nm only for JH, otherwise fall back to d_soA_nm_default.
        if place == "JH":
            sub = growth_metrics[growth_metrics["place"] == place]["d_eff_nm"]
            q = sub.quantile(0.75)
            if np.isfinite(q) and q > 0:
                d_target_nm = float(q)
    m_p_ug = _compute_mass_per_particle_ug(d_target_nm, d_nuc_nm, rho_g_cm3)
    return float(m_p_ug * 1e6)


def _rolling_integral(prod: pd.Series, window_steps: int, dt_seconds: float) -> pd.Series:
    return prod.rolling(window_steps, min_periods=1).sum() * dt_seconds


def _rolling_sum(values: pd.Series, window_steps: int) -> pd.Series:
    return values.rolling(window_steps, min_periods=1).sum()


def _fit_linear(x: pd.Series, y: pd.Series) -> FitStats:
    """
    References
    ----------
    - Standard least-squares regression.

    Mathematical expression
    -----------------------
    y_hat = a x + b, with a,b from least squares; and uncentered R^2:
    R2_uc = 1 - sum((y - y_hat)^2) / sum(y^2).

    Parameter meanings
    ------------------
    - x: integrated formation proxy I_delta (cm^-3).
    - y: mass increment DeltaM (ug m^-3).
    - a: fitted slope (ug m^-3 per cm^-3).
    - b: fitted intercept (ug m^-3).
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 5:
        return FitStats(np.nan, np.nan, np.nan, np.nan)
    slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    r = float(x.corr(y))
    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot_unc = float((y**2).sum())
    r2_uc = 1.0 - ss_res / ss_tot_unc if ss_tot_unc > 0 else np.nan
    return FitStats(float(slope), float(intercept), float(r), float(r2_uc))


def _fit_high_i_with_fixed_intercept(
    x: pd.Series, y: pd.Series, *, q: float, intercept_fixed: float
) -> tuple[FitStats, int]:
    """
    References
    ----------
    - Keep consistent with the mass-closure analysis script.

    Mathematical expression
    -----------------------
    Define high-signal subset by I_delta >= Q_q(I_delta).
    Fit slope_hi on (x_hi, y_hi), but evaluate with a fixed intercept:
    y_hat_hi = slope_hi * x_hi + intercept_fixed.
    R2_uc_hi = 1 - sum((y_hi - y_hat_hi)^2) / sum(y_hi^2).

    Parameter meanings
    ------------------
    - q: high-signal quantile threshold (e.g., 0.8 keeps top 20%).
    - intercept_fixed: intercept from the full-sample regression.
    """
    if not (0.0 < q < 1.0) or len(x) < 10:
        return FitStats(np.nan, float(intercept_fixed), np.nan, np.nan), 0
    thr = float(x.quantile(q))
    mask_hi = x >= thr
    x_hi = x[mask_hi]
    y_hi = y[mask_hi]
    n_hi = int(mask_hi.sum())
    if len(x_hi) < 5 or np.isclose(float(x_hi.std()), 0.0) or np.isclose(float(y_hi.std()), 0.0):
        return FitStats(np.nan, float(intercept_fixed), np.nan, np.nan), n_hi
    slope_hi, _ = np.polyfit(x_hi.to_numpy(), y_hi.to_numpy(), 1)
    r_hi = float(x_hi.corr(y_hi))
    y_hat = slope_hi * x_hi + float(intercept_fixed)
    ss_res = float(((y_hi - y_hat) ** 2).sum())
    ss_tot_unc = float((y_hi**2).sum())
    r2_uc_hi = 1.0 - ss_res / ss_tot_unc if ss_tot_unc > 0 else np.nan
    return FitStats(float(slope_hi), float(intercept_fixed), float(r_hi), float(r2_uc_hi)), n_hi


def _downsample_xy(x: pd.Series, y: pd.Series, max_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x.to_numpy(), y.to_numpy()
    idx = np.random.default_rng(0).choice(len(x), size=max_points, replace=False)
    return x.to_numpy()[idx], y.to_numpy()[idx]


def _plot_mass_closure_panel(
    ax: plt.Axes,
    hf: pd.DataFrame,
    *,
    delta_label: str,
    window_steps: int,
    alpha_theory_by_place: Dict[str, float],
    show_ylabel: bool,
) -> Dict[str, MassClosureStats]:
    """
    References
    ----------
    - See the original Fig. 8 script output: paper_bundle\\paper\\figure\\SOA_mass_closure_deltaM_vs_I.png
    - See the gate diagnostics figure: paper_bundle\\paper\\figure\\formation_rate_vs_survival.png

    Mathematical expression
    -----------------------
    I_Delta(t) = int_{t-Delta}^{t} J_app(t) S_surv(t) dt
    DeltaSOA_Delta(t) = sum_{k=t-w+1}^{t} delta_mass(k)
    DeltaSOA_Delta approx alpha * I_Delta

    Parameter meanings
    ------------------
    - delta_label: window label for display (e.g., "10 s", "1 min", "60 min").
    - window_steps: window length in 10 s steps (w = Delta / DT_SECONDS).
    - alpha_theory_by_place: theoretical alpha_th for each site, used for the reference line.
    - HIGH_I_QUANTILE: high-signal quantile threshold q for the subset fit.
    """
    stats_by_place: Dict[str, MassClosureStats] = {}
    x_store: list[np.ndarray] = []
    y_store: list[np.ndarray] = []
    for place in ["JH", "CM"]:
        sub = hf[hf["place"] == place].sort_index()
        if sub.empty:
            stats_by_place[place] = MassClosureStats(
                fit=FitStats(np.nan, np.nan, np.nan, np.nan),
                fit_hi=FitStats(np.nan, np.nan, np.nan, np.nan),
                n_hi=0,
                alpha_theory=float(alpha_theory_by_place.get(place, np.nan)),
            )
            continue
        prod = (sub["J_app"] * sub["S_surv"]).fillna(0.0)
        x = _rolling_integral(prod, window_steps, DT_SECONDS)
        # References: align with the mass-closure script (do not fill NaNs with zeros before rolling sum).
        # Equation: DeltaM_delta(t) = sum_{k=t-w+1}^t delta_mass(k).
        # Parameters: delta_mass is the 10 s increment from step05; w is the window size in 10 s steps.
        y = _rolling_sum(pd.to_numeric(sub["delta_mass"], errors="coerce"), window_steps)
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        if not x.empty and not y.empty:
            x_store.append(x.to_numpy())
            y_store.append(y.to_numpy())
        fit = _fit_linear(x, y)
        fit_hi, n_hi = _fit_high_i_with_fixed_intercept(
            x, y, q=HIGH_I_QUANTILE, intercept_fixed=float(fit.intercept)
        )
        stats_by_place[place] = MassClosureStats(
            fit=fit,
            fit_hi=fit_hi,
            n_hi=int(n_hi),
            alpha_theory=float(alpha_theory_by_place.get(place, np.nan)),
        )

        xs, ys = _downsample_xy(x, y, max_points=2000)
        marker = "s" if place == "JH" else "o"
        ax.scatter(
            xs,
            ys,
            s=9,
            alpha=0.18,
            color=COLORS[place],
            edgecolors="none",
            marker=marker,
            rasterized=True,
        )

        # Lines are drawn later using a shared panel-wide x-range so all lines have equal lengths.

    ax.set_title(delta_label)
    ax.set_xlabel(r"$\mathbf{I}_{\boldsymbol{\Delta}}$ ($\mathbf{cm}^{-3}$)")
    if show_ylabel:
        ax.set_ylabel(r"$\boldsymbol{\Delta}\mathbf{SOA}_{\boldsymbol{\Delta}}$ ($\boldsymbol{\mu}\mathbf{g}\ \cdot\ \mathbf{m}^{-3}$)")
    _apply_axis_style(ax)

    # Zoom axis limits based on central quantiles to avoid extreme outliers dominating.
    if x_store and y_store:
        x_all = np.concatenate(x_store)
        y_all = np.concatenate(y_store)
        x_max_q = float(np.nanquantile(x_all, 0.995)) if np.isfinite(x_all).any() else np.nan
        y_lo = float(np.nanquantile(y_all, 0.005)) if np.isfinite(y_all).any() else np.nan
        y_hi = float(np.nanquantile(y_all, 0.995)) if np.isfinite(y_all).any() else np.nan
        if np.isfinite(x_max_q) and x_max_q > 0:
            ax.set_xlim(0.0, x_max_q * 1.05)
        if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
            pad = 0.10 * (y_hi - y_lo)
            ax.set_ylim(y_lo - pad, y_hi + pad)

    # Draw theory and fit lines using the shared axis limits (aligns with the Fig. 8 styling logic).
    x_min_plot, x_max_plot = ax.get_xlim()
    xline_full = np.linspace(max(0.0, float(x_min_plot)), float(x_max_plot), 100)
    for place in ["JH", "CM"]:
        alpha_theory = float(alpha_theory_by_place.get(place, np.nan))
        if np.isfinite(alpha_theory):
            line_th = ax.plot(
                xline_full,
                alpha_theory * xline_full,
                color=COLORS["theory"],
                linewidth=1.8,
                linestyle="--" if place == "JH" else "-.",
                alpha=0.85,
                zorder=4,
            )[0]
            if place == "JH":
                line_th.set_dashes([6, 4])
            else:
                 line_th.set_dashes([2, 4])
        # Use the high-signal slope (alpha_hat_hi) for the displayed fit line to stay
        # consistent with the panel annotations and the resolution-dependence panels below.
        fit = stats_by_place.get(place).fit_hi if place in stats_by_place else None
        if fit is None or not np.isfinite(fit.slope) or not np.isfinite(fit.intercept):
            fit = stats_by_place.get(place).fit if place in stats_by_place else None
        if fit is not None and np.isfinite(fit.slope) and np.isfinite(fit.intercept):
            ax.plot(
                xline_full,
                fit.slope * xline_full + fit.intercept,
                color=COLORS[place],
                linewidth=1.7,
                linestyle="-",
                zorder=5,
            )
    return stats_by_place


def _plot_slope_vs_resolution(
    axes: Tuple[plt.Axes, plt.Axes],
    table: pd.DataFrame,
    alpha_theory_by_place: Dict[str, float],
) -> None:
    """
    Plot theoretical vs fitted slopes across integration windows (JH and CM, stacked).
    """
    table = table.copy()
    table = table[(table["Lag_min"] == 0.0) & (table["Place"].isin(["JH", "CM"]))].dropna(
        subset=["Delta_min", "alpha_hat"]
    )
    if table.empty:
        for ax in axes:
            ax.axis("off")
        return
    table["resolution_s"] = table["Delta_min"].astype(float) * 60.0

    for ax, place in zip(axes, ["JH", "CM"]):
        sub = table[table["Place"] == place].sort_values("resolution_s")
        if sub.empty:
            ax.axis("off")
            continue
        x = sub["resolution_s"].to_numpy()
        col = "alpha_hat_highI" if "alpha_hat_highI" in sub.columns else "alpha_hat"
        y_fit = sub[col].to_numpy()
        ax.plot(
            x,
            y_fit,
            color=COLORS[place],
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            label=r"$\hat{\boldsymbol{\alpha}}_{\mathbf{hi}}$",
        )
        alpha_theory = alpha_theory_by_place.get(place, np.nan)
        if np.isfinite(alpha_theory):
            ax.axhline(
                alpha_theory,
                color=COLORS["theory"],
                linewidth=1.8,
                linestyle="--",
                label=r"$\boldsymbol{\alpha}_{\mathbf{th}}$",
            )

        ax.set_xscale("log")
        ax.set_ylabel(r"$\hat{\boldsymbol{\alpha}}_{\mathbf{hi}}$ ($\boldsymbol{\mu}\mathbf{g}\ \mathbf{m}^{-3}\ \mathbf{per}\ \mathbf{cm}^{-3}$)")
        ax.set_title(f"{place}")
        _apply_axis_style(ax)
        if place == "JH":
            ax.legend(frameon=False, loc="upper right")

    axes[-1].set_xlabel("Temporal resolution (s)")


def _plot_slope_decay_panel(
    ax: plt.Axes,
    hf: pd.DataFrame,
    alpha_theory_by_place: Dict[str, float],
    *,
    window_steps: list[int] | None = None,
) -> None:
    """
    References
    ----------
    - See _plot_mass_closure_panel for the definition of I_Delta and DeltaSOA_Delta.

    Mathematical expression
    -----------------------
    For each window size Delta (steps*w = Delta/DT_SECONDS):
    - I_Delta(t) = int_{t-Delta}^{t} J_app(t) S_surv(t) dt
    - DeltaSOA_Delta(t) approx alpha_hat(Delta) * I_Delta(t)   (high-signal subset)

    Parameter meanings
    ------------------
    - window_steps: list of integration window sizes in 10 s steps.
    - HIGH_I_QUANTILE: high-signal threshold q (e.g., 0.8 keeps top 20%).
    - alpha_theory_by_place: theoretical alpha per site for reference.
    """
    if window_steps is None:
        # Keep the original checkpoints and add intermediate windows to increase point density.
        window_steps = [1, 2, 3, 4, 6, 12, 18, 30, 60, 120, 180, 240, 360]

    rows: list[dict[str, float | str]] = []
    for place in ["JH", "CM"]:
        sub = hf[hf["place"] == place].sort_index()
        if sub.empty:
            continue
        prod = (sub["J_app"] * sub["S_surv"]).fillna(0.0)
        delta_mass = pd.to_numeric(sub["delta_mass"], errors="coerce")
        for w in window_steps:
            if int(w) < 1:
                continue
            x = _rolling_integral(prod, int(w), DT_SECONDS)
            y = _rolling_sum(delta_mass, int(w))
            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]
            if len(x) < 20:
                continue
            fit = _fit_linear(x, y)
            intercept_fixed = float(fit.intercept) if np.isfinite(fit.intercept) else 0.0
            fit_hi, _ = _fit_high_i_with_fixed_intercept(x, y, q=HIGH_I_QUANTILE, intercept_fixed=intercept_fixed)
            if not np.isfinite(fit_hi.slope):
                continue
            rows.append(
                {
                    "Place": place,
                    "resolution_s": float(w) * float(DT_SECONDS),
                    "alpha_hat_hi": float(fit_hi.slope),
                }
            )

    if not rows:
        ax.axis("off")
        return

    df = pd.DataFrame(rows).sort_values(["Place", "resolution_s"])
    for place in ["JH", "CM"]:
        sub = df[df["Place"] == place]
        if sub.empty:
            continue
        ax.plot(
            sub["resolution_s"].to_numpy(),
            sub["alpha_hat_hi"].to_numpy(),
            color=COLORS[place],
            linewidth=1.6,
            marker="o" if place == "CM" else "s",
            markersize=3.8,
            alpha=0.95,
        )

    # Reference theory lines (site-specific alpha_th as horizontal bands).
    for place in ["JH", "CM"]:
        alpha_th = float(alpha_theory_by_place.get(place, np.nan))
        if np.isfinite(alpha_th):
            ax.axhline(
                alpha_th,
                color=COLORS[place],
                linewidth=1.2,
                linestyle="--",
                alpha=0.75,
                zorder=1.0,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Resolution (s)")
    ax.set_ylabel(r"$\hat{\boldsymbol{\alpha}}_{\mathbf{hi}}$ ($\boldsymbol{\mu}\mathbf{g}\ \mathbf{m}^{-3}\ \mathbf{per}\ \mathbf{cm}^{-3}$)")
    _apply_axis_style(ax)

    handles = [
        Line2D([0], [0], color=COLORS["JH"], marker="s", linestyle="-", linewidth=1.6, markersize=4.2, label="JH"),
        Line2D([0], [0], color=COLORS["CM"], marker="o", linestyle="-", linewidth=1.6, markersize=4.2, label="CM"),
        Line2D([0], [0], color=COLORS["JH"], linestyle="--", linewidth=1.2, label=r"JH $\boldsymbol{\alpha}_{\mathbf{th}}$"),
        Line2D([0], [0], color=COLORS["CM"], linestyle="--", linewidth=1.2, label=r"CM $\boldsymbol{\alpha}_{\mathbf{th}}$"),
    ]
    leg = ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=10, ncol=2, columnspacing=1.0)
    _legend_no_background(leg)


def _plot_delta_soa_distribution_inset(
    ax: plt.Axes, hf: pd.DataFrame, *, max_points: int = 60000
) -> None:
    """
    Panel D inset: Delta SOA distribution with zero-mean Gaussian overlay.
    """
    sigma_by_place: Dict[str, float] = {}
    for place in ["JH", "CM"]:
        sub = hf[hf["place"] == place]
        vals = pd.to_numeric(sub["delta_mass"], errors="coerce").dropna()
        if vals.empty:
            continue
        vals = vals - float(vals.mean())
        if len(vals) > max_points:
            vals = vals.sample(n=max_points, random_state=0)
        sigma = float(vals.std(ddof=1)) if len(vals) > 2 else np.nan
        sigma_by_place[place] = float(sigma) if np.isfinite(sigma) else np.nan
        bins = 35
        ax.hist(
            vals.to_numpy(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=COLORS[place],
            label=f"{place}",
        )
        if np.isfinite(sigma) and sigma > 0:
            xgrid = np.linspace(float(vals.quantile(0.01)), float(vals.quantile(0.99)), 300)
            pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(xgrid**2) / (2.0 * sigma**2))
            ax.plot(xgrid, pdf, color=COLORS[place], linewidth=1.2, alpha=0.8)

    ax.set_title(r"Zero-mean $\boldsymbol{\Delta}\mathbf{SOA}$ noise")
    ax.set_xlabel(r"$\boldsymbol{\Delta}\mathbf{SOA}$ ($\boldsymbol{\mu}\mathbf{g}\ \cdot\ \mathbf{m}^{-3}$)")
    ax.set_ylabel("PDF")
    _apply_axis_style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    if sigma_by_place:
        ax.text(
            0.02,
            0.96,
            rf"$\boldsymbol{{\sigma}}$ (JH)={sigma_by_place.get('JH', np.nan):.3g}"
            "\n"
            rf"$\boldsymbol{{\sigma}}$ (CM)={sigma_by_place.get('CM', np.nan):.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )


def _compute_normality_diagnostics(z: np.ndarray) -> Dict[str, float]:
    """
    References
    ----------
    - Jarque, C. M., and Bera, A. K. (1987) Int. Stat. Rev.
    - Shapiro, S. S., and Wilk, M. B. (1965) Biometrika.
    - Anderson, T. W., and Darling, D. A. (1952) Ann. Math. Stat.

    Mathematical expression
    -----------------------
    - Standardization: z = (x - mean(x)) / std(x)
    - Jarque-Bera: JB = n/6 * (S^2 + (K-3)^2/4), where S is skewness and K is kurtosis.
    - One-sample KS compares empirical CDF of z with Phi(z) (standard normal CDF).

    Parameter meanings
    ------------------
    - z: standardized sample values intended to be compared with N(0, 1).
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = int(z.size)
    if n < 8:
        return {
            "n": float(n),
            "jb_p": np.nan,
            "ks_p": np.nan,
            "sw_p": np.nan,
            "ad_stat": np.nan,
            "ad_crit_5": np.nan,
            "ad_reject_5": np.nan,
        }

    # NOTE: KS p-value is only diagnostic here because mean/std are estimated from data.
    jb = stats.jarque_bera(z)
    ks = stats.kstest(z, "norm")

    # Shapiro-Wilk is sensitive and costly for large n; subsample for diagnostics only.
    rng = np.random.default_rng(0)
    z_sw = z
    if n > 5000:
        z_sw = rng.choice(z, size=5000, replace=False)
    sw = stats.shapiro(z_sw)

    ad = stats.anderson(z, dist="norm")
    ad_crit_5 = np.nan
    for sl, cv in zip(ad.significance_level, ad.critical_values):
        if float(sl) == 5.0:
            ad_crit_5 = float(cv)
            break
    ad_reject_5 = float(ad.statistic > ad_crit_5) if np.isfinite(ad_crit_5) else np.nan

    return {
        "n": float(n),
        "jb_p": float(getattr(jb, "pvalue", jb[1])),
        "ks_p": float(getattr(ks, "pvalue", ks[1])),
        "sw_p": float(getattr(sw, "pvalue", sw[1])),
        "ad_stat": float(ad.statistic),
        "ad_crit_5": float(ad_crit_5),
        "ad_reject_5": float(ad_reject_5),
    }


def _plot_delta_soa_distribution_with_diagnostics(
    axes: Tuple[plt.Axes, plt.Axes],
    hf: pd.DataFrame,
    *,
    max_points: int = 60000,
) -> None:
    """
    References
    ----------
    - Normality diagnostics references: see _compute_normality_diagnostics.

    Mathematical expression
    -----------------------
    - DeltaSOA(t) = delta_mass(t) - mean(delta_mass) (zero-mean increment).
    - z = DeltaSOA / sigma, where sigma is sample std (ddof=1).

    Parameter meanings
    ------------------
    - axes: (ax_hist, ax_qq)
    - hf: high-frequency dataframe containing delta_mass and place columns.
    """
    ax_hist, ax_qq = axes

    # Prefer the same delta definitions used by SOA_timeseries_stationarity.png when available.
    use_stationarity = {"delta_SOA", "delta_CS", "delta_C_hat"}.issubset(hf.columns)
    if use_stationarity:
        series_specs: list[tuple[str, str, str, str]] = [
            ("delta_SOA", r"$\boldsymbol{\Delta}\mathbf{SOA}$", "#3C5488", "s"),
            ("delta_CS", r"$\boldsymbol{\Delta}\mathbf{CS}$", "#00A087", "o"),
            ("delta_C_hat", r"$\boldsymbol{\Delta}\mathbf{C}$", "#E64B35", "^"),
        ]
        pooled = hf.copy()
    else:
        # Backward compatible fallback if the stationarity cache is missing.
        series_specs = [("delta_mass", r"$\boldsymbol{\Delta}\mathbf{SOA}$", COLORS["accent"], "o")]
        pooled = hf.copy()

    test_by_series: Dict[str, Dict[str, float]] = {}
    z_by_series: Dict[str, np.ndarray] = {}

    bins = np.linspace(-4.0, 4.0, 41)
    for col, label, color, _marker in series_specs:
        vals = pd.to_numeric(pooled[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        if len(vals) > max_points:
            vals = vals.sample(n=max_points, random_state=0)
        sigma = float(vals.std(ddof=1)) if len(vals) > 2 else np.nan
        if not (np.isfinite(sigma) and sigma > 0):
            continue
        x = vals.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 20:
            continue
        z = x / sigma
        z = z[np.isfinite(z)]
        if z.size < 20:
            continue
        z_by_series[col] = z

        # References
        # ----------
        # - Wilcoxon signed-rank test: Wilcoxon, F. (1945) Biometrics Bulletin.
        # - Sign test (binomial): nonparametric test of P(X>0)=0.5 for symmetry around zero.
        #
        # Mathematical expression
        # -----------------------
        # - Wilcoxon tests median(x)=0 using signed ranks of |x_i| for x_i != 0.
        # - Sign test uses k ~ Binomial(n, 0.5) where k is the count of x_i > 0 (excluding zeros).
        #
        # Parameter meanings
        # ------------------
        # - x: increment sample (not centered).
        # - p_wsr: p-value for Wilcoxon signed-rank test of median=0.
        # - p_sign: p-value for sign symmetry test around zero.
        x_test = x
        if x_test.size > 5000:
            x_test = np.random.default_rng(0).choice(x_test, size=5000, replace=False)
        x_nz = x_test[x_test != 0.0]
        p_wsr = np.nan
        if x_nz.size >= 20:
            try:
                wsr = stats.wilcoxon(x_nz, alternative="two-sided", zero_method="wilcox")
                p_wsr = float(getattr(wsr, "pvalue", wsr[1]))
            except Exception:
                p_wsr = np.nan
        mean_obs = float(np.mean(x_test)) if x_test.size > 0 else np.nan
        mean_z = float(mean_obs / sigma) if np.isfinite(sigma) and sigma > 0 else np.nan
        ci_lo = np.nan
        ci_hi = np.nan
        p_mean = np.nan
        if x_test.size >= 20 and np.isfinite(mean_obs):
            rng = np.random.default_rng(0)
            b = 1200
            boot_means = rng.choice(x_test, size=(b, x_test.size), replace=True).mean(axis=1)
            ci_lo = float(np.quantile(boot_means, 0.025) / sigma)
            ci_hi = float(np.quantile(boot_means, 0.975) / sigma)
            # Sign-flip permutation for mean=0 under symmetric null (emphasizes cancellation).
            flips = rng.choice(np.array([-1.0, 1.0]), size=(b, x_test.size), replace=True)
            perm_means = (flips * x_test).mean(axis=1)
            p_mean = float(np.mean(np.abs(perm_means) >= abs(mean_obs)))
        pos = int(np.sum(x_test > 0.0))
        neg = int(np.sum(x_test < 0.0))
        p_sign = np.nan
        if pos + neg >= 20:
            try:
                if hasattr(stats, "binomtest"):
                    bt = stats.binomtest(pos, n=pos + neg, p=0.5, alternative="two-sided")
                    p_sign = float(bt.pvalue)
                else:
                    p_sign = float(stats.binom_test(pos, n=pos + neg, p=0.5))
            except Exception:
                p_sign = np.nan
        test_by_series[col] = {
            "mean_z": mean_z,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_mean": p_mean,
            "p_wsr": p_wsr,
            "p_sign": p_sign,
        }

        ax_hist.hist(
            z,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )

    # Standard normal reference.
    xgrid = np.linspace(-4.0, 4.0, 400)
    ax_hist.plot(
        xgrid,
        stats.norm.pdf(xgrid),
        color=COLORS["theory"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.85,
        label=r"$\mathbf{N}(0,1)$",
    )

    ax_hist.set_title("Zero-mean increments (standardized)")
    ax_hist.set_xlabel("Standardized increment")
    ax_hist.set_ylabel("PDF")
    _apply_axis_style(ax_hist)
    handles, labels = ax_hist.get_legend_handles_labels()
    label_to_handle = {lab: h for h, lab in zip(handles, labels)}
    desired = [
        r"$\boldsymbol{\Delta}\mathbf{SOA}$",
        r"$\boldsymbol{\Delta}\mathbf{C}$",
        r"$\boldsymbol{\Delta}\mathbf{CS}$",
        r"$\mathbf{N}(0,1)$",
    ]
    ordered_handles = [label_to_handle[lab] for lab in desired if lab in label_to_handle]
    ordered_labels = [lab for lab in desired if lab in label_to_handle]
    leg_hist = ax_hist.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        fontsize=10.5,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.6,
    )
    _legend_no_background(leg_hist)

    if test_by_series:
        def _fmt_p01(pv: float) -> str:
            if not np.isfinite(pv):
                return "NA"
            return "<0.01" if pv < 0.01 else f"{pv:.2f}"

        def _fmt_no_neg_zero(x: float) -> str:
            if not np.isfinite(x):
                return "NA"
            s = f"{x:.2f}"
            return "0.00" if s == "-0.00" else s

        mean_map: dict[str, str] = {}
        p_map: dict[str, str] = {}
        label_math_map = {
            "delta_SOA": r"\boldsymbol{\Delta}\mathbf{SOA}",
            "delta_CS": r"\boldsymbol{\Delta}\mathbf{CS}",
            "delta_C_hat": r"\boldsymbol{\Delta}\mathbf{C}",
        }
        for col, _label, _color, _marker in series_specs:
            if col not in test_by_series:
                continue
            t = test_by_series[col]
            key = label_math_map.get(col, col)
            mean_map[key] = _fmt_no_neg_zero(float(t.get("mean_z", np.nan)))
            p_map[key] = _fmt_p01(float(t.get("p_mean", np.nan)))

        # Keep the annotation compact: report only mean and the sign-flip p-value for mean=0.
        # Start with a leading space so Matplotlib treats this as text with embedded math, allowing multiple $...$ segments.
        desired_keys = [r"\boldsymbol{\Delta}\mathbf{SOA}", r"\boldsymbol{\Delta}\mathbf{CS}", r"\boldsymbol{\Delta}\mathbf{C}"]
        lines: list[str] = []
        for k in desired_keys:
            if k not in mean_map or k not in p_map:
                continue
            lines.append(" " + rf"${k}: \mathbf{{mean}}={mean_map[k]}$")
            lines.append(" " + rf"$(\mathbf{{p}}_{{\mathbf{{sf}}}}={p_map[k]})$")
        ax_hist.text(
            0.985,
            0.970,
            "\n".join(lines),
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=9.2,
            fontweight="normal",
            color="#222222",
            linespacing=1.02,
        )

    # Normal Q-Q plot for standardized increments.
    ax_qq.set_title("Normal Q-Q (standardized)")
    p = np.linspace(0.02, 0.98, 180)
    x_theory = stats.norm.ppf(p)
    for col, label, color, marker in series_specs:
        if col not in z_by_series:
            continue
        z = z_by_series[col]
        y_sample = np.quantile(z, p)
        ax_qq.plot(
            x_theory,
            y_sample,
            linestyle="None",
            marker=marker,
            markersize=4.6,
            alpha=1.0,
            color=color,
            markeredgewidth=0.0,
            label=label,
        )
    lo = -3.0
    hi = 3.0
    ax_qq.plot([lo, hi], [lo, hi], color=COLORS["theory"], linewidth=1.2, linestyle="--", alpha=0.8)
    ax_qq.set_xlim(lo, hi)
    ax_qq.set_ylim(lo, hi)
    ax_qq.set_xlabel("Normal quantiles")
    ax_qq.set_ylabel("Sample quantiles")
    ax_qq.text(
        0.04,
        0.92,
        "Heavy tails",
        transform=ax_qq.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#111111",
    )
    _apply_axis_style(ax_qq)
    leg_qq = ax_qq.legend(frameon=False, fontsize=12, loc="lower right", ncol=1, markerscale=1.25)
    _legend_no_background(leg_qq)


def main() -> None:
    _set_figure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    output = japp_surv.load_japp_survival_checkpoint(CHECKPOINT_DIR)
    output = japp_surv.maybe_join_growth_metrics(output, CHECKPOINT_DIR)

    ratio_df, best_eta, gate_metrics = japp_surv.compute_survival_gate_fit(
        output, fallback_eta=1.0
    )
    eta_by_place = _compute_eta_by_place(output)
    s_obs_df, s_obs_meta = _compute_empirical_survival(
        output,
        cfg,
        d1_um=SURV_OBS_D1_UM,
        d2_um=SURV_OBS_D2_UM,
        window_minutes=SURV_OBS_WINDOW_MIN,
    )
    window_data = japp_surv.select_japp_surv_window(output)

    growth_metrics = None
    growth_path = CHECKPOINT_DIR / "step04_growth_metrics_hf.parquet"
    if growth_path.exists():
        growth_metrics = pd.read_parquet(growth_path)
    alpha_theory_by_place = {
        place: _alpha_theory_ug_m3_per_cm3(growth_metrics, place=place) for place in ["JH", "CM"]
    }

    table_path = CHECKPOINT_DIR / "Table_MassClosure_alpha.csv"
    table = pd.DataFrame()
    if table_path.exists():
        table = pd.read_csv(table_path)

    fig = plt.figure(figsize=(13.2, 7.2))
    fig.patch.set_facecolor("white")
    outer = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.05, 1.70],
        height_ratios=[1.0, 1.0],
        wspace=0.26,
        hspace=0.34,
    )

    # Panel A: Transition regime in (CS_star, |G|) with a gate side panel.
    a_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], width_ratios=[2.65, 0.95], wspace=0.30)
    ax_a = fig.add_subplot(a_spec[0, 0])
    ax_a_gate = fig.add_subplot(a_spec[0, 1])
    _add_panel_label(ax_a, "A")
    panel_a_ready = False
    panel_a_df = pd.DataFrame()
    if {"CS_star", "G_abs"}.issubset(output.columns):
        panel_a_df = output[["CS_star", "G_abs"]].replace([np.inf, -np.inf], np.nan).dropna()
        panel_a_df = panel_a_df[(panel_a_df["CS_star"] > 0) & (panel_a_df["G_abs"] > 0)]
    if not panel_a_df.empty:
        x = np.log10(panel_a_df["CS_star"].to_numpy())
        y = np.log10(panel_a_df["G_abs"].to_numpy())
        x_lo = float(np.nanmin(x))
        x_hi = float(np.nanmax(x))
        y_lo = float(np.nanmin(y))
        y_hi = float(np.nanmax(y))
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            x_lo, x_hi = 0.0, 1.0
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
            y_lo, y_hi = 0.0, 1.0
        x_span = x_hi - x_lo
        y_span = y_hi - y_lo
        pad_left = 0.01 * x_span
        pad_right = 0.07 * x_span
        pad_top = 0.01 * y_span
        pad_bottom = 0.07 * y_span
        hb = ax_a.hexbin(
            x,
            y,
            gridsize=65,
            mincnt=3,
            cmap="viridis",
            norm=LogNorm(),
            linewidths=0.0,
            rasterized=True,
        )
        ax_a.set_xlim(x_lo - pad_left, x_hi + pad_right)
        ax_a.set_ylim(y_lo - pad_bottom, y_hi + pad_top)
        x_line = np.linspace(x_lo, x_hi, 200)
        ratio_series = (panel_a_df["CS_star"] / panel_a_df["G_abs"]).replace([np.inf, -np.inf], np.nan)
        ratio_series = ratio_series[ratio_series > 0]
        log_ratio = np.log10(ratio_series.to_numpy(dtype=float))
        if log_ratio.size:
            d0 = int(np.round(np.nanmedian(log_ratio)))
            ratio_decades = sorted({d0 - 1, d0, d0 + 1, 5})
            ratio_levels = [10.0 ** d for d in ratio_decades]
        else:
            ratio_levels, ratio_decades = _select_ratio_decade_levels(log_ratio, max_levels=3)
        for r_level, r_decade in zip(ratio_levels, ratio_decades):
            ax_a.plot(
                x_line,
                x_line - np.log10(r_level),
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                color=COLORS["theory"],
            )
            pts: list[tuple[float, float]] = []
            y_at_x_lo = x_lo - np.log10(r_level)
            if y_lo <= y_at_x_lo <= y_hi:
                pts.append((x_lo, y_at_x_lo))
            y_at_x_hi = x_hi - np.log10(r_level)
            if y_lo <= y_at_x_hi <= y_hi:
                pts.append((x_hi, y_at_x_hi))
            x_at_y_lo = y_lo + np.log10(r_level)
            if x_lo <= x_at_y_lo <= x_hi:
                pts.append((x_at_y_lo, y_lo))
            x_at_y_hi = y_hi + np.log10(r_level)
            if x_lo <= x_at_y_hi <= x_hi:
                pts.append((x_at_y_hi, y_hi))
            if len(pts) >= 2:
                pts_sorted = sorted({(round(px, 6), round(py, 6)) for px, py in pts})
                x_mid = 0.5 * (pts_sorted[0][0] + pts_sorted[-1][0])
                y_mid = 0.5 * (pts_sorted[0][1] + pts_sorted[-1][1])
                p1 = ax_a.transData.transform((x_mid, y_mid))
                p2 = ax_a.transData.transform((x_mid + 1.0, y_mid + 1.0))
                angle = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
                ax_a.text(
                    x_mid,
                    y_mid,
                    rf"$\mathbf{{CS}}^{{\ast}}/\left|\mathbf{{G}}\right|=10^{{{r_decade}}}$",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    rotation=angle,
                    rotation_mode="anchor",
                    color=COLORS["theory"],
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                )
        cb = fig.colorbar(hb, ax=ax_a, orientation="vertical", fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8)
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        panel_a_ready = True

        ratio_series = (panel_a_df["CS_star"] / panel_a_df["G_abs"]).replace([np.inf, -np.inf], np.nan)
        ratio_series = ratio_series[ratio_series > 0]
        ratio_for_curve = ratio_series.copy()
        if not ratio_for_curve.empty:
            r_low = float(np.nanmin(ratio_for_curve))
            r_high = float(np.nanmax(ratio_for_curve))
            if not np.isfinite(r_low) or not np.isfinite(r_high) or r_high <= r_low:
                r_low, r_high = 1e-6, 1e-3
            r_low = max(r_low, 1e-6)
            r_high = max(r_high, r_low * 10.0)
            log_low = np.log10(r_low)
            log_high = np.log10(r_high)
            if np.isfinite(log_low) and np.isfinite(log_high) and log_high > log_low:
                pad = 0.12 * (log_high - log_low)
                r_low = 10.0 ** (log_low - pad)
                r_high = 10.0 ** (log_high + pad)
        else:
            r_low, r_high = 1e-3, 1e3
        log_r_low = float(np.log10(r_low)) if r_low > 0 else float("nan")
        log_r_high = float(np.log10(r_high)) if r_high > 0 else float("nan")
        # References: Kerminen and Kulmala (2002); Kulmala et al. (2007).
        # Equation: S_gate = 1 / (1 + eta * r), r = CS_star / G_abs.
        # Parameters: S_gate is dimensionless; eta is a dimensionless gate efficiency; r is CS_star over G_abs.
        r_vals = np.logspace(np.log10(r_low), np.log10(r_high), 200)
        ax_gate = ax_a_gate
        ax_hist = ax_gate.twinx()
        ax_gate.set_zorder(3)
        ax_hist.set_zorder(2)
        ax_gate.patch.set_alpha(0.0)
        ax_hist.patch.set_alpha(0.0)
        log_ratio_gate = log_ratio[np.isfinite(log_ratio)]
        q05 = float("nan")
        q50 = float("nan")
        q95 = float("nan")
        x_lo_gate = log_r_low
        x_hi_gate = log_r_high
        if log_ratio_gate.size:
            ax_hist.hist(
                log_ratio_gate,
                bins=18,
                density=True,
                histtype="stepfilled",
                linewidth=0.0,
                color="#777777",
                alpha=0.28,
            )
            q05, q50, q95 = np.nanquantile(log_ratio_gate, [0.05, 0.50, 0.95])
            if np.isfinite(q05):
                ax_gate.axvline(q05, color="#666666", linewidth=1.0, linestyle=":", alpha=0.8)
            if np.isfinite(q95):
                ax_gate.axvline(q95, color="#666666", linewidth=1.0, linestyle=":", alpha=0.8)
        if np.isfinite(x_lo_gate) and np.isfinite(x_hi_gate) and x_hi_gate > x_lo_gate:
            ax_gate.set_xlim(x_lo_gate, x_hi_gate)
        xgrid = np.linspace(x_lo_gate, x_hi_gate, 250)
        r_vals = 10.0 ** xgrid
        s_gate = 1.0 / (1.0 + best_eta * r_vals)
        ax_gate.plot(
            xgrid,
            s_gate,
            color=COLORS["accent"],
            linewidth=2.4,
            solid_capstyle="round",
            zorder=4,
        )
        gate_top = float(np.nanquantile(s_gate, 0.995)) * 1.3
        if not np.isfinite(gate_top) or gate_top <= 0.0:
            gate_top = 0.05
        ax_gate.set_ylim(0.0, min(1.0, gate_top))
        ax_hist.set_yticks([])
        ax_hist.set_ylabel("")
        ax_hist.grid(False)
        ax_gate.set_xlabel(r"$\log_{10}(\mathbf{CS}^{\ast}/\left|\mathbf{G}\right|)$", fontsize=9)
        ax_gate.set_ylabel(r"$\mathbf{S}_{\mathbf{surv}}$ (-)", fontsize=9, color=COLORS["accent"])
        ax_gate.set_title("Gate mapping", fontsize=10, fontweight="bold")
        ax_gate.tick_params(axis="x", labelsize=7)
        ax_gate.tick_params(axis="y", labelsize=7, labelcolor=COLORS["accent"])
        _apply_axis_style(ax_gate)
        for spine in ax_hist.spines.values():
            spine.set_color(COLORS["spine"])
            spine.set_linewidth(1.0)
            spine.set_alpha(0.3)
        eq_text = (
            r"$\mathbf{S}_{\mathbf{surv}}=\frac{1}{1+\boldsymbol{\eta}\,\mathbf{CS}^{\ast}/\left|\mathbf{G}\right|}$"
            "\n" + rf"$\boldsymbol{{\eta}}={best_eta:.3f}$"
        )
        ax_gate.text(
            0.04,
            0.92,
            eq_text,
            transform=ax_gate.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color="#222222",
        )
        # Removed lower-right annotation per request.

    ax_a.set_title(r"Transition regime in $\mathbf{CS}^{\ast}$ and $\left|\mathbf{G}\right|$")
    ax_a.set_xlabel(r"$\log_{10}(\mathbf{CS}^{\ast})$ ($\mathbf{s}^{-1}$)")
    ax_a.set_ylabel(r"$\log_{10}(\left|\mathbf{G}\right|)$ ($\mathbf{nm}\ \mathbf{s}^{-1}$)")
    _apply_axis_style(ax_a)
    if not panel_a_ready:
        ax_a.text(
            0.5,
            0.5,
            "CS_star or G_abs not available",
            transform=ax_a.transAxes,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color="#222222",
        )
        ax_a_gate.axis("off")

    # Panel B: Process link time series (anti-correlation) with a split axis:
    # upper half shows J_app (positive), lower half shows S_surv (positive but plotted downward).
    b_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 0], height_ratios=[1.0, 1.0], hspace=0.0)
    ax_b_top = fig.add_subplot(b_spec[0, 0])
    ax_b_bot = fig.add_subplot(b_spec[1, 0], sharex=ax_b_top)
    _add_panel_label(ax_b_top, "B")

    legend_entries_j: list[tuple[str, str, str]] = []
    legend_entries_s: list[tuple[str, str, str]] = []
    places = ["JH", "CM"] if ("place" in window_data.columns) else ["ALL"]
    for place in places:
        sub = window_data[window_data["place"] == place] if place != "ALL" else window_data
        if sub.empty:
            continue
        j_series = sub["J_app"].copy()
        s_series = sub["S_surv"].copy()
        if "gap" in sub.columns:
            j_series[sub["gap"]] = np.nan
            s_series[sub["gap"]] = np.nan
        colors = getattr(japp_surv, "SITE_COLORS", {}).get(place, getattr(japp_surv, "SINGLE_COLORS", {}))
        j_color = colors.get("japp", "#3C5488")
        s_color = colors.get("surv", "#00A087")

        ax_b_top.plot(
            sub["elapsed_hours"],
            j_series,
            color=j_color,
            linewidth=1.2,
            alpha=0.95,
            zorder=2,
        )
        ax_b_top.fill_between(
            sub["elapsed_hours"],
            0.0,
            j_series,
            where=pd.to_numeric(j_series, errors="coerce").notna(),
            color=j_color,
            alpha=0.08,
            linewidth=0.0,
            zorder=1,
        )
        ax_b_bot.plot(
            sub["elapsed_hours"],
            s_series,
            color=s_color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.55,
            zorder=2,
        )
        ax_b_bot.fill_between(
            sub["elapsed_hours"],
            0.0,
            s_series,
            where=pd.to_numeric(s_series, errors="coerce").notna(),
            color=s_color,
            alpha=0.06,
            linewidth=0.0,
            zorder=1,
        )
        if place != "ALL":
            legend_entries_j.append((rf"$\mathbf{{J}}_{{\mathbf{{app}}}}$ ({place})", j_color, "-"))
            legend_entries_s.append((rf"$\mathbf{{S}}_{{\mathbf{{surv}}}}$ ({place})", s_color, "--"))

    # Style: use the shared boundary as the zero-axis.
    ax_b_top.axhline(0.0, color=COLORS["spine"], linewidth=1.0, alpha=0.9)
    ax_b_bot.axhline(0.0, color=COLORS["spine"], linewidth=1.0, alpha=0.9)
    ax_b_top.spines["bottom"].set_visible(False)
    ax_b_bot.spines["top"].set_visible(False)

    ax_b_top.set_title(r"Process link: $\mathbf{J}_{\mathbf{app}}$ vs $\mathbf{S}_{\mathbf{surv}}$ (10 s)")
    ax_b_bot.set_xlabel("Elapsed time (h)")
    ax_b_top.set_ylabel(r"$\mathbf{J}_{\mathbf{app}}$ ($\mathbf{cm}^{-3}\ \mathbf{s}^{-1}$)")
    ax_b_bot.set_ylabel(r"$\mathbf{S}_{\mathbf{surv}}$ (-)")
    _apply_axis_style(ax_b_top)
    _apply_axis_style(ax_b_bot)
    plt.setp(ax_b_top.get_xticklabels(), visible=False)

    # Limits: upper axis from 0 upward; lower axis inverted so values extend downward from 0.
    if not window_data.empty and "J_app" in window_data.columns and "S_surv" in window_data.columns:
        j_q = float(pd.to_numeric(window_data["J_app"], errors="coerce").quantile(0.99))
        s_q = float(pd.to_numeric(window_data["S_surv"], errors="coerce").quantile(0.99))
        if np.isfinite(j_q) and j_q > 0:
            ax_b_top.set_ylim(0.0, j_q * 1.10)
        if np.isfinite(s_q) and s_q > 0:
            ax_b_bot.set_ylim(s_q * 1.10, 0.0)

    # Pearson correlation between J_app and S_surv over the displayed window (all samples).
    r_val = np.nan
    pairs = window_data[["J_app", "S_surv"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pairs) >= 3:
        r_val = float(pairs["J_app"].corr(pairs["S_surv"], method="pearson"))
    if np.isfinite(r_val):
        ax_b_top.text(
            0.98,
            0.92,
            f"Pearson r = {r_val:.2f}",
            transform=ax_b_top.transAxes,
            ha="right",
            va="top",
            fontsize=10.0,
            fontweight="bold",
            color="#111111",
        )

    if legend_entries_j:
        proxies_j: list[Line2D] = []
        labels_j: list[str] = []
        seen_j = set()
        for label, color, linestyle in legend_entries_j:
            if label in seen_j:
                continue
            seen_j.add(label)
            proxies_j.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.6))
            labels_j.append(label)
        leg_j = ax_b_top.legend(proxies_j, labels_j, frameon=False, loc="upper left", ncol=1, fontsize=10.0)
        _legend_no_background(leg_j)

    if legend_entries_s:
        proxies_s: list[Line2D] = []
        labels_s: list[str] = []
        seen_s = set()
        for label, color, linestyle in legend_entries_s:
            if label in seen_s:
                continue
            seen_s.add(label)
            proxies_s.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.6))
            labels_s.append(label)
        leg_s = ax_b_bot.legend(proxies_s, labels_s, frameon=False, loc="lower left", ncol=1, fontsize=10.0)
        _legend_no_background(leg_s)

    # Panel C: Mass closure (3 windows) + slope decay (second row).
    right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[:, 1], height_ratios=[1.10, 0.90], hspace=0.34)
    c_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=right[0, 0], height_ratios=[1.0, 0.62], hspace=0.62)
    c_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=c_spec[0, 0], wspace=0.34)
    ax_c1 = fig.add_subplot(c_top[0, 0])
    ax_c2 = fig.add_subplot(c_top[0, 1])
    ax_c3 = fig.add_subplot(c_top[0, 2])
    ax_c4 = fig.add_subplot(c_spec[1, 0])
    _add_panel_label(ax_c1, "C")

    hf = output.copy()
    if "Time" in hf.columns:
        hf["Time"] = pd.to_datetime(hf["Time"])
        hf = hf.set_index("Time")
    hf = hf.sort_index()
    if "place" not in hf.columns:
        hf["place"] = "ALL"

    # References: see _fit_linear and _alpha_theory_ug_m3_per_cm3.
    # Equation: DeltaSOA_Delta approx alpha * I_Delta, where I_Delta = int_{t-Delta}^{t} J_app(t) S_surv(t) dt.
    # Parameters: Delta is window length (s), alpha is conversion slope, J_app is formation rate, S_surv is survival probability.
    stats_10 = _plot_mass_closure_panel(
        ax_c1,
        hf,
        delta_label="10 s",
        window_steps=1,
        alpha_theory_by_place=alpha_theory_by_place,
        show_ylabel=True,
    )
    stats_60 = _plot_mass_closure_panel(
        ax_c2,
        hf,
        delta_label="1 min",
        window_steps=6,
        alpha_theory_by_place=alpha_theory_by_place,
        show_ylabel=False,
    )
    stats_3600 = _plot_mass_closure_panel(
        ax_c3,
        hf,
        delta_label="60 min",
        window_steps=int(round(60.0 * 60.0 / DT_SECONDS)),
        alpha_theory_by_place=alpha_theory_by_place,
        show_ylabel=False,
    )

    def _fmt_sci(x: float) -> str:
        if not np.isfinite(x) or x == 0.0:
            return "NA" if not np.isfinite(x) else "0"
        exp = int(np.floor(np.log10(abs(float(x)))))
        mant = float(x) / (10.0**exp)
        return rf"${mant:.2f}\times 10^{{{exp:d}}}$"

    def _fmt_r2(x: float) -> str:
        return f"{x:.2f}" if np.isfinite(x) else "NA"

    def _add_mass_stats_box(ax: plt.Axes, stats_by_place: Dict[str, MassClosureStats]) -> None:
        jh = stats_by_place.get(
            "JH",
            MassClosureStats(
                fit=FitStats(np.nan, np.nan, np.nan, np.nan),
                fit_hi=FitStats(np.nan, np.nan, np.nan, np.nan),
                n_hi=0,
                alpha_theory=np.nan,
            ),
        )
        cm = stats_by_place.get(
            "CM",
            MassClosureStats(
                fit=FitStats(np.nan, np.nan, np.nan, np.nan),
                fit_hi=FitStats(np.nan, np.nan, np.nan, np.nan),
                n_hi=0,
                alpha_theory=np.nan,
            ),
        )
        txt = (
            r"$\mathbf{q}=$" + f"{HIGH_I_QUANTILE:.1f}" + " (high-signal)\n"
            + r"JH: $\hat{\boldsymbol{\alpha}}_{\mathbf{hi}}$=" + f"{_fmt_sci(jh.fit_hi.slope)}, " + r"$\mathbf{R}^2_{\mathbf{uc,hi}}$=" + f"{_fmt_r2(jh.fit_hi.r2_uncentered)}\n"
            + r"CM: $\hat{\boldsymbol{\alpha}}_{\mathbf{hi}}$=" + f"{_fmt_sci(cm.fit_hi.slope)}, " + r"$\mathbf{R}^2_{\mathbf{uc,hi}}$=" + f"{_fmt_r2(cm.fit_hi.r2_uncentered)}"
        )
        ax.text(
            0.98,
            0.02,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#111111",
            linespacing=1.05,
        )

    _add_mass_stats_box(ax_c1, stats_10)
    _add_mass_stats_box(ax_c2, stats_60)
    _add_mass_stats_box(ax_c3, stats_3600)

    # Compact legend for panel C (site markers + line styles).
    th_jh = _fmt_sci(float(alpha_theory_by_place.get("JH", np.nan))).strip("$")
    th_cm = _fmt_sci(float(alpha_theory_by_place.get("CM", np.nan))).strip("$")
    jh_theory_proxy = Line2D(
        [0],
        [0],
        color=COLORS["theory"],
        linestyle="--",
        linewidth=1.8,
        label=rf"JH $\boldsymbol{{\alpha}}_{{\mathbf{{th}}}}={th_jh}$",
    )
    jh_theory_proxy.set_dashes([6, 4])
    cm_theory_proxy = Line2D(
        [0],
        [0],
        color=COLORS["theory"],
        linestyle="-.",
        linewidth=1.8,
        label=rf"CM $\boldsymbol{{\alpha}}_{{\mathbf{{th}}}}={th_cm}$",
    )
    cm_theory_proxy.set_dashes([2, 4])
    proxies_c = [
        Line2D([0], [0], marker="s", linestyle="None", color=COLORS["JH"], markersize=5, label="JH"),
        Line2D([0], [0], marker="o", linestyle="None", color=COLORS["CM"], markersize=5, label="CM"),
        Line2D([0], [0], color="#111111", linestyle="-", linewidth=1.8, label=r"$\hat{\boldsymbol{\alpha}}_{\mathbf{hi}}$ fit"),
        jh_theory_proxy,
        cm_theory_proxy,
    ]
    # Use a figure-level legend to guarantee a true single line without colliding with subplot titles.
    leg_c = fig.legend(
        handles=proxies_c,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.75, 0.972),
        fontsize=9.6,
        ncol=5,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    _legend_no_background(leg_c)
    for ax in (ax_c1, ax_c2, ax_c3):
        ax.xaxis.labelpad = 1
    _plot_slope_decay_panel(ax_c4, hf, alpha_theory_by_place)

    # Panel D: Delta distributions + diagnostics (Q-Q side-by-side).
    d_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=right[1, 0], width_ratios=[1.05, 0.95], wspace=0.30)
    ax_d1 = fig.add_subplot(d_spec[0, 0])
    ax_d2 = fig.add_subplot(d_spec[0, 1])
    _add_panel_label(ax_d1, "D")
    ts_hour = pd.DataFrame()
    ts_hour_path = CHECKPOINT_DIR / "ts_stationarity_df_hour.parquet"
    if ts_hour_path.exists():
        ts_hour = pd.read_parquet(ts_hour_path)
    _plot_delta_soa_distribution_with_diagnostics((ax_d1, ax_d2), ts_hour if not ts_hour.empty else hf)

    out_path = FIGURE_DIR / TARGET_NAME
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.90)
    _raise_all_text_to_front(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
