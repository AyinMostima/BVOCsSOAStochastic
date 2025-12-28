"""
Time-series dashboard for BVOCs, SOA, CS, and their increments with stationarity and cointegration tests.
This script loads the cleaned dataset, computes condensation sink (CS), builds hourly aggregates,
and generates a Nature-style composite figure with:
  (a) BVOCs, SOA, CS time series (per site, dual-axis);
  (b) $\\Delta$SOA and $\\Delta$CS dynamics;
  (c) Stationarity (ADF, KPSS) and cointegration (BVOCs-SOA, CS-SOA) significance summary.

Usage:
  python scripts\\time_series_stationarity.py

Outputs:
  figures\\Fig_timeseries_stationarity.png
  figures\\Fig_timeseries_stationarity.svg
  figures\\Fig_delta_SOA_JH.[png|svg]
  figures\\Fig_delta_SOA_CM.[png|svg]
  figures\\Fig_delta_SOA_combined.[png|svg]
  figures\\SOA_process_scale_case.[png|svg]
"""

from __future__ import annotations

import sys
import warnings
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from matplotlib import cm, font_manager
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator, date2num, num2date
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator
from scipy.stats import gaussian_kde, norm
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

THIS_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

warnings.filterwarnings("ignore", category=InterpolationWarning)

from src.workflow import modeling_framework as mf
from src.workflow.config import WorkflowConfig, default_config


@dataclass
class TestResult:
    place: str
    series: str
    adf_p: float
    kpss_p: float


def _register_helvetica() -> None:
    font_files = [
        THIS_ROOT.parent / "HelveticaNeueLTPro-Roman.otf",
        THIS_ROOT.parent / "HelveticaNeueLTPro-Bd.otf",
        THIS_ROOT.parent / "HelveticaNeueLTPro-It.otf",
    ]
    for font_file in font_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))


def _set_style() -> None:
    _register_helvetica()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue LT Pro", "Helvetica", "Arial", "DejaVu Sans"],
            "font.weight": "bold",
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "font.size": 16,
            "legend.fontsize": 14,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.dpi": 150,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
        }
    )
    sns.set_palette("colorblind")


def _apply_axis_style(ax) -> None:
    """Lightweight axis styling for consistent background and spine clarity."""
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#4d4d4d")
        spine.set_linewidth(0.8)


def _apply_time_ticks(ax, rotation: float = 20.0, hour_interval: int = 4) -> None:
    """Standardize time axis ticks to hour:minute."""
    ax.xaxis.set_major_locator(HourLocator(interval=hour_interval))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
        label.set_fontweight("bold")


def _apply_burst_time_ticks(ax, rotation: float = 15.0) -> None:
    """Use 30-min ticks for zoomed burst panels."""
    ax.xaxis.set_major_locator(MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
        label.set_fontweight("bold")


def _apply_stitched_time_ticks(
    ax,
    total_minutes: float,
    origin: pd.Timestamp,
    rotation: float = 0.0,
) -> None:
    """Use coarser elapsed-hour ticks for stitched panels to avoid label crowding."""
    interval = 60
    if total_minutes > 12 * 60:
        interval = 180
    if total_minutes > 24 * 60:
        interval = 360
    if total_minutes > 48 * 60:
        interval = 720
    origin_ts = pd.to_datetime(origin)
    # Use fixed ticks anchored at the stitched origin (00:00) to avoid repeated HH:MM sequences.
    n_steps = int(np.floor(max(total_minutes, 0.0) / float(interval)))
    ticks = [origin_ts + pd.Timedelta(minutes=int(k * interval)) for k in range(n_steps + 1)]
    if not ticks:
        ticks = [origin_ts]
    ax.xaxis.set_major_locator(FixedLocator(date2num(ticks)))

    def _fmt_elapsed(x, _pos) -> str:
        dt = pd.Timestamp(num2date(x)).tz_localize(None)
        delta_min = (dt - origin_ts).total_seconds() / 60.0
        if not np.isfinite(delta_min):
            return ""
        total_h = int(round(delta_min / 60.0))
        if total_h < 0:
            total_h = 0
        return f"{total_h:d}"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_elapsed))
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("center")
        label.set_fontweight("bold")


def _apply_stitched_datetime_ticks(
    ax,
    segments: List[Dict[str, object]],
    rotation: float = 0.0,
    max_labels: int = 6,
) -> None:
    """
    Use stitched x positions but format ticks as original datetime to avoid "elapsed > 24h" confusion.
    Keep the number of labels small to prevent overlap in stitched multi-window figures.
    """
    if not segments:
        return
    segs = sorted(segments, key=lambda s: pd.to_datetime(s["start"]))
    # Prefer one tick per unique date (based on original timestamps) to avoid repeated labels.
    candidate_idxs: List[int] = []
    seen_dates: set[str] = set()
    for i, seg in enumerate(segs):
        t0 = pd.to_datetime(seg.get("orig_start", pd.NaT))
        if pd.isna(t0):
            continue
        key = t0.strftime("%Y-%m-%d")
        if key in seen_dates:
            continue
        seen_dates.add(key)
        candidate_idxs.append(i)
    if not candidate_idxs:
        candidate_idxs = list(range(len(segs)))
    if len(candidate_idxs) > max_labels:
        chosen = np.linspace(0, len(candidate_idxs) - 1, max_labels)
        idxs = sorted(set(candidate_idxs[int(round(float(v)))] for v in chosen))
    else:
        idxs = candidate_idxs
    ticks = [pd.to_datetime(segs[i]["start"]) for i in idxs]
    labels: List[str] = []
    for i in idxs:
        t0 = pd.to_datetime(segs[i].get("orig_start", pd.NaT))
        # Use explicit year to avoid confusing 06-20 vs 08-05 ordering across years.
        labels.append("" if pd.isna(t0) else t0.strftime("%y-%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(FixedLocator(date2num(ticks)))
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(FixedLocator([]))
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("center")
        label.set_fontweight("bold")
        label.set_fontsize(12)


def _compute_global_stationarity(df: pd.DataFrame) -> Dict[str, float]:
    # References: Dickey and Fuller (1979) ADF unit-root test; Engle and Granger (1987) two-step cointegration.
    # Equation: ADF fits Delta x_t = alpha + beta t + gamma x_{t-1} + sum_{i=1}^p phi_i Delta x_{t-i} + epsilon_t and tests gamma = 0; cointegration regresses S_t on B_t then applies ADF to residuals.
    # Parameters: x_t target series (BVOCs or SOA), B_t BVOC series, S_t SOA series, p lag order selected by AIC, gamma unit-root coefficient, phi_i short-run lag coefficients.
    df_use = df.copy()
    df_use["bvocs"] = pd.to_numeric(df_use.get("bvocs"), errors="coerce")
    df_use["SOA"] = pd.to_numeric(df_use.get("SOA"), errors="coerce")
    df_use = df_use.sort_values("Time")
    series_b = df_use["bvocs"].replace([np.inf, -np.inf], np.nan).dropna()
    series_s = df_use["SOA"].replace([np.inf, -np.inf], np.nan).dropna()
    aligned = pd.concat([series_b.rename("B"), series_s.rename("S")], axis=1).dropna()
    adf_b = np.nan
    adf_s = np.nan
    coint_p = np.nan
    try:
        if series_b.shape[0] >= 80:
            adf_b = float(adfuller(series_b, autolag="AIC")[1])
    except Exception:
        adf_b = np.nan
    try:
        if series_s.shape[0] >= 80:
            adf_s = float(adfuller(series_s, autolag="AIC")[1])
    except Exception:
        adf_s = np.nan
    try:
        if aligned.shape[0] >= 120 and aligned["B"].std(skipna=True) > 0 and aligned["S"].std(skipna=True) > 0:
            coint_p = float(coint(aligned["B"], aligned["S"])[1])
    except Exception:
        coint_p = np.nan
    return {"adf_bvocs_p": adf_b, "adf_soa_p": adf_s, "coint_p": coint_p}


def _format_stationarity_text(stats: Dict[str, float]) -> str:
    def _fmt_p(p: float) -> str:
        if not np.isfinite(p):
            return "NA"
        if p < 1e-4:
            return "<1e-4"
        return f"{p:.3f}"

    def _stars(p: float) -> str:
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    adf_b = stats.get("adf_bvocs_p", np.nan)
    adf_s = stats.get("adf_soa_p", np.nan)
    coint_p = stats.get("coint_p", np.nan)
    return (
        "Stationarity and co-integration (all sites): "
        f"ADF(BVOCs) p={_fmt_p(adf_b)}{_stars(adf_b)}, "
        f"ADF(SOA) p={_fmt_p(adf_s)}{_stars(adf_s)}, "
        f"Coint(B,S) p={_fmt_p(coint_p)}{_stars(coint_p)}; "
        "* p<0.05, ** p<0.01, *** p<0.001"
    )


def _compute_stationarity_from_windows(burst_map: Dict[str, List[Tuple[pd.DataFrame, str]]]) -> Dict[str, float]:
    """
    Use all selected windows (across both sites) to compute stationarity / cointegration, instead of a single burst window.
    """
    frames: List[pd.DataFrame] = []
    for cases in burst_map.values():
        for df_case, _ in cases:
            if df_case is None or df_case.empty:
                continue
            frames.append(df_case[["Time", "bvocs", "SOA"]])
    if not frames:
        return {"adf_bvocs_p": np.nan, "adf_soa_p": np.nan, "coint_p": np.nan}
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all = df_all.rename(columns={"Time": "Time"})
    df_all["Time"] = pd.to_datetime(df_all["Time"])
    return _compute_global_stationarity(df_all)


def _compute_stationarity_from_full_df(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute ADF / cointegration using all available hourly data per site (no window restriction),
    aggregating across sites with the minimum p (strongest evidence) to match per-site tables.
    """
    cols = ["Time", "bvocs", "SOA", "place"]
    if not set(cols).issubset(df.columns):
        return {"adf_bvocs_p": np.nan, "adf_soa_p": np.nan, "coint_p": np.nan}

    adf_b_list: List[float] = []
    adf_s_list: List[float] = []
    coint_list: List[float] = []

    for _, sub in df[cols].groupby("place"):
        sub = sub.copy()
        sub["Time"] = pd.to_datetime(sub["Time"])
        sub = sub.sort_values("Time")
        # Raw (unsmoothed) series
        series_b = pd.to_numeric(sub["bvocs"], errors="coerce")
        series_s = pd.to_numeric(sub["SOA"], errors="coerce")

        sb = series_b.dropna()
        ss = series_s.dropna()
        if sb.shape[0] >= 40:
            try:
                adf_b_list.append(float(adfuller(sb, autolag="AIC")[1]))
            except Exception:
                adf_b_list.append(np.nan)
        else:
            adf_b_list.append(np.nan)

        if ss.shape[0] >= 40:
            try:
                adf_s_list.append(float(adfuller(ss, autolag="AIC")[1]))
            except Exception:
                adf_s_list.append(np.nan)
        else:
            adf_s_list.append(np.nan)

        aligned = sub.dropna(subset=["bvocs", "SOA"]).sort_values("Time")
        if aligned.shape[0] >= 60 and aligned["bvocs"].std(skipna=True) > 0 and aligned["SOA"].std(skipna=True) > 0:
            try:
                coint_list.append(float(coint(aligned["bvocs"], aligned["SOA"])[1]))
            except Exception:
                coint_list.append(np.nan)
        else:
            coint_list.append(np.nan)

    adf_b = np.nanmin(np.array(adf_b_list)) if adf_b_list else np.nan
    adf_s = np.nanmin(np.array(adf_s_list)) if adf_s_list else np.nan
    coint_p = np.nanmin(np.array(coint_list)) if coint_list else np.nan
    return {"adf_bvocs_p": adf_b, "adf_soa_p": adf_s, "coint_p": coint_p}


def _add_cs_background(
    ax,
    times: pd.Series,
    cs: pd.Series,
    resample_rule: str = "10min",
    base_alpha: float = 0.12,
    alpha_span: float = 0.45,
    color: str = "#8f8f8f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Add CS background stripes with unified color and varying alpha."""
    t = pd.to_datetime(times)
    cs_series = pd.Series(pd.to_numeric(cs, errors="coerce").to_numpy(), index=t)
    cs_span = cs_series.resample(resample_rule).mean().dropna()
    if cs_span.shape[0] < 2:
        return
    vmin_use = float(np.nanmin(cs_span)) if vmin is None else float(vmin)
    vmax_use = float(np.nanmax(cs_span)) if vmax is None else float(vmax)
    if vmin_use == vmax_use:
        vmax_use = vmin_use + 1e-9
    norm = Normalize(vmin=vmin_use, vmax=vmax_use)
    for i in range(len(cs_span) - 1):
        alpha_raw = float(norm(float(cs_span.iloc[i])))
        alpha_norm = float(np.clip(alpha_raw, 0.0, 1.0))
        alpha = float(np.clip(base_alpha + alpha_span * alpha_norm, 0.0, 1.0))
        ax.axvspan(cs_span.index[i], cs_span.index[i + 1], color=color, alpha=alpha, zorder=-10)


def _plot_burst_case_stacked(
    ax_top,
    ax_bottom,
    window_df: pd.DataFrame,
    place: str,
    date_str: str,
    show_temp_label: bool,
    show_bvoc_label: bool,
    show_soa_label: bool,
    show_temp_ticks: bool,
    show_bvoc_ticks: bool,
    show_soa_ticks: bool,
    cs_vmin: float | None = None,
    cs_vmax: float | None = None,
    bv_ymax: float | None = None,
    soa_ymax: float | None = None,
) -> None:
    df = window_df.sort_values("Time").copy()
    if "Time_plot" in df.columns:
        t = pd.to_datetime(df["Time_plot"])
    else:
        t = pd.to_datetime(df["Time"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[df[["temperature_c", "bvocs", "SOA"]].notna().any(axis=1)].copy()
    if df.shape[0] < 10:
        ax_top.text(0.5, 0.5, "Not enough valid points", transform=ax_top.transAxes, ha="center", va="center")
        ax_bottom.axis("off")
        return

    t = t.loc[df.index]
    tmin = pd.to_datetime(t.min()).floor("30min")
    tmax = pd.to_datetime(t.max()).ceil("30min")

    _add_cs_background(
        ax_top,
        t,
        df["CS"],
        resample_rule="10min",
        base_alpha=0.13,
        alpha_span=0.42,
        color="#8f8f8f",
        vmin=cs_vmin,
        vmax=cs_vmax,
    )
    _add_cs_background(
        ax_bottom,
        t,
        df["CS"],
        resample_rule="10min",
        base_alpha=0.11,
        alpha_span=0.36,
        color="#8f8f8f",
        vmin=cs_vmin,
        vmax=cs_vmax,
    )

    temp = pd.to_numeric(df["temperature_c"], errors="coerce")
    bvocs = pd.to_numeric(df["bvocs"], errors="coerce")
    soa = pd.to_numeric(df["SOA"], errors="coerce")

    ax_top.plot(t, temp, color="#d62728", linewidth=2.6, zorder=3)
    if temp.notna().any():
        lo, hi = np.nanpercentile(temp.dropna(), [1, 99])
        ax_top.set_ylim(float(lo) - 0.5, float(hi) + 0.5)
    ax_top.set_ylabel("Temperature (C)" if show_temp_label else "", color="#d62728")
    ax_top.tick_params(axis="y", labelcolor="#d62728", labelleft=show_temp_ticks)

    ax_bv = ax_top.twinx()
    bv_smooth = pd.Series(bvocs).rolling(window=3, center=True, min_periods=3).mean()
    bv_plot = np.ma.masked_invalid(bv_smooth.to_numpy(dtype=float))
    ax_bv.plot(t, bv_plot, color="#2ca02c", linewidth=2.3, alpha=0.95, zorder=4)
    if bv_ymax is not None:
        ax_bv.set_ylim(0.0, max(float(bv_ymax), 1e-3))
    elif bv_smooth.notna().any():
        hi = float(np.nanpercentile(bv_smooth.dropna(), 99))
        ax_bv.set_ylim(0.0, max(hi * 1.10, 1e-3))
    ax_bv.set_ylabel("BVOCs (ug m$^{-3}$)" if show_bvoc_label else "", color="#2ca02c")
    ax_bv.tick_params(axis="y", labelcolor="#2ca02c", labelright=show_bvoc_ticks, labelsize=9, length=3)

    soa_smooth = pd.Series(soa).rolling(window=3, center=True, min_periods=3).mean()
    ax_bottom.plot(t, soa_smooth, color="#5b3ea4", linewidth=2.3, zorder=4)
    if soa_ymax is not None:
        ax_bottom.set_ylim(0.0, max(float(soa_ymax), 1e-3))
    elif soa_smooth.notna().any():
        hi = float(np.nanpercentile(soa_smooth.dropna(), 99))
        ax_bottom.set_ylim(0.0, max(hi * 1.10, 1e-3))
    ax_bottom.set_ylabel("SOA (ug m$^{-3}$)" if show_soa_label else "", color="#5b3ea4")
    ax_bottom.tick_params(axis="y", labelcolor="#5b3ea4", labelleft=show_soa_ticks, labelsize=9, length=3)

    if bvocs.notna().any():
        burst_idx = bvocs.astype(float).idxmax()
        burst_time = t.loc[burst_idx]
        for axis in (ax_top, ax_bottom):
            axis.axvline(burst_time, color="#6c6c6c", linestyle="--", linewidth=1.2, alpha=0.9, zorder=5)
        burst_label_time = burst_time + pd.Timedelta(minutes=2)
        burst_ha = "left"
        if burst_label_time > (tmax - pd.Timedelta(minutes=6)):
            burst_label_time = burst_time - pd.Timedelta(minutes=2)
            burst_ha = "right"
        ax_top.text(
            burst_label_time,
            0.92,
            "Precursor burst",
            transform=ax_top.get_xaxis_transform(),
            rotation=90,
            ha=burst_ha,
            va="top",
            fontsize=13,
            color="#4d4d4d",
        )
        # Mark BVOC peak and SOA response to highlight co-movement
        try:
            pos_bv = df.index.get_loc(burst_idx)
            ax_bv.scatter(
                burst_time,
                bv_plot[pos_bv],
                color="#2ca02c",
                marker="^",
                s=26,
                edgecolor="#ffffff",
                linewidth=0.9,
                zorder=6,
            )
        except Exception:
            pass
        if soa_smooth.notna().any():
            soa_idx = soa_smooth.astype(float).idxmax()
            try:
                pos_soa = df.index.get_loc(soa_idx)
                soa_time = t.iloc[pos_soa]
                soa_val = soa_smooth.iloc[pos_soa]
                ax_bottom.scatter(
                    soa_time,
                    soa_val,
                    color="#5b3ea4",
                    marker="s",
                    s=24,
                    edgecolor="#ffffff",
                    linewidth=0.9,
                    zorder=6,
                )
                ax_bottom.annotate(
                    "SOA response",
                    xy=(soa_time, soa_val),
                    xytext=(soa_time, soa_val + (soa_smooth.max() * 0.08 if soa_smooth.max() > 0 else 0.3)),
                    arrowprops=dict(arrowstyle="->", color="#5b3ea4", linewidth=1.2),
                    fontsize=11,
                    color="#5b3ea4",
                )
            except Exception:
                pass

    ax_top.set_title(
        f"Process-scale burst case ({place}, {date_str})",
        fontweight="bold",
        fontsize=15,
        loc="left",
    )
    ax_top.set_xlim(tmin, tmax)
    ax_bottom.set_xlim(tmin, tmax)
    _apply_axis_style(ax_top)
    _apply_axis_style(ax_bottom)
    ax_top.grid(alpha=0.18, linestyle=":", linewidth=0.7)
    ax_bottom.grid(alpha=0.18, linestyle=":", linewidth=0.7)
    _apply_burst_time_ticks(ax_bottom, rotation=15)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_bottom.set_xlabel("Time (HH:MM)")


def _stitch_burst_cases(
    cases: List[Tuple[pd.DataFrame, str]],
    gap_minutes: int = 15,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """
    Stitches multiple burst windows into a single continuous time axis for plotting.
    The stitched axis is pseudo-time (not real clock time) but preserves within-window spacing.
    """
    base = pd.Timestamp("2000-01-01 00:00:00")
    stitched_parts: List[pd.DataFrame] = []
    segments: List[Dict[str, object]] = []
    offset = pd.Timedelta(minutes=0)
    for seg_idx, (df_case, date_str) in enumerate(cases):
        df_seg = df_case.copy().sort_values("Time").reset_index(drop=True)
        if df_seg.empty:
            continue
        t_raw = pd.to_datetime(df_seg["Time"])
        t0 = pd.to_datetime(t_raw.min())
        t_rel = t_raw - t0
        t_stitched = base + offset + t_rel
        df_seg["Time_stitched"] = t_stitched
        df_seg["segment_idx"] = seg_idx
        stitched_parts.append(df_seg)
        seg_start = base + offset
        seg_end = base + offset + (pd.to_datetime(t_raw.max()) - t0)
        orig_start = pd.to_datetime(t_raw.min())
        orig_end = pd.to_datetime(t_raw.max())
        segments.append(
            {
                "idx": seg_idx,
                "date": date_str,
                "start": seg_start,
                "end": seg_end,
                "orig_start": orig_start,
                "orig_end": orig_end,
            }
        )
        offset = seg_end - base + pd.Timedelta(minutes=gap_minutes)
    if not stitched_parts:
        return pd.DataFrame(), []
    df_all = pd.concat(stitched_parts, axis=0, ignore_index=True)
    return df_all, segments


def _plot_burst_case_stitched(
    ax_header,
    ax_top,
    ax_bottom,
    ax_stats,
    cases: List[Tuple[pd.DataFrame, str]],
    place: str,
    cs_vmin: float | None,
    cs_vmax: float | None,
    bv_ymax: float | None,
    soa_ymax: float | None,
) -> None:
    """
    Plot multiple burst windows stitched on a continuous axis.
    """
    # Use a small gap to keep the stitched axis compact while still showing segment boundaries.
    df_all, segments = _stitch_burst_cases(cases, gap_minutes=2)
    if df_all.empty:
        ax_top.text(0.5, 0.5, "No burst windows", transform=ax_top.transAxes, ha="center", va="center")
        ax_bottom.axis("off")
        if ax_header is not None:
            ax_header.axis("off")
        if ax_stats is not None:
            ax_stats.axis("off")
        return

    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    t = pd.to_datetime(df_all["Time_stitched"])
    total_minutes = float((t.max() - t.min()).total_seconds() / 60.0) if len(t) > 1 else 0.0
    # Downsample CS background for long stitched records to avoid dense vertical striping.
    cs_rule = "10min"
    if total_minutes > 12 * 60:
        cs_rule = "30min"
    if total_minutes > 24 * 60:
        cs_rule = "60min"
    if total_minutes > 48 * 60:
        cs_rule = "120min"

    _add_cs_background(
        ax_top,
        t,
        df_all["CS"],
        resample_rule=cs_rule,
        base_alpha=0.20,
        alpha_span=0.42,
        color="#8f8f8f",
        vmin=cs_vmin,
        vmax=cs_vmax,
    )
    _add_cs_background(
        ax_bottom,
        t,
        df_all["CS"],
        resample_rule=cs_rule,
        base_alpha=0.18,
        alpha_span=0.38,
        color="#8f8f8f",
        vmin=cs_vmin,
        vmax=cs_vmax,
    )

    temp = pd.to_numeric(df_all["temperature_c"], errors="coerce")
    bvocs = pd.to_numeric(df_all["bvocs"], errors="coerce")
    soa = pd.to_numeric(df_all["SOA"], errors="coerce")

    # Set shared axis ranges using pooled percentiles (do not plot across segments to avoid false continuity).
    if temp.notna().any():
        lo, hi = np.nanpercentile(temp.dropna(), [1, 99])
        ax_top.set_ylim(float(lo) - 0.5, float(hi) + 0.5)
    # Use "deg C" symbol with Helvetica-compatible glyphs.
    # The dedicated Celsius symbol (U+2103) is not available in the bundled Helvetica Neue LT Pro OTF files.
    ax_top.set_ylabel("Temperature\n(deg C)", color="#d62728", labelpad=4)
    ax_top.yaxis.set_label_coords(-0.06, 0.52)
    ax_top.tick_params(axis="y", labelcolor="#d62728", labelsize=9, pad=2)
    ax_top.yaxis.label.set_size(11)

    ax_bv = ax_top.twinx()
    bv_smooth = pd.Series(bvocs).rolling(window=3, center=True, min_periods=3).mean()
    bv_plot = np.ma.masked_invalid(bv_smooth.to_numpy(dtype=float))
    if bv_ymax is not None:
        ax_bv.set_ylim(0.0, max(float(bv_ymax), 1e-3))
    elif bv_smooth.notna().any():
        hi = float(np.nanpercentile(bv_smooth.dropna(), 99))
        ax_bv.set_ylim(0.0, max(hi * 1.10, 1e-3))
    ax_bv.set_ylabel("BVOCs\n(ug m$^{-3}$)", color="#2ca02c", labelpad=3)
    ax_bv.yaxis.set_label_coords(1.053, 0.52)
    ax_bv.tick_params(axis="y", labelcolor="#2ca02c", labelsize=9, pad=1)
    ax_bv.yaxis.label.set_size(11)

    soa_smooth = pd.Series(soa).rolling(window=3, center=True, min_periods=3).mean()
    if soa_ymax is not None:
        ax_bottom.set_ylim(0.0, max(float(soa_ymax), 1e-3))
    elif soa_smooth.notna().any():
        hi = float(np.nanpercentile(soa_smooth.dropna(), 99))
        ax_bottom.set_ylim(0.0, max(hi * 1.10, 1e-3))
    ax_bottom.set_ylabel("SOA\n(ug m$^{-3}$)", color="#5b3ea4", labelpad=4)
    ax_bottom.yaxis.set_label_coords(-0.06, 0.52)
    ax_bottom.tick_params(axis="y", labelcolor="#5b3ea4", labelsize=9, pad=2)
    ax_bottom.yaxis.label.set_size(11)

    # Segment separators, burst emphasis, and per-segment co-movement metrics
    gap_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(len(segments) - 1):
        gap_spans.append((segments[i]["end"], segments[i + 1]["start"]))

    # Date labels: one per unique original date, rendered in a header axis (outside the data axes).
    # Group segments by real date for header boxes.
    date_groups: Dict[str, List[Dict[str, object]]] = {}
    date_labels: Dict[str, str] = {}
    for seg in segments:
        t0 = pd.to_datetime(seg.get("orig_start", pd.NaT))
        if pd.isna(t0):
            continue
        key = t0.strftime("%Y-%m-%d")
        date_groups.setdefault(key, []).append(seg)
        date_labels[key] = t0.strftime("%y-%m-%d")
    for start_gap, end_gap in gap_spans:
        if end_gap <= start_gap:
            continue
        for axis in (ax_top, ax_bottom):
            # Make stitched gaps visually explicit (prevents "same line" impression).
            axis.axvspan(start_gap, end_gap, color="#ffffff", alpha=0.85, zorder=0)

    if ax_header is not None:
        ax_header.axis("off")
        ax_header.set_xlim(t.min(), t.max())
        ax_header.set_ylim(0.0, 1.0)
        ax_header.text(
            0.0,
            1,
            f"Process-scale burst cases stitched ({place})",
            transform=ax_header.transAxes,
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#1a1a1a",
            clip_on=False,
        )

    if ax_header is not None and date_groups:
        total_span = (t.max() - t.min()) if len(t) > 1 else pd.Timedelta(minutes=1)
        trans_h = blended_transform_factory(ax_header.transData, ax_header.transAxes)
        ordered = sorted(date_groups.items(), key=lambda kv: pd.to_datetime(kv[1][0]["start"]))
        for date_key, segs_day in ordered:
            starts = [pd.to_datetime(s["start"]) for s in segs_day]
            ends = [pd.to_datetime(s["end"]) for s in segs_day]
            x0 = min(starts)
            x1 = max(ends)
            frac = float((x1 - x0) / total_span) if total_span.total_seconds() > 0 else 0.0
            if frac < 0.03:
                continue
            ax_header.text(
                x0 + (x1 - x0) * 0.5,
                0.40,
                date_labels.get(date_key, date_key),
                transform=trans_h,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#4d4d4d",
                clip_on=False,
                bbox=dict(facecolor="#f2f2f2", edgecolor="none", boxstyle="square,pad=0.20", alpha=0.88),
                zorder=12,
            )

    for seg in segments:
        x0 = seg["start"]
        x1 = seg["end"]
        for axis in (ax_top, ax_bottom):
            axis.axvline(x0, color="#b0b0b0", linewidth=0.8, alpha=0.22, linestyle="-", zorder=9)
        seg_mask = df_all["segment_idx"] == int(seg["idx"])
        df_seg = df_all.loc[seg_mask].copy()
        if df_seg.empty:
            continue
        t_seg = pd.to_datetime(df_seg["Time_stitched"])
        bv_seg = pd.to_numeric(df_seg["bvocs"], errors="coerce")
        soa_seg = pd.to_numeric(df_seg["SOA"], errors="coerce")
        temp_seg = pd.to_numeric(df_seg["temperature_c"], errors="coerce")

        # Plot each segment separately to avoid connecting different days with a single line.
        # This keeps the stitched view honest while still allowing faint connectors below.
        ax_top.plot(t_seg, temp_seg, color="#d62728", linewidth=1.4, alpha=0.33, zorder=1)

        # Emphasize multiple burst intervals using local peaks above a percentile threshold.
        bv_seg_smooth = pd.Series(bv_seg).rolling(window=3, center=True, min_periods=3).mean()
        soa_seg_smooth = pd.Series(soa_seg).rolling(window=3, center=True, min_periods=3).mean()
        ax_bv.plot(t_seg, bv_seg_smooth, color="#2ca02c", linewidth=1.4, alpha=0.33, zorder=1)
        ax_bottom.plot(t_seg, soa_seg_smooth, color="#5b3ea4", linewidth=1.4, alpha=0.33, zorder=1)
        peak_times: List[pd.Timestamp] = []
        if bv_seg_smooth.notna().any():
            bv_pos = bv_seg_smooth.clip(lower=0.0)
            thresh = float(np.nanpercentile(bv_pos.dropna(), 92)) if bv_pos.dropna().shape[0] > 10 else float(bv_pos.max(skipna=True))
            if not np.isfinite(thresh):
                thresh = float(bv_pos.max(skipna=True))
            w = 5
            candidates = []
            values = bv_pos.to_numpy(dtype=float)
            for k in range(w, len(values) - w):
                v = values[k]
                if not np.isfinite(v) or v < thresh:
                    continue
                left = values[k - w : k]
                right = values[k + 1 : k + 1 + w]
                if np.all(v >= left) and np.all(v >= right):
                    candidates.append((k, v))
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            # Enforce minimum separation between peaks.
            selected = []
            min_sep = 30
            for k, v in candidates:
                if any(abs(k - kk) < min_sep for kk, _ in selected):
                    continue
                selected.append((k, v))
                if len(selected) >= 2:
                    break
            for k, _ in sorted(selected, key=lambda x: x[0]):
                peak_times.append(pd.to_datetime(t_seg.iloc[k]))

        for t_bv in peak_times:
            burst_left = t_bv - pd.Timedelta(minutes=15)
            burst_right = t_bv + pd.Timedelta(minutes=20)
            burst_mask = (t_seg >= burst_left) & (t_seg <= burst_right)
            ax_top.plot(t_seg[burst_mask], temp_seg[burst_mask], color="#d62728", linewidth=2.6, alpha=0.95, zorder=4)
            ax_bv.plot(t_seg[burst_mask], bv_seg_smooth[burst_mask], color="#2ca02c", linewidth=2.3, alpha=0.95, zorder=5)
            ax_bottom.plot(t_seg[burst_mask], soa_seg_smooth[burst_mask], color="#5b3ea4", linewidth=2.3, alpha=0.95, zorder=5)
            # Peak marker and SOA response marker (max within +5 to +40 min).
            v_bv = np.nan
            if bv_seg_smooth.notna().any():
                try:
                    pos_peak = int(np.argmin(np.abs((t_seg - t_bv).dt.total_seconds().to_numpy(dtype=float))))
                    v_bv = float(bv_seg_smooth.iloc[pos_peak])
                except Exception:
                    v_bv = np.nan
            if np.isfinite(v_bv):
                ax_bv.scatter(
                    t_bv,
                    v_bv,
                    color="#2ca02c",
                    marker="^",
                    s=26,
                    edgecolor="#ffffff",
                    linewidth=0.9,
                    zorder=6,
                )
            resp_mask = (t_seg >= t_bv + pd.Timedelta(minutes=5)) & (t_seg <= t_bv + pd.Timedelta(minutes=40))
            if resp_mask.any():
                resp_vals = soa_seg_smooth[resp_mask]
                if resp_vals.notna().any():
                    k_max = int(resp_vals.astype(float).idxmax())
                    try:
                        pos_resp = int(df_seg.index.get_indexer([k_max])[0])
                        t_resp = t_seg.iloc[pos_resp]
                        v_resp = float(soa_seg_smooth.iloc[pos_resp])
                        ax_bottom.scatter(
                            t_resp,
                            v_resp,
                            color="#5b3ea4",
                            marker="s",
                            s=24,
                            edgecolor="#ffffff",
                            linewidth=0.9,
                            zorder=6,
                        )
                    except Exception:
                        pass

        # Keep the stitched panels minimal: no lag or correlation annotations.

    ax_top.set_title("")
    _apply_axis_style(ax_top)
    _apply_axis_style(ax_bottom)
    ax_top.grid(alpha=0.16, linestyle=":", linewidth=0.7)
    ax_bottom.grid(alpha=0.16, linestyle=":", linewidth=0.7)
    _apply_stitched_datetime_ticks(ax_bottom, segments=segments, rotation=0.0, max_labels=4)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_bottom.tick_params(axis="x", pad=8)
    ax_bottom.xaxis.labelpad = 12
    ax_bottom.set_xlabel("")

    # Add ADF and cointegration summaries on the right of the bottom row (outside the plotting area).
    if ax_stats is not None:
        ax_stats.axis("off")
        series_b = pd.to_numeric(df_all["bvocs"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        series_s = pd.to_numeric(df_all["SOA"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        aligned = pd.concat([series_b.rename("B"), series_s.rename("S")], axis=1).dropna()

        def _p_stars(p: float) -> str:
            if not np.isfinite(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        adf_b = np.nan
        adf_s = np.nan
        coint_p = np.nan
        try:
            if series_b.dropna().shape[0] >= 40:
                adf_b = float(adfuller(series_b.dropna(), autolag="AIC")[1])
        except Exception:
            adf_b = np.nan
        try:
            if series_s.dropna().shape[0] >= 40:
                adf_s = float(adfuller(series_s.dropna(), autolag="AIC")[1])
        except Exception:
            adf_s = np.nan
        try:
            if aligned.shape[0] >= 60 and aligned["B"].std(skipna=True) > 0 and aligned["S"].std(skipna=True) > 0:
                coint_p = float(coint(aligned["B"], aligned["S"])[1])
        except Exception:
            coint_p = np.nan

        def _fmt_p(p: float) -> str:
            if not np.isfinite(p):
                return "NA"
            if p < 1e-4:
                return "<1e-4"
            return f"{p:.3f}"

        # text = (
        #     "Stationarity and co-integration\n"
        #     f"ADF(BVOCs): p={_fmt_p(adf_b)}{_p_stars(adf_b)}\n"
        #     f"ADF(SOA):   p={_fmt_p(adf_s)}{_p_stars(adf_s)}\n"
        #     f"Coint(B,S): p={_fmt_p(coint_p)}{_p_stars(coint_p)}\n"
        #     "* p<0.05, ** p<0.01, *** p<0.001"
        # )
        ax_stats.text(
            0.0,
            1.0,
            text,
            transform=ax_stats.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#2c2c2c",
            bbox=dict(facecolor="#f2f2f2", edgecolor="none", boxstyle="square,pad=0.30", alpha=0.95),
        )

    # Add faint connecting lines across gaps to reduce visual discontinuity.
    for i in range(len(segments) - 1):
        left_seg = df_all[df_all["segment_idx"] == int(segments[i]["idx"])].copy()
        right_seg = df_all[df_all["segment_idx"] == int(segments[i + 1]["idx"])].copy()
        if left_seg.empty or right_seg.empty:
            continue
        left_last = left_seg.iloc[-1]
        right_first = right_seg.iloc[0]
        t_left = pd.to_datetime(left_last["Time_stitched"])
        t_right = pd.to_datetime(right_first["Time_stitched"])
        v_t0 = pd.to_numeric(left_last.get("temperature_c"), errors="coerce")
        v_t1 = pd.to_numeric(right_first.get("temperature_c"), errors="coerce")
        if np.isfinite(v_t0) and np.isfinite(v_t1):
            t_gap = pd.date_range(t_left, t_right, periods=12)
            y_gap = np.linspace(float(v_t0), float(v_t1), num=len(t_gap))
            ax_top.plot(
                t_gap,
                y_gap,
                color="#d62728",
                linewidth=0.9,
                alpha=0.06,
                linestyle="--",
                zorder=1,
            )
        v_s0 = pd.to_numeric(left_last.get("SOA"), errors="coerce")
        v_s1 = pd.to_numeric(right_first.get("SOA"), errors="coerce")
        if np.isfinite(v_s0) and np.isfinite(v_s1):
            t_gap = pd.date_range(t_left, t_right, periods=12)
            y_gap = np.linspace(float(v_s0), float(v_s1), num=len(t_gap))
            ax_bottom.plot(
                t_gap,
                y_gap,
                color="#5b3ea4",
                linewidth=0.9,
                alpha=0.06,
                linestyle="--",
                zorder=1,
            )
        v_b0 = pd.to_numeric(left_last.get("bvocs"), errors="coerce")
        v_b1 = pd.to_numeric(right_first.get("bvocs"), errors="coerce")
        if np.isfinite(v_b0) and np.isfinite(v_b1):
            t_gap = pd.date_range(t_left, t_right, periods=12)
            y_gap = np.linspace(float(v_b0), float(v_b1), num=len(t_gap))
            ax_bv.plot(
                t_gap,
                y_gap,
                color="#2ca02c",
                linewidth=0.9,
                alpha=0.06,
                linestyle="--",
                zorder=1,
            )


def _add_day_background(ax, times: pd.Series, alpha: float = 0.08, color: str = "#cccccc") -> None:
    """Add light day-level background stripes to highlight day breaks."""
    times = pd.to_datetime(times)
    if times.empty:
        return
    dates = times.dt.normalize().drop_duplicates().sort_values()
    for i, date_val in enumerate(dates):
        start = date_val
        end = date_val + pd.Timedelta(days=1)
        if i % 2 == 0:
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)


def _compute_cs(df: pd.DataFrame, cfg: WorkflowConfig) -> pd.Series:
    """
    References: Fuchs and Sutugin (1971) for condensation sink.
    Equation: CS = sum_i 4*pi*D_v*r_i*F(Kn_i)*N_i where Kn_i is Knudsen number and F is slip correction.
    Parameters: D_v vapor diffusivity (m^2 s^-1), r_i particle radius (m), Kn_i dimensionless, N_i number concentration (m^-3).
    """
    cs = mf.compute_cs(df, cfg)
    return cs


def _add_deltas(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df_out = df.copy()
    for col in cols:
        delta_name = f"delta_{col}"
        df_out[delta_name] = (
            df_out.groupby("place")[col]
            .diff()
            .fillna(0.0)
        )
    return df_out


def _prepare_diurnal_df(df: pd.DataFrame, cfg: WorkflowConfig) -> pd.DataFrame:
    """
    Align with the existing workflow aggregation (aggregate_by_hour_min) to avoid gaps on the time axis.
    """
    df = mf.aggregate_by_hour_min(df)
    df["CS"] = _compute_cs(df, cfg)
    df = _add_deltas(df, ["SOA", "CS"])
    return df.reset_index()


def _load_process_scale_source_data(cfg: WorkflowConfig) -> pd.DataFrame:
    """
    Load high-frequency source data for process-scale burst visualization.

    This loader intentionally uses the raw multi-site inputs (chemistry joblib + size distribution csv)
    instead of the pre-aggregated intermediate/step01_clean.parquet, so that we can select multiple
    burst windows per available day without collapsing variability.
    """
    resample_rule = getattr(cfg, "resample_rule", "1min")
    soa_bins = ["0.25um", "0.28um", "0.30um"]
    required_number_cols_prefix = "C"

    frames: List[pd.DataFrame] = []
    for place, chem_path in cfg.chemistry_joblib_sites.items():
        size_path = cfg.size_distribution_csv_sites.get(place)
        if size_path is None:
            continue

        chem_path = Path(chem_path)
        if not chem_path.is_absolute():
            chem_path = (BUNDLE_ROOT.parent / chem_path).resolve()
        chem_obj = joblib.load(str(chem_path))
        if isinstance(chem_obj, dict):
            chem_frames = [v for v in chem_obj.values() if isinstance(v, pd.DataFrame)]
            df_chem = pd.concat(chem_frames, axis=0) if chem_frames else pd.DataFrame()
        elif isinstance(chem_obj, pd.DataFrame):
            df_chem = chem_obj.copy()
        else:
            df_chem = pd.DataFrame()
        if df_chem.empty:
            continue
        if not isinstance(df_chem.index, pd.DatetimeIndex):
            if "Time" in df_chem.columns:
                df_chem["Time"] = pd.to_datetime(df_chem["Time"])
                df_chem = df_chem.set_index("Time")
            else:
                continue
        df_chem = df_chem.sort_index()

        temp_col = cfg.meteorology_columns.get("temperature_c", "temperature_c")
        iso_col = next((c for c in cfg.bvoc_columns if c in df_chem.columns), None)
        if iso_col is None and "Isoprene" in df_chem.columns:
            iso_col = "Isoprene"
        if temp_col not in df_chem.columns or iso_col is None:
            continue
        df_chem = df_chem[[temp_col, iso_col]].copy()
        df_chem = df_chem.rename(columns={temp_col: "temperature_c", iso_col: "bvocs"})
        df_chem["temperature_c"] = pd.to_numeric(df_chem["temperature_c"], errors="coerce")
        df_chem["bvocs"] = pd.to_numeric(df_chem["bvocs"], errors="coerce").clip(lower=0.0)
        df_chem = df_chem.resample(resample_rule).mean()

        df_size = pd.read_csv(str(size_path))
        time_col = "Datetime" if "Datetime" in df_size.columns else ("Time" if "Time" in df_size.columns else None)
        if time_col is None:
            continue
        df_size[time_col] = pd.to_datetime(df_size[time_col])
        df_size = df_size.set_index(time_col).sort_index()
        number_cols = [c for c in df_size.columns if str(c).startswith(required_number_cols_prefix) and str(c).endswith("um")]
        keep_cols = [c for c in (soa_bins + number_cols) if c in df_size.columns]
        if not keep_cols:
            continue
        df_size = df_size[keep_cols].apply(pd.to_numeric, errors="coerce")
        df_size = df_size.resample(resample_rule).mean()

        # Use the size-distribution timeline as the backbone so CS can be computed where number bins exist.
        merged = df_size.join(df_chem, how="left")
        merged["place"] = place
        if all(c in merged.columns for c in soa_bins):
            merged["SOA"] = merged[soa_bins].sum(axis=1, min_count=1).clip(lower=0.0)
        else:
            merged["SOA"] = np.nan
        try:
            merged["CS"] = mf.compute_cs(merged, cfg)
        except Exception:
            merged["CS"] = np.nan
        frames.append(merged.reset_index().rename(columns={time_col: "Time"}))

    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all["Time"] = pd.to_datetime(df_all["Time"])
    df_all = df_all.sort_values(["place", "Time"]).reset_index(drop=True)
    return df_all


def _fit_xi_kloss(df: pd.DataFrame) -> Tuple[float, float]:
    """
    References: mechanism reconstruction Section 2.3.
    Equation: dSOA/dt = xi * CS * C - k_loss * SOA.
    Parameters: xi condensation efficiency (m^3 s^-1), k_loss first-order loss (s^-1), CS condensation sink (s^-1), C condensable vapor (ug m^-3), SOA (ug m^-3).
    Here we use BVOCs as proxy for C in regression to estimate xi and k_loss.
    """
    y = df["dSOA_dt"].to_numpy()
    X1 = (df["CS"] * df["bvocs"]).to_numpy()
    X2 = df["SOA"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(X1) & np.isfinite(X2) & (df["CS"] > 0)
    if mask.sum() < 10:
        return 1e-6, 0.0
    A = np.vstack([X1[mask], -X2[mask], np.ones_like(X1[mask])]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
    xi_hat = max(coeffs[0], 1e-6)
    k_loss_hat = max(coeffs[1], 0.0)
    return xi_hat, k_loss_hat


def _smooth(series: pd.Series, window: int = 6) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _run_stationarity_tests(df: pd.DataFrame, cols: Iterable[str]) -> List[TestResult]:
    results: List[TestResult] = []
    for place, sub in df.groupby("place"):
        for col in cols:
            series = pd.to_numeric(sub[col], errors="coerce").dropna()
            if series.shape[0] < 20:
                results.append(TestResult(place, col, np.nan, np.nan))
                continue
            try:
                adf_p = adfuller(series, autolag="AIC")[1]
            except Exception:
                adf_p = np.nan
            try:
                kpss_p = kpss(series, regression="c", nlags="auto")[1]
            except Exception:
                kpss_p = np.nan
            results.append(TestResult(place, col, adf_p, kpss_p))
    return results


def _run_cointegration(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for place, sub in df.groupby("place"):
        for a, b in pairs:
            series_a = pd.to_numeric(sub[a], errors="coerce").dropna()
            series_b = pd.to_numeric(sub[b], errors="coerce").dropna()
            aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
            if aligned.shape[0] < 30:
                out[(place, a, b)] = np.nan
                continue
            try:
                coint_p = coint(aligned[a], aligned[b])[1]
            except Exception:
                coint_p = np.nan
            out[(place, a, b)] = coint_p
    return out


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3f}"


def _format_p_star(p: float, alpha: float = 0.05) -> str:
    """Return p text with significance star if below alpha."""
    text = _format_p(p)
    if not np.isnan(p) and p < alpha:
        return f"{text}*"
    return text


def _select_burst_window(
    df_raw: pd.DataFrame,
    cfg: WorkflowConfig,
    window_hours: float = 4.0,
    start_hour: int = 10,
    end_hour: int = 15,
    preferred_dates: Dict[str, List[str]] | None = None,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Pick a midday burst window where temperature, BVOCs, and SOA show a clear coupled excursion.
    The scoring favors (1) large temperature and BVOC ranges, (2) positive T-BVOC coupling, and
    (3) BVOC leading SOA with a short positive lag.
    """
    df = df_raw.copy()
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    else:
        df = df.reset_index().rename(columns={"index": "Time"})
        df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")
    if "CS" not in df.columns:
        df["CS"] = _compute_cs(df, cfg)
    df = df.dropna(subset=["bvocs", "temperature_c", "SOA", "CS"])
    window_minutes = int(window_hours * 60)
    best: Dict[str, object] | None = None
    for place, sub_place in df.groupby("place"):
        sub_place = sub_place.infer_objects(copy=False).set_index("Time")
        numeric_cols = sub_place.select_dtypes(include=[np.number]).columns
        sub_place_numeric = sub_place[numeric_cols].resample("1min").mean()
        sub_place_numeric = sub_place_numeric.interpolate(limit=5, limit_direction="both")
        for col in ["bvocs", "SOA", "CS"]:
            if col in sub_place_numeric.columns:
                sub_place_numeric[col] = sub_place_numeric[col].clip(lower=0.0)
        sub_place = sub_place_numeric.copy()
        sub_place["place"] = place
        sub_place = sub_place.reset_index().rename(columns={"index": "Time"})
        sub_place["date"] = sub_place["Time"].dt.date
        sub_mid = sub_place[
            (sub_place["Time"].dt.hour >= start_hour)
            & (sub_place["Time"].dt.hour <= end_hour)
        ]
        date_groups = list(sub_mid.groupby("date"))
        if preferred_dates and place in preferred_dates:
            pref_list = [pd.to_datetime(x).date() for x in preferred_dates[place]]
            preferred_set = set(pref_list)
            preferred_groups = [(d, g) for d, g in date_groups if d in preferred_set]
            if preferred_groups:
                # Preserve the order in pref_list to try the desired days first.
                date_groups = sorted(preferred_groups, key=lambda x: pref_list.index(x[0]) if x[0] in pref_list else 999)
        for date_value, day_df in date_groups:
            day_sorted = day_df.sort_values("Time")
            if day_sorted.shape[0] < max(80, window_minutes // 2):
                continue
            day_base = pd.to_datetime(str(date_value))
            start_base = day_base + pd.Timedelta(hours=start_hour)
            end_base = day_base + pd.Timedelta(hours=end_hour) - pd.Timedelta(minutes=window_minutes)
            if end_base <= start_base:
                continue
            start_range = pd.date_range(start_base, end_base, freq="30min")
            for start_time in start_range:
                end_time = start_time + pd.Timedelta(minutes=window_minutes)
                window_df = day_sorted[
                    (day_sorted["Time"] >= start_time) & (day_sorted["Time"] <= end_time)
                ]
                preferred_flag = bool(preferred_dates and place in preferred_dates and date_value in [pd.to_datetime(x).date() for x in preferred_dates[place]])
                min_count_ratio = 0.35 if preferred_flag else 0.4
                if window_df.shape[0] < window_minutes * min_count_ratio:
                    continue
                coverage = window_df["bvocs"].count() / max(window_df.shape[0], 1)
                cov_thresh = 0.5 if preferred_flag else 0.6
                if coverage < cov_thresh:
                    continue
                edge_minutes = 30
                first_edge = window_df[window_df["Time"] < start_time + pd.Timedelta(minutes=edge_minutes)]
                last_edge = window_df[window_df["Time"] > end_time - pd.Timedelta(minutes=edge_minutes)]
                if first_edge.empty or last_edge.empty:
                    continue
                first_cov = first_edge["bvocs"].count() / max(first_edge.shape[0], 1)
                last_cov = last_edge["bvocs"].count() / max(last_edge.shape[0], 1)
                edge_thresh = 0.3 if preferred_flag else 0.4
                if min(first_cov, last_cov) < edge_thresh and not preferred_flag:
                    continue
                if preferred_flag and coverage >= cov_thresh:
                    pass
                elif min(first_cov, last_cov) < edge_thresh:
                    continue
                temp_range = float(window_df["temperature_c"].max() - window_df["temperature_c"].min())
                temp_std = float(window_df["temperature_c"].std(skipna=True))
                bvocs_range = float(window_df["bvocs"].max() - window_df["bvocs"].min())
                bvocs_std = float(window_df["bvocs"].std(skipna=True))
                bvocs_diff_peak = float(window_df["bvocs"].diff().abs().rolling(5, min_periods=3).max().max(skipna=True))
                soa_span = float(window_df["SOA"].max() - window_df["SOA"].min())
                corr_tb = 0.0
                temp_bv = (
                    window_df[["temperature_c", "bvocs"]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if (
                    temp_bv["temperature_c"].std(skipna=True) > 0
                    and temp_bv["bvocs"].std(skipna=True) > 0
                    and temp_bv.shape[0] > 2
                ):
                    with np.errstate(invalid="ignore"):
                        corr_tb = float(
                            np.corrcoef(temp_bv["temperature_c"], temp_bv["bvocs"])[0, 1]
                        )
                    if not np.isfinite(corr_tb):
                        corr_tb = 0.0
                soa_shift = window_df["SOA"].shift(-20)
                pair_lag = pd.concat(
                    [window_df["bvocs"], soa_shift.rename("SOA")], axis=1
                ).dropna()
                pair_lag = pair_lag.replace([np.inf, -np.inf], np.nan).dropna()
                soa_lag_corr = 0.0
                if (
                    pair_lag["bvocs"].std(skipna=True) > 0
                    and pair_lag["SOA"].std(skipna=True) > 0
                    and pair_lag.shape[0] > 2
                ):
                    with np.errstate(invalid="ignore"):
                        soa_lag_corr = float(
                            np.corrcoef(pair_lag["bvocs"], pair_lag["SOA"])[0, 1]
                        )
                    if not np.isfinite(soa_lag_corr):
                        soa_lag_corr = 0.0
                diff_min = 0.8
                if preferred_dates and place in preferred_dates:
                    pref_list = [pd.to_datetime(x).date() for x in preferred_dates[place]]
                    if date_value in pref_list:
                        diff_min = 0.4
                if bvocs_range < 0.5 or bvocs_diff_peak < diff_min:
                    continue
                score_temp = 0.5 * temp_range + 0.5 * temp_std
                score_bv = 0.45 * bvocs_range + 0.35 * bvocs_std + 0.20 * bvocs_diff_peak
                score = (score_temp + 1e-3) * (score_bv + 1e-3)
                score *= 1.0 + max(0.0, corr_tb if not np.isnan(corr_tb) else 0.0)
                score *= 1.0 + max(0.0, soa_lag_corr if not np.isnan(soa_lag_corr) else 0.0)
                score *= 1.0 + max(0.0, soa_span)
                if preferred_dates and place in preferred_dates:
                    pref_list = [pd.to_datetime(x).date() for x in preferred_dates[place]]
                    if date_value in pref_list:
                        rank = pref_list.index(date_value)
                        if rank == 0:
                            score *= 5.0
                        score *= 1.0 + 0.15 * float(len(pref_list) - rank)
                if best is None or score > best["score"]:
                    best = {
                        "window": window_df.copy(),
                        "place": place,
                        "date": str(date_value),
                        "score": score,
                        "start": start_time,
                        "end": end_time,
                    }
    if best is None:
        fallback = df.sort_values("Time")
        fallback = fallback[
            (fallback["Time"].dt.hour >= start_hour) & (fallback["Time"].dt.hour <= end_hour)
        ]
        if fallback.empty:
            return fallback.reset_index(drop=True), "unknown", "unknown"
        fallback_window = fallback.head(max(120, window_minutes)).reset_index(drop=True)
        start_time = pd.to_datetime(fallback_window["Time"].min())
        end_time = pd.to_datetime(fallback_window["Time"].max())
        fallback_window["Time_plot"] = pd.to_datetime("2000-01-01") + (
            fallback_window["Time"] - fallback_window["Time"].dt.normalize()
        )
        fallback_window["WindowStart"] = start_time
        fallback_window["WindowEnd"] = end_time
        fallback_window["WindowStart_plot"] = pd.to_datetime("2000-01-01") + (
            start_time - start_time.normalize()
        )
        fallback_window["WindowEnd_plot"] = pd.to_datetime("2000-01-01") + (
            end_time - end_time.normalize()
        )
        return fallback_window, str(fallback_window["place"].iloc[0]), str(fallback_window["Time"].dt.date.iloc[0])
    window = best["window"].copy().sort_values("Time").reset_index(drop=True)
    window["Time_plot"] = pd.to_datetime("2000-01-01") + (
        window["Time"] - window["Time"].dt.normalize()
    )
    window_start = pd.to_datetime(best["start"])
    window_end = pd.to_datetime(best["end"])
    window["WindowStart"] = window_start
    window["WindowEnd"] = window_end
    window["WindowStart_plot"] = pd.to_datetime("2000-01-01") + (
        window_start - window_start.normalize()
    )
    window["WindowEnd_plot"] = pd.to_datetime("2000-01-01") + (
        window_end - window_end.normalize()
    )
    return window, str(best["place"]), str(best["date"])


def _select_burst_windows_topk(
    df_raw: pd.DataFrame,
    cfg: WorkflowConfig,
    top_k: int = 3,
    window_hours: float = 4.0,
    start_hour: int = 10,
    end_hour: int = 15,
    preferred_dates: Dict[str, List[str]] | None = None,
    max_per_date: int = 1,
) -> Dict[str, List[Tuple[pd.DataFrame, str]]]:
    """
    Return top-k burst windows per site using the same scoring as _select_burst_window.
    """
    df = df_raw.copy()
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    else:
        df = df.reset_index().rename(columns={"index": "Time"})
        df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")
    if "CS" not in df.columns:
        df["CS"] = _compute_cs(df, cfg)
    # Do not drop rows requiring all variables simultaneously.
    # Coverage checks are applied per-window below to allow spanning the full record even with partial gaps.
    window_minutes = int(window_hours * 60)
    candidates: List[Dict[str, object]] = []
    # Also keep a "representative" window per date (passes coverage checks, not necessarily strongest burst),
    # so we can stitch segments spanning the full record without duplicating near-identical windows.
    best_any: Dict[str, Dict[str, Dict[str, object]]] = {}
    # Keep multiple representative windows per date to fill non-burst segments in light color.
    any_by_date: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for place, sub_place in df.groupby("place"):
        sub_place = sub_place.infer_objects(copy=False).set_index("Time")
        numeric_cols = sub_place.select_dtypes(include=[np.number]).columns
        sub_place_numeric = sub_place[numeric_cols].resample("1min").mean()
        sub_place_numeric = sub_place_numeric.interpolate(limit=5, limit_direction="both")
        for col in ["bvocs", "SOA", "CS"]:
            if col in sub_place_numeric.columns:
                sub_place_numeric[col] = sub_place_numeric[col].clip(lower=0.0)
        sub_place = sub_place_numeric.copy()
        sub_place["place"] = place
        sub_place = sub_place.reset_index().rename(columns={"index": "Time"})
        sub_place["date"] = sub_place["Time"].dt.date
        sub_mid = sub_place[
            (sub_place["Time"].dt.hour >= start_hour)
            & (sub_place["Time"].dt.hour <= end_hour)
        ]
        date_groups = list(sub_mid.groupby("date"))
        if preferred_dates and place in preferred_dates:
            pref_list = [pd.to_datetime(x).date() for x in preferred_dates[place]]
            preferred_set = set(pref_list)
            preferred_groups = [(d, g) for d, g in date_groups if d in preferred_set]
            if preferred_groups:
                non_preferred_groups = [(d, g) for d, g in date_groups if d not in preferred_set]
                preferred_groups = sorted(
                    preferred_groups,
                    key=lambda x: pref_list.index(x[0]) if x[0] in pref_list else 999,
                )
                date_groups = preferred_groups + non_preferred_groups
        for date_value, day_df in date_groups:
            day_sorted = day_df.sort_values("Time")
            if day_sorted.shape[0] < max(80, window_minutes // 2):
                continue
            day_key = str(date_value)
            day_base = pd.to_datetime(str(date_value))
            start_base = day_base + pd.Timedelta(hours=start_hour)
            end_base = day_base + pd.Timedelta(hours=end_hour) - pd.Timedelta(minutes=window_minutes)
            if end_base <= start_base:
                continue
            start_range = pd.date_range(start_base, end_base, freq="30min")
            for start_time in start_range:
                end_time = start_time + pd.Timedelta(minutes=window_minutes)
                window_df = day_sorted[
                    (day_sorted["Time"] >= start_time) & (day_sorted["Time"] <= end_time)
                ]
                preferred_flag = bool(preferred_dates and place in preferred_dates and date_value in [pd.to_datetime(x).date() for x in preferred_dates[place]])
                min_count_ratio = 0.35 if preferred_flag else 0.4
                if window_df.shape[0] < window_minutes * min_count_ratio:
                    continue
                coverage = window_df["bvocs"].count() / max(window_df.shape[0], 1)
                cov_thresh = 0.25 if preferred_flag else 0.35
                if coverage < cov_thresh:
                    continue
                cov_temp = window_df["temperature_c"].count() / max(window_df.shape[0], 1)
                cov_soa = window_df["SOA"].count() / max(window_df.shape[0], 1)
                cov_cs = window_df["CS"].count() / max(window_df.shape[0], 1)
                if min(cov_temp, cov_soa, cov_cs) < (0.20 if preferred_flag else 0.25):
                    continue
                edge_minutes = 30
                first_edge = window_df[window_df["Time"] < start_time + pd.Timedelta(minutes=edge_minutes)]
                last_edge = window_df[window_df["Time"] > end_time - pd.Timedelta(minutes=edge_minutes)]
                if first_edge.empty or last_edge.empty:
                    continue
                first_cov = first_edge["bvocs"].count() / max(first_edge.shape[0], 1)
                last_cov = last_edge["bvocs"].count() / max(last_edge.shape[0], 1)
                edge_thresh = 0.15 if preferred_flag else 0.20
                if min(first_cov, last_cov) < edge_thresh and not preferred_flag:
                    continue
                if preferred_flag and coverage >= cov_thresh:
                    pass
                elif min(first_cov, last_cov) < edge_thresh:
                    continue

                # Representative (coverage-passing) window scoring for full-period coverage.
                temp_std_any = float(window_df["temperature_c"].std(skipna=True))
                bvocs_std_any = float(window_df["bvocs"].std(skipna=True))
                soa_std_any = float(window_df["SOA"].std(skipna=True))
                score_any = (coverage + 1e-3) * (temp_std_any + 1e-3) * (bvocs_std_any + 1e-3) * (soa_std_any + 1e-3)
                if np.isfinite(score_any):
                    best_any.setdefault(place, {})
                    current = best_any[place].get(day_key)
                    if current is None or float(score_any) > float(current.get("score", -np.inf)):
                        best_any[place][day_key] = {
                            "window": window_df.copy(),
                            "place": place,
                            "date": day_key,
                            "score": float(score_any),
                            "start": start_time,
                            "end": end_time,
                        }
                    any_by_date.setdefault(place, {})
                    any_by_date[place].setdefault(day_key, [])
                    any_by_date[place][day_key].append(
                        {
                            "window": window_df.copy(),
                            "place": place,
                            "date": day_key,
                            "score": float(score_any),
                            "start": start_time,
                            "end": end_time,
                        }
                    )
                    any_by_date[place][day_key] = sorted(any_by_date[place][day_key], key=lambda x: x["score"], reverse=True)[:12]

                temp_range = float(window_df["temperature_c"].max() - window_df["temperature_c"].min())
                temp_std = float(window_df["temperature_c"].std(skipna=True))
                bvocs_range = float(window_df["bvocs"].max() - window_df["bvocs"].min())
                bvocs_std = float(window_df["bvocs"].std(skipna=True))
                bvocs_diff_peak = float(window_df["bvocs"].diff().abs().rolling(5, min_periods=3).max().max(skipna=True))
                soa_span = float(window_df["SOA"].max() - window_df["SOA"].min())
                corr_tb = 0.0
                temp_bv = (
                    window_df[["temperature_c", "bvocs"]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if (
                    temp_bv["temperature_c"].std(skipna=True) > 0
                    and temp_bv["bvocs"].std(skipna=True) > 0
                    and temp_bv.shape[0] > 2
                ):
                    with np.errstate(invalid="ignore"):
                        corr_tb = float(
                            np.corrcoef(temp_bv["temperature_c"], temp_bv["bvocs"])[0, 1]
                        )
                    if not np.isfinite(corr_tb):
                        corr_tb = 0.0
                soa_shift = window_df["SOA"].shift(-20)
                pair_lag = pd.concat(
                    [window_df["bvocs"], soa_shift.rename("SOA")], axis=1
                ).dropna()
                pair_lag = pair_lag.replace([np.inf, -np.inf], np.nan).dropna()
                soa_lag_corr = 0.0
                if (
                    pair_lag["bvocs"].std(skipna=True) > 0
                    and pair_lag["SOA"].std(skipna=True) > 0
                    and pair_lag.shape[0] > 2
                ):
                    with np.errstate(invalid="ignore"):
                        soa_lag_corr = float(
                            np.corrcoef(pair_lag["bvocs"], pair_lag["SOA"])[0, 1]
                        )
                    if not np.isfinite(soa_lag_corr):
                        soa_lag_corr = 0.0
                diff_min = 0.8
                if preferred_dates and place in preferred_dates:
                    pref_list = [pd.to_datetime(x).date() for x in preferred_dates[place]]
                    if date_value in pref_list:
                        diff_min = 0.4
                if bvocs_range < 0.5 or bvocs_diff_peak < diff_min:
                    continue
                score_temp = 0.5 * temp_range + 0.5 * temp_std
                score_bv = 0.45 * bvocs_range + 0.35 * bvocs_std + 0.20 * bvocs_diff_peak
                score = (score_temp + 1e-3) * (score_bv + 1e-3)
                score *= 1.0 + max(0.0, corr_tb if not np.isnan(corr_tb) else 0.0)
                score *= 1.0 + max(0.0, soa_lag_corr if not np.isnan(soa_lag_corr) else 0.0)
                score *= 1.0 + max(0.0, soa_span)
                if preferred_dates and place in preferred_dates:
                    pref_list = [pd.to_datetime(x).date() for x in preferred_dates[place]]
                    if date_value in pref_list:
                        rank = pref_list.index(date_value)
                        if rank == 0:
                            score *= 5.0
                        score *= 1.0 + 0.15 * float(len(pref_list) - rank)
                candidates.append(
                    {
                        "window": window_df.copy(),
                        "place": place,
                        "date": str(date_value),
                        "score": score,
                        "start": start_time,
                        "end": end_time,
                    }
                )
    result: Dict[str, List[Tuple[pd.DataFrame, str]]] = {}
    for place in df["place"].unique():
        place_candidates = [c for c in candidates if c["place"] == place]
        if not place_candidates and place not in best_any:
            continue
        place_candidates = sorted(place_candidates, key=lambda x: x["score"], reverse=True)
        # Pool: representative-by-date (coverage passing) plus burst candidates. Allow multiple segments per date.
        best_any_by_date: Dict[str, Dict[str, object]] = dict(best_any.get(place, {}))
        any_by_date_place: Dict[str, List[Dict[str, object]]] = dict(any_by_date.get(place, {}))
        cands_by_date: Dict[str, List[Dict[str, object]]] = {}
        for cand in place_candidates:
            d = str(cand["date"])
            cands_by_date.setdefault(d, []).append(cand)
        for d, lst in cands_by_date.items():
            cands_by_date[d] = sorted(lst, key=lambda x: x["score"], reverse=True)

        def _overlap_frac(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> float:
            left = max(a0, b0)
            right = min(a1, b1)
            inter = max(0.0, (right - left).total_seconds())
            denom = min((a1 - a0).total_seconds(), (b1 - b0).total_seconds())
            if denom <= 0:
                return 1.0
            return float(inter / denom)

        def _select_non_overlapping(
            lst: List[Dict[str, object]],
            n_keep: int,
            max_overlap: float = 0.15,
            min_sep_min: float = 45.0,
        ) -> List[Dict[str, object]]:
            chosen: List[Dict[str, object]] = []
            for cand in lst:
                a0 = pd.to_datetime(cand["start"])
                a1 = pd.to_datetime(cand["end"])
                ok = True
                for prev in chosen:
                    b0 = pd.to_datetime(prev["start"])
                    b1 = pd.to_datetime(prev["end"])
                    if _overlap_frac(a0, a1, b0, b1) > max_overlap:
                        ok = False
                        break
                    if abs((a0 - b0).total_seconds()) < min_sep_min * 60.0:
                        ok = False
                        break
                if ok:
                    chosen.append(cand)
                if len(chosen) >= n_keep:
                    break
            return chosen

        def _fill_with_any(
            chosen: List[Dict[str, object]],
            pool: List[Dict[str, object]],
            n_total: int,
        ) -> List[Dict[str, object]]:
            if len(chosen) >= n_total:
                return chosen
            for cand in pool:
                if len(chosen) >= n_total:
                    break
                a0 = pd.to_datetime(cand["start"])
                a1 = pd.to_datetime(cand["end"])
                ok = True
                for prev in chosen:
                    b0 = pd.to_datetime(prev["start"])
                    b1 = pd.to_datetime(prev["end"])
                    if _overlap_frac(a0, a1, b0, b1) > 0.20:
                        ok = False
                        break
                    if abs((a0 - b0).total_seconds()) < 30.0 * 60.0:
                        ok = False
                        break
                if ok:
                    chosen.append(cand)
            return chosen

        dates_all = sorted(
            set(list(best_any_by_date.keys()) + list(cands_by_date.keys())),
            key=lambda s: pd.to_datetime(s, errors="coerce"),
        )
        selected: List[Dict[str, object]] = []
        selected_dates: set[str] = set()

        if preferred_dates and place in preferred_dates:
            for ds in preferred_dates[place]:
                if ds in selected_dates:
                    continue
                picked = _select_non_overlapping(cands_by_date.get(ds, []), max(1, int(max_per_date)))
                picked = _fill_with_any(picked, any_by_date_place.get(ds, []), max(1, int(max_per_date)))
                if not picked and ds in best_any_by_date:
                    picked = [best_any_by_date[ds]]
                for p in picked:
                    selected.append(p)
                selected_dates.add(ds)

        remaining = [d for d in dates_all if d not in selected_dates]
        for d in remaining:
            if len(selected) >= int(top_k):
                break
            picked = _select_non_overlapping(cands_by_date.get(d, []), max(1, int(max_per_date)))
            picked = _fill_with_any(picked, any_by_date_place.get(d, []), max(1, int(max_per_date)))
            if not picked and d in best_any_by_date:
                picked = [best_any_by_date[d]]
            for p in picked:
                if len(selected) >= int(top_k):
                    break
                selected.append(p)

        if not selected:
            selected = place_candidates[: min(top_k, len(place_candidates))]
        result[place] = []
        for cand in selected:
            window = cand["window"].copy().sort_values("Time").reset_index(drop=True)
            window["Time_plot"] = pd.to_datetime("2000-01-01") + (
                window["Time"] - window["Time"].dt.normalize()
            )
            window_start = pd.to_datetime(cand["start"])
            window_end = pd.to_datetime(cand["end"])
            window["WindowStart"] = window_start
            window["WindowEnd"] = window_end
            window["WindowStart_plot"] = pd.to_datetime("2000-01-01") + (
                window_start - window_start.normalize()
            )
            window["WindowEnd_plot"] = pd.to_datetime("2000-01-01") + (
                window_end - window_end.normalize()
            )
            result[place].append((window, str(cand["date"])))
    return result


def _plot_burst_case(
    ax,
    window_df: pd.DataFrame,
    place: str,
    date_str: str,
    show_temp_label: bool = True,
    show_conc_label: bool = True,
    show_temp_ticks: bool = True,
    show_conc_ticks: bool = True,
    add_legend: bool = True,
    conc_ylim: Tuple[float, float] | None = None,
) -> None:
    if window_df.empty:
        ax.text(0.5, 0.5, "No burst window found", transform=ax.transAxes, ha="center", va="center")
        return
    window_df = window_df.sort_values("Time")
    window_start_plot = (
        pd.to_datetime(window_df["WindowStart_plot"].iloc[0])
        if "WindowStart_plot" in window_df.columns
        else None
    )
    window_end_plot = (
        pd.to_datetime(window_df["WindowEnd_plot"].iloc[0])
        if "WindowEnd_plot" in window_df.columns
        else None
    )
    valid_mask = (
        window_df[["temperature_c", "bvocs", "SOA"]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .any(axis=1)
    )
    window_df = window_df.loc[valid_mask].copy()
    if window_df.shape[0] < 10:
        ax.text(0.5, 0.5, "Not enough valid points", transform=ax.transAxes, ha="center", va="center")
        return
    times = window_df["Time_plot"] if "Time_plot" in window_df.columns else window_df["Time"]
    _add_day_background(ax, times, alpha=0.05, color="#c0c0c0")
    # Downsample CS shading to avoid dense stripes at the window start.
    cs_series = pd.Series(window_df["CS"].to_numpy(), index=pd.to_datetime(times))
    cs_span = cs_series.resample("3min").mean().dropna()
    cs_times = cs_span.index.to_pydatetime()
    if len(cs_span) < 2:
        cs_span = cs_series
        cs_times = cs_span.index.to_pydatetime()
    if cs_span.notna().any():
        vmin = float(np.nanmin(cs_span))
        vmax = float(np.nanmax(cs_span))
        if vmin == vmax:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        cs_use = cs_span.fillna(cs_span.median())
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
        cs_use = pd.Series([0.0] * len(cs_span), index=cs_times)
    for i in range(len(cs_use) - 1):
        cs_val = float(cs_use.iloc[i])
        alpha = 0.03 + 0.18 * float(norm(cs_val))
        color_val = "#c6c6c6"
        ax.axvspan(cs_use.index[i], cs_use.index[i + 1], color=color_val, alpha=alpha, zorder=-6)
    ax.plot(times, window_df["temperature_c"], color="#d62728", linewidth=2.0, label="Temperature", zorder=3)
    temp_min = float(np.nanmin(window_df["temperature_c"]))
    temp_max = float(np.nanmax(window_df["temperature_c"]))
    ax.set_ylim(temp_min - 0.5, temp_max + 0.5)
    ax.set_ylabel("Temperature (C)" if show_temp_label else "", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728", labelleft=show_temp_ticks)
    # Use raw BVOCs to preserve amplitude.
    bv_raw = window_df["bvocs"]
    # Avoid smoothing across missing segments to prevent visually artificial plateaus.
    bv_smooth = pd.Series(bv_raw).rolling(window=3, center=True, min_periods=3).mean()
    soa_series = window_df["SOA"]
    soa_smooth = pd.Series(soa_series).rolling(window=3, center=True, min_periods=3).mean()

    ax_right = ax.twinx()
    ax_right.grid(False)
    ax_right.set_ylabel("Concentration (ug m$^{-3}$)" if show_conc_label else "", color="#2c2c2c", labelpad=10)
    ax_right.tick_params(axis="y", colors="#2c2c2c", labelsize=9, length=3, labelright=show_conc_ticks)

    bv_plot = np.ma.masked_invalid(bv_smooth.to_numpy(dtype=float))
    soa_plot = np.ma.masked_invalid(soa_smooth.to_numpy(dtype=float))
    ax_right.fill_between(times, 0, bv_plot, color="#2ca02c", alpha=0.18, zorder=2)
    ax_right.plot(times, bv_plot, color="#2ca02c", linewidth=1.8, alpha=0.9, label="Isoprene (BVOCs)", zorder=4)
    ax_right.plot(times, soa_plot, color="#5b3ea4", linewidth=2.0, label="SOA", zorder=5)

    if conc_ylim is not None:
        ax_right.set_ylim(conc_ylim[0], conc_ylim[1])
    else:
        conc = (
            np.concatenate([bv_plot.compressed(), soa_plot.compressed()])
            if (bv_plot.count() + soa_plot.count()) > 0
            else np.array([0.0])
        )
        y_low = 0.0
        y_high = float(np.nanpercentile(conc, 99)) if conc.size else 1.0
        y_high = max(y_high, 1e-3)
        ax_right.set_ylim(y_low, y_high * 1.10)
    burst_idx = int(window_df["bvocs"].astype(float).idxmax())
    burst_time = window_df.loc[burst_idx, "Time_plot"] if "Time_plot" in window_df.columns else window_df.loc[burst_idx, "Time"]
    ax.axvline(burst_time, color="#6c6c6c", linestyle="--", linewidth=1.3, alpha=0.9)
    ax.text(
        burst_time,
        0.92,
        "Precursor burst",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=13,
        color="#4d4d4d",
    )
    _apply_axis_style(ax)
    _apply_burst_time_ticks(ax, rotation=15)
    if window_start_plot is not None and window_end_plot is not None:
        start_lim = window_start_plot.floor("30min")
        end_lim = window_end_plot.ceil("30min")
    else:
        start_lim = pd.to_datetime(times.min()).floor("30min")
        end_lim = pd.to_datetime(times.max()).ceil("30min")
    ax.set_xlim(start_lim, end_lim)
    ax_right.set_xlim(start_lim, end_lim)
    ax.set_xlabel("Time (HH:MM)")
    ax.set_title(
        f"Process-scale burst case ({place}, {date_str})",
        fontweight="bold",
        fontsize=15,
        loc="left",
    )
    if add_legend:
        legend_handles = [
            Line2D([0], [0], color="#d62728", linewidth=2.0, label="Temperature"),
            Line2D([0], [0], color="#2ca02c", linewidth=1.8, label="Isoprene (BVOCs)"),
            Line2D([0], [0], color="#5b3ea4", linewidth=2.0, label="SOA"),
            Line2D([0], [0], color="#9c9c9c", linewidth=6.0, alpha=0.30, label="CS background"),
        ]
        ax_right.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper right",
            ncol=1,
            fontsize=10,
            columnspacing=1.0,
            handlelength=2.2,
        )


def _build_process_scale_figure(
    df_hour: pd.DataFrame,
    burst_map: Dict[str, List[Tuple[pd.DataFrame, str]]],
    out_dir: Path,
    stationarity_text: str | None = None,
) -> None:
    panels: List[Tuple[str, str, pd.DataFrame]] = []
    for place, cases in burst_map.items():
        per_place_count = 0
        used_starts: List[pd.Timestamp] = []
        for df_case, date_str in cases:
            if df_case.empty:
                continue
            start_time = pd.to_datetime(df_case["Time"].min())
            if any(abs((start_time - s).total_seconds()) < 20.0 * 60.0 for s in used_starts):
                continue
            panels.append((place, date_str, df_case))
            used_starts.append(start_time)
            per_place_count += 1
            if per_place_count >= 14:  # sample enough segments to estimate shared axis ranges
                break
    if not panels:
        return

    cs_pool: List[float] = []
    bv_pool: List[float] = []
    soa_pool: List[float] = []
    for _, _, df_cs in panels:
        df_cs = df_cs.replace([np.inf, -np.inf], np.nan)
        if df_cs.empty:
            continue
        t_cs = pd.to_datetime(df_cs["Time_plot"] if "Time_plot" in df_cs.columns else df_cs["Time"])
        cs_series = pd.Series(pd.to_numeric(df_cs["CS"], errors="coerce").to_numpy(), index=t_cs)
        cs_span = cs_series.resample("10min").mean().dropna()
        cs_pool.extend([float(v) for v in cs_span.to_list() if np.isfinite(v)])
        bv_tmp = pd.to_numeric(df_cs.get("bvocs"), errors="coerce")
        soa_tmp = pd.to_numeric(df_cs.get("SOA"), errors="coerce")
        bv_tmp = pd.Series(bv_tmp).rolling(window=3, center=True, min_periods=3).mean()
        soa_tmp = pd.Series(soa_tmp).rolling(window=3, center=True, min_periods=3).mean()
        bv_pool.extend([float(v) for v in bv_tmp.dropna().to_list() if np.isfinite(v) and v >= 0])
        soa_pool.extend([float(v) for v in soa_tmp.dropna().to_list() if np.isfinite(v) and v >= 0])
    cs_vmin = float(np.nanpercentile(np.asarray(cs_pool), 5)) if cs_pool else None
    cs_vmax = float(np.nanpercentile(np.asarray(cs_pool), 95)) if cs_pool else None
    if cs_vmin is not None and cs_vmax is not None and cs_vmin == cs_vmax:
        cs_vmax = cs_vmin + 1e-9
    bv_ymax = float(np.nanpercentile(np.asarray(bv_pool), 99)) * 1.10 if bv_pool else None
    soa_ymax = float(np.nanpercentile(np.asarray(soa_pool), 99)) * 1.10 if soa_pool else None

    panels = sorted(
        panels,
        key=lambda x: pd.to_datetime(x[2]["Time"].min()) if not x[2].empty else pd.Timestamp.max,
    )
    # A flatter layout improves readability for stitched multi-window panels.
    fig = plt.figure(figsize=(11, 5.5))
    # Leave headroom for the legend and the header date boxes (outside data axes).
    outer = GridSpec(1, 2, hspace=0.0, wspace=0.28, left=0.070, right=0.985, top=0.67, bottom=0.20)

    legend_handles = [
        Line2D([0], [0], color="#d62728", linewidth=2.6, label="Temperature"),
        Line2D([0], [0], color="#2ca02c", linewidth=2.3, label="Isoprene (BVOCs)"),
        Line2D([0], [0], color="#5b3ea4", linewidth=2.3, label="SOA"),
        Patch(facecolor="#8f8f8f", edgecolor="none", alpha=0.45, label="CS background"),
    ]

    for col_idx, place in enumerate(["JH", "CM"]):
        cases = burst_map.get(place, [])
        # Keep multiple burst segments, but avoid duplicates and keep figure readable.
        cases_sorted = sorted(cases, key=lambda x: pd.to_datetime(x[0]["Time"].min()) if not x[0].empty else pd.Timestamp.max)
        used_starts: List[pd.Timestamp] = []
        cases_keep: List[Tuple[pd.DataFrame, str]] = []
        for df_case, date_str in cases_sorted:
            if df_case.empty:
                continue
            start_time = pd.to_datetime(df_case["Time"].min())
            if any(abs((start_time - s).total_seconds()) < 20.0 * 60.0 for s in used_starts):
                continue
            cases_keep.append((df_case, date_str))
            used_starts.append(start_time)
        max_segments = 18
        if len(cases_keep) > max_segments:
            idx = np.linspace(0, len(cases_keep) - 1, max_segments)
            chosen_idx: List[int] = []
            for v in idx:
                k = int(round(float(v)))
                if k < 0:
                    k = 0
                if k >= len(cases_keep):
                    k = len(cases_keep) - 1
                if k not in chosen_idx:
                    chosen_idx.append(k)
            cases_use = [cases_keep[k] for k in chosen_idx]
        else:
            cases_use = cases_keep
        # Use a single column per site; header row holds date boxes and titles.
        sub = GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=outer[0, col_idx],
            height_ratios=[0.34, 1.0, 0.96],
            hspace=0.06,
        )
        ax_header = fig.add_subplot(sub[0, 0])
        ax_top = fig.add_subplot(sub[1, 0], sharex=ax_header)
        ax_bottom = fig.add_subplot(sub[2, 0], sharex=ax_header)
        ax_stats = None
        _plot_burst_case_stitched(
            ax_header,
            ax_top,
            ax_bottom,
            ax_stats,
            cases_use,
            place,
            cs_vmin=cs_vmin,
            cs_vmax=cs_vmax,
            bv_ymax=bv_ymax,
            soa_ymax=soa_ymax,
        )

    # Put legend in top margin to avoid overlap with titles/dates.
    legend_y = 0.76
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, legend_y), 
        ncol=4,
        fontsize=12,
        handlelength=2.3,
        columnspacing=1.15,
    )
    if stationarity_text:
        fig.text(
            0.52,
            legend_y - 0.07,
            stationarity_text,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#2c2c2c",
        )
    fig.savefig(out_dir / "SOA_process_scale_case.png", bbox_inches="tight", transparent=False, dpi=500)
    fig.savefig(out_dir / "SOA_process_scale_case.svg", bbox_inches="tight", transparent=False, dpi=500)
    plt.close(fig)



def main() -> None:
    # First, generate the core time-series stationarity figures with the main project script.
    repo_root = BUNDLE_ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts import time_series_stationarity as core_ts  # type: ignore
    except Exception as exc:
        print(f"[WARN] Could not import scripts.time_series_stationarity: {exc}")
    else:
        try:
            core_ts.main()
        except Exception as exc:
            print(f"[WARN] core time_series_stationarity.main() failed: {exc}")
        else:
            src_dir = repo_root / "figures"
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            mapping = {
                "Fig_timeseries_stationarity.png": "SOA_timeseries_stationarity.png",
                "Fig_delta_SOA_combined.png": "SOA_delta_timeseries_combined.png",
            }
            for src_name, dst_name in mapping.items():
                src = src_dir / src_name
                dst = FIGURE_DIR / dst_name
                if src.exists():
                    shutil.copyfile(src, dst)
                else:
                    print(f"[WARN] Missing expected source figure: {src}")

    # Then, generate the process-scale stitched burst figure within the paper bundle.
    _set_style()
    cfg = default_config()

    df_raw = mf.load_base_data(cfg)
    df_raw["CS"] = _compute_cs(df_raw, cfg)
    df = _prepare_diurnal_df(df_raw, cfg)

    cols_to_keep = ["bvocs", "SOA", "CS", "temperature_c", "delta_SOA", "delta_CS"]
    df_hour = df[["Time", "place", *cols_to_keep]].copy()
    # Compute dSOA/dt per place using time difference (hours).
    df_hour["dSOA_dt"] = np.nan
    for place, sub in df_hour.groupby("place"):
        sub = sub.sort_values("Time")
        dt_hours = sub["Time"].diff().dt.total_seconds() / 3600.0
        dsoa = sub["SOA"].diff()
        df_hour.loc[sub.index, "dSOA_dt"] = (dsoa / dt_hours).astype(float)
    # Estimate xi and k_loss per place, then derive C_hat and its delta.
    df_hour["C_hat"] = np.nan
    df_hour["delta_C_hat"] = np.nan
    for place, sub in df_hour.groupby("place"):
        xi_hat, k_loss_hat = _fit_xi_kloss(sub)
        sub = sub.sort_values("Time")
        # Avoid division by zero.
        denom = xi_hat * sub["CS"].replace(0, np.nan)
        c_hat = (sub["dSOA_dt"] + k_loss_hat * sub["SOA"]) / denom
        c_hat = c_hat.astype(float)
        df_hour.loc[sub.index, "C_hat"] = c_hat
        df_hour.loc[sub.index, "delta_C_hat"] = c_hat.diff().fillna(0.0)

    for col in cols_to_keep:
        df_hour[f"{col}_smooth"] = df_hour.groupby("place")[col].transform(_smooth)
    df_hour["delta_C_hat_smooth"] = df_hour.groupby("place")["delta_C_hat"].transform(_smooth)

    out_dir = FIGURE_DIR
    # Process-scale stitched figure: use the raw source inputs (chemistry + size distribution)
    # to allow selecting multiple burst windows per available day.
    df_process = _load_process_scale_source_data(cfg)
    burst_windows = _select_burst_windows_topk(
        df_process,
        cfg,
        top_k=36,
        window_hours=2.0,
        start_hour=0,
        end_hour=23,
        preferred_dates=None,
        max_per_date=3,
    )
    # stationarity_text = _format_stationarity_text(_compute_stationarity_from_full_df(df_hour))
    stationarity_text = None
    for place, cases in burst_windows.items():
        preview = cases[:8]
        for df_case, date_str in preview:
            if not df_case.empty:
                print(
                    f"[burst] Selected window {place} {date_str} "
                    f"{df_case['Time'].min()} -> {df_case['Time'].max()}"
                )
        if len(cases) > len(preview):
            print(f"[burst] {place} total windows selected: {len(cases)}")
    _build_process_scale_figure(df_hour, burst_windows, out_dir)
    print(
        "Saved SOA_timeseries_stationarity.png, SOA_delta_timeseries_combined.png, and SOA_process_scale_case.[png|svg] to paper\\figure"
    )


if __name__ == "__main__":
    main()
