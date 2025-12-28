from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.container import BarContainer
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


def adjust_color_lightness(color_hex: str, amount: float = 0.8) -> str:
    base_rgb = np.array(mcolors.to_rgb(color_hex))
    new_rgb = 1.0 - (1.0 - base_rgb) * amount
    new_rgb = np.clip(new_rgb, 0.0, 1.0)
    return mcolors.to_hex(new_rgb)
from skopt.plots import plot_objective
from sklearn.metrics import r2_score

from pathlib import Path

from paper.workflow.lib.paper_paths import CHECKPOINT_DIR, FIGURE_DIR, TABLES_DIR
from .modeling_framework_paper import (
    save_dual,
    _r2_uncentered,
    _build_env_features,
    _default_plot_labels,
    _train_ml_model,
    build_ml_training_set,
    default_config,
    load_cached_results,
    set_plot_style,
)
from src.workflow import explainers


def _compute_effects(params_df: pd.DataFrame, df_place: pd.DataFrame, model_id: str) -> pd.DataFrame:
    env = _build_env_features(df_place)
    rows = []
    for _, row in params_df.iterrows():
        param = row["Parameter"]
        std_err = row.get("StdErr", np.nan)
        p_val = row.get("p_value", np.nan)
        if model_id == "3":
            if param != "C_T_hat":
                continue
            feature_mean = 1.0
        elif model_id == "2":
            if param not in env:
                continue
            feature_mean = (env[param] * df_place["bvoc_mu_hat"]).mean()
        else:
            if param not in env:
                continue
            feature_mean = (env[param] * df_place["bvocs"]).mean()
        rows.append(
            {
                "Parameter": param,
                "Effect": row["Estimate"] * feature_mean,
                "StdErr": std_err * feature_mean if pd.notna(std_err) else np.nan,
                "Significance": row.get("Significance", ""),
                "p_value": p_val,
                "Place": row["Place"],
            }
        )
    return pd.DataFrame(rows)


def plot_category_i(
    df: pd.DataFrame,
    cat1_outputs: dict,
    cat2_outputs: dict | None = None,
    labels_cfg: dict | None = None,
) -> None:
    # plt.rcParams.update(...) removed to respect global set_plot_style()
    label_cfg = labels_cfg or _default_plot_labels()
    # Complementary color pair tuned for Nature-style contrast.
    palette_places_default = {"JH": "#2c5c7d", "CM": "#c26c35"}
    palette_places = {**palette_places_default, **label_cfg.get("palette_places", {})}
    palette_extra_defaults = {
        "line": "#1f3b4d",
        "cs": "#6a994e",
        "temp": "#38598b",
        "cs_m1": "#2c7bb6",
        "cs_m2": "#66a61e",
        "cs_m3": "#7b3294",
    }
    palette_extra = {**palette_extra_defaults, **label_cfg.get("palette_extra", {})}
    marker_map_defaults = {"linear": "o", "cs": "D"}
    marker_map = {**marker_map_defaults, **label_cfg.get("marker_map", {})}
    marker_linear = marker_map.get("linear", "o")
    marker_cs = marker_map.get("cs", "D")
    marker_edgecolor = label_cfg.get("marker_edgecolor", "none")
    marker_edgewidth = float(label_cfg.get("marker_edgewidth", 0.0))
    marker_size_linear = int(label_cfg.get("marker_size_linear", 18))
    marker_size_cs = int(label_cfg.get("marker_size_cs", 18))
    scatter_alpha_linear = float(label_cfg.get("scatter_alpha_linear", 0.55))
    scatter_alpha_cs = float(label_cfg.get("scatter_alpha_cs", 0.45))
    cs_lightness_amount = float(label_cfg.get("cs_lightness_amount", 0.35))
    cs_jitter_scale = float(label_cfg.get("cs_jitter_scale", 0.0))
    kde_cmap = label_cfg.get("kde_cmap", "crest")
    panel_label_size = int(label_cfg.get("panel_label_size", 15))
    legend_fontsize = float(label_cfg.get("legend_fontsize", 10))
    legend_loc_main = label_cfg.get("legend_loc_main", "lower right")
    legend_bbox_main = label_cfg.get("legend_bbox_main")
    legend_loc = label_cfg.get("legend_loc", legend_loc_main)
    legend_bbox = label_cfg.get("legend_bbox", legend_bbox_main or (0.02, 0.98))
    rng = np.random.default_rng(20241125)
    def sci_fmt(val: float) -> str:
        try:
            return f"{val:.2e}"
        except Exception:
            return "nan"
    def sci_tex(val: float) -> str:
        try:
            if not np.isfinite(val):
                return "nan"
            if val == 0:
                return "0"
            exp = int(np.floor(np.log10(abs(val))))
            if exp == 0:
                return f"{val:.2f}"
            mant = val / (10 ** exp)
            return f"{mant:.2f}" + r"\times 10^{" + f"{exp}" + r"}"
        except Exception:
            return "nan"
    def fmt_sig(sig: str, p_val: float | None = None) -> str:
        sig_clean = (sig or "").strip()
        if not sig_clean and p_val is not None and np.isfinite(p_val):
            if p_val <= 0.001:
                sig_clean = "***"
            elif p_val <= 0.01:
                sig_clean = "**"
            elif p_val <= 0.05:
                sig_clean = "*"
        if not sig_clean:
            return ""
        if "*" in sig_clean:
            return "*" * max(1, sig_clean.count("*"))
        if sig_clean.lower().startswith("p") and "<" in sig_clean:
            if "0.001" in sig_clean:
                return "***"
            if "0.01" in sig_clean:
                return "**"
            return "*"
        return "*"
    base_param_labels = {
        "HNO3": r"$\mathrm{HNO_3}$",
        "H2SO4": r"$\mathrm{H_2SO_4}$",
        "H2SO4O3": r"$\mathrm{H_2SO_4}\cdot\mathrm{O_3}$",
        "HNO3O3": r"$\mathrm{HNO_3}\cdot\mathrm{O_3}$",
        "O3hv": r"$\mathrm{O_3}\cdot h\nu$",
        "O3": r"$\mathrm{O_3}$",
        "NOx": r"$\mathrm{NO_x}$",
        "SO2": r"$\mathrm{SO_2}$",
        "RH": r"$\mathrm{RH}$",
        "hv": r"$h\nu$",
        "K": r"$k_{\mathrm{env}}$",
        "C_T_hat": r"$k_{\mathrm{temp}}$",
        "beta_max": r"$k_{\mathrm{env,max}}$",
        "CS0": r"$CS_{0}$",
    }
    user_param_labels_raw = label_cfg.get("param_labels", {})
    def _strip_math_dollars(text: str) -> str:
        if isinstance(text, str) and text.startswith("$") and text.endswith("$"):
            return text[1:-1]
        return text
    def _format_param_label(raw_label: str | None, fallback_key: str | None = None) -> str:
        label = raw_label if raw_label not in (None, "") else (fallback_key or "")
        if not isinstance(label, str):
            label = str(label)
        label = label.strip()
        if not label:
            return fallback_key or ""
        if "$" in label:
            return label
        if fallback_key and fallback_key in base_param_labels:
            return base_param_labels[fallback_key]
        if label in base_param_labels:
            return base_param_labels[label]
        parts = [p.strip() for p in label.split("*") if p.strip()]
        if len(parts) > 1:
            joined = r"\times".join([_strip_math_dollars(_format_param_label(part, part)) for part in parts])
            return f"${joined}$"
        chem_token = re.sub(r"(\d+)", r"_\\1", label)
        return rf"$\mathrm{{{chem_token}}}$"
    merged_param_labels = {**base_param_labels, **user_param_labels_raw}
    param_label_map = {key: _format_param_label(val, key) for key, val in merged_param_labels.items()}
    def readable_param(name: str) -> str:
        if name in param_label_map:
            return param_label_map[name]
        return _format_param_label(name, name)
    def apply_sci_formatter(ax, x_axis: bool = True, y_axis: bool = True):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        if x_axis:
            ax.xaxis.set_major_formatter(formatter)
        if y_axis:
            ax.yaxis.set_major_formatter(formatter)
    def make_legend_opaque(legend_obj) -> None:
        if legend_obj is None:
            return
        handles = getattr(legend_obj, "legendHandles", None)
        if handles is None:
            handles = getattr(legend_obj, "legend_handles", None)
        if handles is None:
            return
        for handle in handles:
            try:
                handle.set_alpha(1.0)
            except Exception:
                continue
    helv_bold = FontProperties(family="Helvetica Neue LT Pro", weight="bold")
    def make_bold(ax) -> None:
        if helv_bold is not None:
            ax.title.set_fontproperties(helv_bold)
            ax.xaxis.label.set_fontproperties(helv_bold)
            ax.yaxis.label.set_fontproperties(helv_bold)
        else:
            ax.title.set_fontweight("bold")
            ax.xaxis.label.set_fontweight("bold")
            ax.yaxis.label.set_fontweight("bold")
        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if helv_bold is not None:
                tick.set_fontproperties(helv_bold)
            else:
                tick.set_fontweight("bold")
    place_priority = ["JH", "CM"]
    preds = cat1_outputs["predictions"]
    params_df = cat1_outputs["params"].copy()
    if not params_df.empty and "ModelID" in params_df.columns:
        params_df["ModelID"] = params_df["ModelID"].astype(str)
    cs_preds = {} if cat2_outputs is None else cat2_outputs.get("predictions", {})
    cs_params = pd.DataFrame() if cat2_outputs is None else cat2_outputs.get("params", pd.DataFrame()).copy()
    if not cs_params.empty and "ModelID" in cs_params.columns:
        cs_params["ModelID"] = cs_params["ModelID"].astype(str)
    cs_diag = {} if cat2_outputs is None else cat2_outputs.get("diag", {})
    def get_cs_pred(place: str, mid: str):
        keys = [(place, f"{mid}_cs"), (place, mid), (place, f"{mid}_CS"), (place, "CS"), place]
        for key in keys:
            if isinstance(cs_preds, dict) and key in cs_preds:
                return cs_preds.get(key)
        return None
    def get_cs_diag(place: str, mid: str):
        keys = [(place, f"{mid}_cs"), (place, mid), (place, f"{mid}_CS"), (place, "CS"), place]
        for key in keys:
            if isinstance(cs_diag, dict) and key in cs_diag:
                return cs_diag.get(key, {})
        return {}
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.4, 13.2),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.95], "hspace": 0.32, "wspace": 0.28},
    )
    places = sorted(df["place"].dropna().unique())
    place_order_global = [p for p in place_priority if p in places] + [p for p in places if p not in place_priority]
    model_labels = {"1": "(1) Observed BVOCs", "2": "(2) Temp-fit BVOCs", "3": "(3) Temperature-only"}
    fmt_val = lambda v: "nan" if pd.isna(v) else f"{v:.2f}"
    def fmt_r_lines(r_center: float, r_uc: float) -> list[str]:
        lines = []
        if pd.notna(r_center) and r_center >= 0:
            lines.append(rf"$R_{{c}}^{{2}}={fmt_val(r_center)}$")
        if pd.notna(r_uc):
            lines.append(rf"$R_{{uc}}^{{2}}={fmt_val(r_uc)}$")
        return lines
    for r, place in enumerate(places[:2]):
        y_true = df.loc[df["place"] == place, "SOA"]
        for c, mid in enumerate(["1", "2", "3"]):
            ax = axes[r, c]
            pred = preds.get((place, mid))
            if pred is None:
                ax.axis("off")
                continue
            pred = pred.reindex(y_true.index)
            mask = y_true.notna() & pred.notna()
            if not mask.any():
                ax.axis("off")
                continue
            r2_main = r2_score(y_true[mask], pred[mask]) if mask.any() else np.nan
            r2_uc_main = _r2_uncentered(y_true[mask], pred[mask]) if mask.any() else np.nan
            cs_series = get_cs_pred(place, mid)
            cs_series = cs_series.reindex(y_true.index) if cs_series is not None else None
            cs_mask = None
            if cs_series is not None:
                cs_mask = y_true.notna() & cs_series.notna()
                if not cs_mask.any():
                    cs_mask = mask
            sns.kdeplot(x=y_true[mask], y=pred[mask], fill=True, cmap=kde_cmap, ax=ax, thresh=0.05, alpha=0.12)
            base_color = palette_places.get(place, next(iter(palette_places.values())))
            cs_color = adjust_color_lightness(base_color, amount=cs_lightness_amount)
            scatter_edgecolor = marker_edgecolor
            scatter_linewidth = marker_edgewidth
            ax.scatter(
                y_true[mask],
                pred[mask],
                s=marker_size_linear,
                alpha=scatter_alpha_linear,
                color=base_color,
                marker=marker_linear,
                edgecolors=scatter_edgecolor,
                linewidth=scatter_linewidth,
            )
            ann_lines: list[str] = []
            ann_lines += [f"base: {line}" for line in fmt_r_lines(r2_main, r2_uc_main)]
            lim_candidates = [y_true[mask], pred[mask]]
            cs_r2 = np.nan
            cs_r2_uc = np.nan
            cs_points_x = None
            cs_points_y = None
            if cs_series is not None and cs_mask is not None:
                cs_points_x = y_true[cs_mask].to_numpy()
                cs_points_y = cs_series[cs_mask].to_numpy()
                plot_cs_x = cs_points_x.copy()
                plot_cs_y = cs_points_y.copy()
                if cs_jitter_scale > 0:
                    span_cs = max(plot_cs_x.max(initial=0.0), plot_cs_y.max(initial=0.0)) - min(
                        plot_cs_x.min(initial=0.0), plot_cs_y.min(initial=0.0)
                    )
                    span_cs = span_cs if span_cs > 0 else 1.0
                    jitter_amt = cs_jitter_scale * span_cs
                    plot_cs_x = plot_cs_x + rng.normal(0.0, jitter_amt, size=plot_cs_x.shape)
                    plot_cs_y = plot_cs_y + rng.normal(0.0, jitter_amt, size=plot_cs_y.shape)
                ax.scatter(
                    plot_cs_x,
                    plot_cs_y,
                    s=marker_size_cs,
                    alpha=scatter_alpha_cs,
                    color=cs_color,
                    marker=marker_cs,
                    label="Linear Model 2 + CS",
                    edgecolors=scatter_edgecolor,
                    linewidth=scatter_linewidth,
                )
                cs_r2 = r2_score(y_true[cs_mask], cs_series[cs_mask]) if cs_mask.any() else np.nan
                cs_r2_uc = _r2_uncentered(y_true[cs_mask], cs_series[cs_mask]) if cs_mask.any() else np.nan
                ann_lines += [f"CS: {line}" for line in fmt_r_lines(cs_r2, cs_r2_uc)]
                lim_candidates.extend([pd.Series(plot_cs_x), pd.Series(plot_cs_y)])
            legend_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker=marker_linear,
                    markersize=max(marker_size_linear * 0.8, 7),
                    markerfacecolor=base_color,
                    markeredgecolor=scatter_edgecolor,
                    markeredgewidth=scatter_linewidth,
                    label="Linear Model 2",
                )
            ]
            legend_labels = ["Linear Model 2"]
            if cs_series is not None and cs_mask is not None:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="",
                        marker=marker_cs,
                        markersize=max(marker_size_cs * 0.8, 7),
                        markerfacecolor=cs_color,
                        markeredgecolor=scatter_edgecolor,
                        markeredgewidth=scatter_linewidth,
                        label="Linear Model 2 + CS",
                    )
                )
                legend_labels.append("Linear Model 2 + CS")
            legend_kwargs = {
                "title": "",
                "loc": legend_loc_main,
                "frameon": False,
                "ncol": 1,
                "borderaxespad": 0.35,
                "handletextpad": 0.4,
                "labelspacing": 0.25,
                "handlelength": 1.0,
                "scatterpoints": 1,
                "fontsize": legend_fontsize,
            }
            if legend_bbox_main is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_main
            legend_obj = ax.legend(legend_handles, legend_labels, **legend_kwargs)
            make_legend_opaque(legend_obj)
            if legend_obj is not None:
                for txt in legend_obj.get_texts():
                    txt.set_fontweight("bold")
            ann_pos = {"xy": (0.98, 0.02), "va": "bottom", "ha": "right"}
            if cs_series is not None and cs_mask is not None:
                ann_pos = {"xy": (0.02, 0.98), "va": "top", "ha": "left"}
            ax.text(
                ann_pos["xy"][0],
                ann_pos["xy"][1],
                "\n".join(ann_lines),
                transform=ax.transAxes,
                va=ann_pos["va"],
                ha=ann_pos["ha"],
                fontsize=11,
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
            all_vals = pd.concat(lim_candidates)
            val_min = all_vals.min()
            val_max = all_vals.max()
            padding = 0.05 * (val_max - val_min) if val_max > val_min else 0.1
            x_lim = (val_min - padding, val_max + padding)
            ax.plot([x_lim[0], x_lim[1]], [x_lim[0], x_lim[1]], linestyle="--", color="#6f6f6f", linewidth=1.2)
            ax.set_title(f"{place} | {model_labels[mid]}", fontweight="bold")
            ax.set_xlabel(r"$\mathrm{SOA}_{\mathrm{obs}}$", fontweight="bold")
            ax.set_ylabel(r"$\mathrm{SOA}_{\mathrm{pred}}$", fontweight="bold")
            ax.set_xlim(x_lim)
            ax.set_ylim(x_lim)
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(6))
            apply_sci_formatter(ax)
            make_bold(ax)
    for r_unused in range(len(places), 2):
        for c_unused in range(3):
            axes[r_unused, c_unused].axis("off")
    for idx, mid in enumerate(["1", "2", "3"]):
        ax = axes[2, idx]
        effects = []
        for place in places:
            sub_params = params_df[(params_df["ModelID"] == mid) & (params_df["Place"] == place)]
            if sub_params.empty:
                continue
            df_place = df[df["place"] == place]
            eff = _compute_effects(sub_params, df_place, mid)
            effects.append(eff)
        if not effects:
            ax.axis("off")
            continue
        eff_df = pd.concat(effects)
        eff_df["Readable"] = eff_df["Parameter"].apply(readable_param)
        place_order = [p for p in place_order_global if p in eff_df["Place"].unique()]
        sig_lookup = {(row["Readable"], row["Place"]): row.get("Significance", "") for _, row in eff_df.iterrows()}
        std_lookup = {(row["Readable"], row["Place"]): row.get("StdErr", np.nan) for _, row in eff_df.iterrows()}
        pval_lookup = {(row["Readable"], row["Place"]): row.get("p_value", np.nan) for _, row in eff_df.iterrows()}
        if mid == "3":
            eff_df = eff_df.rename(columns={"Effect": "Estimate"})
            eff_df = eff_df.sort_values(["Readable", "Place"])
            value_all = eff_df["Estimate"].dropna()
            mean_val = float(value_all.mean()) if not value_all.empty else 0.0
            deviations = np.abs(value_all - mean_val) if not value_all.empty else pd.Series([0.0])
            max_dev = float(deviations.max()) if not deviations.empty else 0.0
            max_err = float(eff_df["StdErr"].abs().max()) if "StdErr" in eff_df.columns and not eff_df.empty else 0.0
            span_core = max(max_dev, max_err * 3, 0.1)
            padding = max(span_core * 1.0, 0.08)
            x_lim = (mean_val - padding, mean_val + padding)
            if not value_all.empty:
                err_series = eff_df["StdErr"].abs().fillna(0.0)
                est_minus = (eff_df["Estimate"] - err_series).min()
                est_plus = (eff_df["Estimate"] + err_series).max()
                span_vals = float(est_plus - est_minus) if est_plus > est_minus else 0.0
                margin_val = max(0.08 * span_vals, 0.01)
                x_lim = (float(est_minus) - margin_val, float(est_plus) + margin_val)
            span = x_lim[1] - x_lim[0]
            text_offset = 0.08 * span
            text_offset = 0.12 * span
            place_offsets = {p: (idx - (len(place_order) - 1) / 2) * 0.25 for idx, p in enumerate(place_order)}
            y_positions_map = {p: i for i, p in enumerate(place_order)}
            for place in place_order:
                sub = eff_df[eff_df["Place"] == place]
                if sub.empty:
                    continue
                values = sub["Estimate"].tolist()
                # If multiple parameters somehow appear, use the mapped names; otherwise keep single param.
                names = sub["Readable"].tolist() if "Readable" in sub.columns else ["k_temp"] * len(values)
                errors = sub["StdErr"].tolist() if "StdErr" in sub.columns else [np.nan] * len(values)
                y_positions = [y_positions_map[place] + place_offsets.get(place, 0.0) for _ in values]
                ax.errorbar(
                    values,
                    y_positions,
                    xerr=errors,
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=1.6,
                    capsize=6,
                    alpha=0.9,
                    zorder=2.5,
                )
                seg_half_val = 0.015 * span
                ax.hlines(
                    y_positions,
                    np.array(values) - seg_half_val,
                    np.array(values) + seg_half_val,
                    colors=palette_places.get(place, palette_extra["temp"]),
                    linestyles="--",
                    linewidth=1.0,
                    alpha=0.9,
                )
                ax.plot(
                    values,
                    y_positions,
                    marker="o",
                    linestyle="--",
                    color=palette_places.get(place, palette_extra["temp"]),
                    label=place,
                    linewidth=1.2,
                )
                for xpos, ypos, pname in zip(values, y_positions, names):
                    sig = sig_lookup.get((pname, place), "")
                    p_val = pval_lookup.get((pname, place), np.nan)
                    sig_fmt = fmt_sig(sig, p_val)
                    text_pos = xpos + (text_offset if xpos >= mean_val else -text_offset)
                    text_pos = min(max(text_pos, x_lim[0] + 0.02 * span), x_lim[1] - 0.02 * span)
                    txt_core = sci_tex(xpos)
                    txt_with_sig = (f"${txt_core}$ {sig_fmt}" if sig_fmt else f"${txt_core}$").strip()
                    ax.annotate(
                        txt_with_sig,
                        xy=(text_pos, ypos),
                        xytext=(0, -10),
                        textcoords="offset points",
                        va="top",
                        ha="left" if xpos >= mean_val else "right",
                        fontsize=11,
                        fontweight="bold",
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 0.5},
                    )
            ax.set_yticks([y_positions_map[p] for p in place_order])
            ax.set_yticklabels(place_order)
            ax.axvline(mean_val, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlim(x_lim)
            ax.set_ylim(-0.6, len(place_order) - 0.4)
            ax.set_title("Temp-driven parameters (Model 3)", fontweight="bold")
            ax.set_xlabel("Coefficient value", fontweight="bold")
            ax.set_ylabel("Place", fontweight="bold")
            ax.tick_params(axis="both", labelsize=10, width=1.0)
            legend = ax.legend(title="", frameon=False)
            if legend is not None:
                legend.set_title("")
            make_legend_opaque(legend)
            if legend is not None:
                for txt in legend.get_texts():
                    txt.set_fontweight("bold")
            ax.xaxis.set_major_locator(MaxNLocator(6))
            apply_sci_formatter(ax, x_axis=True, y_axis=False)
            ax.grid(alpha=0.2, linewidth=0.8, axis="x")
            make_bold(ax)
            continue
        param_order = eff_df.groupby("Readable")["Effect"].mean().sort_values().index.tolist()
        effect_palette = label_cfg.get("effect_palette", {"JH": "#5b8c6a", "CM": "#c17c59"})
        sns.barplot(
            data=eff_df,
            x="Effect",
            y="Readable",
            hue="Place",
            order=param_order,
            hue_order=place_order,
            dodge=True,
            palette=effect_palette,
            orient="h",
            ax=ax,
            width=0.6,
        )
        # enforce chemical subscripts in yticks
        yticks_fmt = [readable_param(p) for p in param_order]
        ax.set_yticks(range(len(param_order)))
        ax.set_yticklabels(yticks_fmt, fontweight="bold", fontsize=9)
        # overlay error bars using StdErr if available
        for i, patch in enumerate(ax.patches):
            if patch is None or not hasattr(patch, "get_width"):
                continue
            param_idx = i % len(param_order)
            place_idx = i // len(param_order)
            if place_idx >= len(place_order):
                continue
            param = param_order[param_idx]
            place = place_order[place_idx]
            err_val = std_lookup.get((param, place), np.nan)
            if pd.notna(err_val):
                y_loc = patch.get_y() + patch.get_height() / 2
                x_loc = patch.get_width()
                ax.errorbar(
                    x_loc,
                    y_loc,
                    xerr=err_val,
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=1.4,
                    capsize=5,
                    alpha=0.9,
                    zorder=3,
                )
        legend = ax.legend(title="", frameon=False)
        if legend is not None:
            legend.set_title("")
        make_legend_opaque(legend)
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontweight("bold")
                txt.set_fontsize(9)
        ax.tick_params(axis="both", labelsize=10, width=1.0)
        ax.axvline(0, color="black", linewidth=0.9)
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        text_out = 0.04 * span
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        for idx_container, container in enumerate(bar_containers):
            place_label = place_order[idx_container] if idx_container < len(place_order) else container.get_label()
            if place_label == "_nolegend_":
                continue
            for bar_idx, patch in enumerate(container):
                if patch is None or not hasattr(patch, "get_width"):
                    continue
                width = patch.get_width()
                param = param_order[bar_idx]
                sig_raw = sig_lookup.get((param, place_label), "")
                p_val = pval_lookup.get((param, place_label), np.nan)
                sig_fmt = fmt_sig(sig_raw, p_val)
                y_loc = patch.get_y() + patch.get_height() / 2
                text_val = sci_tex(width)
                x0_cur, x1_cur = ax.get_xlim()
                span_cur = x1_cur - x0_cur if x1_cur != x0_cur else span or 1.0
                opp_offset = text_out
                if width >= 0:
                    x_loc = 0.0 - opp_offset
                    ha = "right"
                else:
                    x_loc = 0.0 + opp_offset
                    ha = "left"
                if text_val != "nan":
                    label_txt_core = f"${text_val}$"
                    label_txt = (f"{label_txt_core} {sig_fmt}" if sig_fmt else label_txt_core).strip()
                else:
                    label_txt = "nan"
                pe = [patheffects.withStroke(linewidth=2.2, foreground="white")]
                ax.text(
                    x_loc,
                    y_loc,
                    label_txt,
                    va="center",
                    ha=ha,
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    path_effects=pe,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.6},
                )
        ax.set_title(f"Effect (coef x mean) Model {mid}", fontweight="bold")
        ax.set_xlabel("Parameter effect", fontweight="bold")
        ax.set_ylabel("Parameter", fontweight="bold")
        ax.grid(alpha=0.2, linewidth=0.8, axis="x")
        apply_sci_formatter(ax, x_axis=True, y_axis=False)
        # widen x-limits to leave room for annotations
        x0, x1 = ax.get_xlim()
        pad_extra = 0.12 * (x1 - x0)
        left_extra = pad_extra
        right_extra = pad_extra
        if mid == "2":
            left_extra = pad_extra + 1.0
        ax.set_xlim(x0 - left_extra, x1 + right_extra)
        ax.tick_params(axis="y", pad=12)
        for tick_lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_lbl.set_fontweight("bold")
        make_bold(ax)
    fig.tight_layout()
    save_dual(fig, "SOA_linear_model_benchmark")
    plt.close(fig)

    # Dedicated CS scaling figure (only two panels; summary panel removed to emphasize fits).
    fig_cs, axes_cs = plt.subplots(
        1,
        2,
        figsize=(14.5, 4.6),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32},
    )
    axes_cs = np.atleast_1d(axes_cs)
    panel_label_size_cs = max(panel_label_size, 17)
    label_fontsize_cs = 14
    tick_fontsize_cs = 13
    legend_fontsize_cs = max(legend_fontsize, 12)
    cs_palette = {**palette_places, **{"CM": "#c27d38", "JH": "#1f7a8c"}}
    fit_color = "#3f4142"
    beta_line_color = "#5b5b5b"
    cs0_color = "#8a8175"
    summary_marker_color = "#2f5d62"
    scatter_size_cs = 26
    legend_kwargs_cs = {
        "frameon": False,
        "loc": "lower right",
        "bbox_to_anchor": (0.98, 0.02),
        "borderaxespad": 0.0,
        "handlelength": 1.5,
        "handletextpad": 0.35,
        "labelspacing": 0.3,
        "fontsize": legend_fontsize_cs,
    }
    for col_idx, place in enumerate(places[:2]):
        ax_cs = axes_cs[col_idx]
        diag_entry = get_cs_diag(place, "2")
        cs_series = diag_entry.get("cs")
        ratio = diag_entry.get("ratio")
        lin_pred = diag_entry.get("lin_pred")
        obs = diag_entry.get("obs")
        if cs_series is None or ratio is None or obs is None or lin_pred is None:
            ax_cs.axis("off")
            continue
        mask_cs = obs.notna() & lin_pred.notna() & cs_series.notna() & ratio.notna()
        if mask_cs.sum() == 0:
            ax_cs.axis("off")
            continue
        cs_vals_masked = cs_series[mask_cs]
        ratio_vals_masked = ratio[mask_cs]
        # Remove extreme values using robust quantiles to stabilize plotting.
        cs_q_low, cs_q_high = np.nanquantile(cs_vals_masked, [0.01, 0.99])
        ratio_q_low, ratio_q_high = np.nanquantile(ratio_vals_masked, [0.01, 0.99])
        mask_trim = (
            (cs_series >= cs_q_low)
            & (cs_series <= cs_q_high)
            & (ratio >= ratio_q_low)
            & (ratio <= ratio_q_high)
        )
        mask_cs = mask_cs & mask_trim
        if mask_cs.sum() == 0:
            ax_cs.axis("off")
            continue
        cs_sorted_idx = cs_series[mask_cs].sort_values().index
        ax_cs.scatter(
            cs_series.loc[cs_sorted_idx],
            ratio.loc[cs_sorted_idx],
            s=scatter_size_cs,
            alpha=0.32,
            color=cs_palette.get(place, palette_extra["line"]),
            label=r"$\mathrm{SOA}/\mathrm{SOA}_{\mathrm{lin}}$",
            edgecolors="none",
            linewidth=0.0,
        )
        cs_sorted = np.linspace(cs_series[mask_cs].min(), cs_series[mask_cs].max(), 200)
        beta_max = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "beta_max"), "Estimate"].mean()
        beta_se = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "beta_max"), "StdErr"].mean()
        cs0 = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "CS0"), "Estimate"].mean()
        cs0_se = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "CS0"), "StdErr"].mean()
        if not np.isnan(beta_max) and not np.isnan(cs0):
            k_curve = beta_max * cs_sorted / (cs_sorted + cs0)
            morandi_line = fit_color
            ax_cs.plot(
                cs_sorted,
                k_curve,
                color=morandi_line,
                linewidth=2.4,
                label=r"$k_{\mathrm{env}}(CS)$ fit",
                alpha=0.95,
            )
            # References: k_env = beta_max * CS / (CS + CS0); 95% CI via delta method assuming independence.
            # Parameters: beta_max, CS0 with their standard errors; CS along cs_sorted grid.
            if not np.isnan(beta_se) and not np.isnan(cs0_se):
                dk_dbeta = cs_sorted / (cs_sorted + cs0)
                dk_dcs0 = -(beta_max * cs_sorted) / (cs_sorted + cs0) ** 2
                var_k = (dk_dbeta * beta_se) ** 2 + (dk_dcs0 * cs0_se) ** 2
                se_k = np.sqrt(np.maximum(var_k, 0.0))
                k_upper = k_curve + 1.96 * se_k
                k_lower = k_curve - 1.96 * se_k
                ax_cs.fill_between(
                    cs_sorted,
                    k_lower,
                    k_upper,
                    color=morandi_line,
                    alpha=0.18,
                    linewidth=0.0,
                    label="95% CI",
                )
            ann_line = rf"$\mathbf{{k_{{\mathrm{{env,max}}}}={fmt_val(beta_max)}}}$, $\mathbf{{CS_{{0}}={fmt_val(cs0)}}}$"
            ax_cs.text(
                0.02,
                0.95,
                ann_line,
                transform=ax_cs.transAxes,
                va="top",
                ha="left",
                fontsize=15,
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
        ax_cs.set_xlabel("CS", fontsize=label_fontsize_cs, fontweight="bold")
        ax_cs.set_ylabel(r"$\mathrm{SOA}/\mathrm{SOA}_{\mathrm{lin}}$", fontsize=label_fontsize_cs, fontweight="bold")
        ax_cs.set_title(
            rf"{place} | $k_{{\mathrm{{env}}}}(CS)$ in Linear Model 2",
            fontsize=label_fontsize_cs + 1,
            fontweight="bold",
        )
        legend_cs = ax_cs.legend(**legend_kwargs_cs)
        make_legend_opaque(legend_cs)
        if legend_cs is not None:
            for txt in legend_cs.get_texts():
                txt.set_fontweight("bold")
        x_min, x_max = cs_series[mask_cs].min(), cs_series[mask_cs].max()
        y_min, y_max = ratio[mask_cs].min(), ratio[mask_cs].max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_pad = 0.05 * x_span if x_span > 0 else 0.5
        y_pad = 0.1 * y_span if y_span > 0 else 0.1
        ax_cs.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_cs.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_cs.xaxis.set_major_locator(MaxNLocator(6))
        ax_cs.yaxis.set_major_locator(MaxNLocator(6))
        ax_cs.grid(alpha=0.28, linewidth=0.9, color="#c7c7c7")
        ax_cs.tick_params(axis="both", labelsize=tick_fontsize_cs, width=1.4, length=6)
        for tick_lbl in list(ax_cs.get_xticklabels()) + list(ax_cs.get_yticklabels()):
            tick_lbl.set_fontweight("bold")
        for spine in ax_cs.spines.values():
            spine.set_linewidth(1.1)
        apply_sci_formatter(ax_cs)
    if len(places) < 2:
        for c_unused in range(len(places), 2):
            axes_cs[c_unused].axis("off")
    fig_cs.tight_layout()
    save_dual(fig_cs, "SOA_linear_model_cs_scaling")
    plt.close(fig_cs)



def plot_category_ii(cat2: dict) -> None:
    diag = cat2.get("diag", {})
    metrics_df = cat2.get("metrics", pd.DataFrame())
    palette_cfg = _default_plot_labels()
    palette_extra_defaults = {
        "line": "#4e79a7",
        "cs": "#d95f02",
        "temp": "#7570b3",
        "cs_m1": "#7b3294",
        "cs_m2": "#e79f00",
        "cs_m3": "#3182bd",
    }
    palette_extra = {**palette_extra_defaults, **palette_cfg.get("palette_extra", {})}
    for key, d in diag.items():
        if d is None:
            continue
        place, model_id = (key if isinstance(key, tuple) and len(key) == 2 else (key, "2_cs"))
        cs_place = d.get("cs")
        ratio = d.get("ratio")
        lin_pred = d.get("lin_pred")
        soa_pred_cs = d.get("soa_pred_cs")
        obs = d.get("obs")
        if cs_place is None or ratio is None or lin_pred is None or soa_pred_cs is None or obs is None:
            continue
        mask = obs.notna() & lin_pred.notna() & cs_place.notna() & ratio.notna()
        if mask.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        color_cs = palette_extra.get(f"cs_m{str(model_id)[0]}", palette_extra.get("cs", "#1b9e77"))
        color_cs_soft = adjust_color_lightness(color_cs, amount=0.55)
        morandi_line = "#8f8c80"
        morandi_aux = "#b1aaa2"
        ax.scatter(
            cs_place[mask],
            ratio[mask],
            s=22,
            alpha=0.28,
            color=color_cs_soft,
            edgecolors="none",
        )
        cs_sorted = np.linspace(cs_place[mask].min(), cs_place[mask].max(), 240)
        beta_max = metrics_df.loc[metrics_df["Place"] == place, "beta_max"].mean()
        cs0 = metrics_df.loc[metrics_df["Place"] == place, "CS0"].mean()
        if not np.isnan(beta_max) and not np.isnan(cs0):
            ax.plot(
                cs_sorted,
                beta_max * cs_sorted / (cs_sorted + cs0),
                color=morandi_line,
                linewidth=2.6,
                alpha=0.9,
                label="Model fit",
            )
        ax.axhline(1.0, color=morandi_aux, linestyle="--", linewidth=1.1, alpha=0.75)
        ax.set_xlabel("CS")
        ax.set_ylabel(r"$\mathrm{SOA}/\mathrm{SOA}_{\mathrm{lin}}$")
        ax.set_title(f"{place} CS response ({model_id})")
        legend_obj = ax.legend(frameon=False, loc="lower right")
        if legend_obj is not None:
            for txt in legend_obj.get_texts():
                txt.set_fontsize(10)
        ax.grid(alpha=0.25, linewidth=0.8)
        save_dual(fig, f"Fig_CS_response_SOA_{place}_{model_id}")
        plt.close(fig)

        fig_res, ax_res = plt.subplots(figsize=(6.2, 4.2))
        resid_before = (obs[mask] - lin_pred[mask]).dropna()
        resid_after = (obs[mask] - soa_pred_cs[mask]).dropna()
        base_color = "#4d708a"
        sns.kdeplot(resid_before, fill=True, label="Baseline", color=base_color, alpha=0.35, linewidth=1.2, ax=ax_res)
        sns.kdeplot(resid_after, fill=True, label="With CS", color=color_cs, alpha=0.35, linewidth=1.2, ax=ax_res)
        ax_res.axvline(0.0, color="#6c757d", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_res.set_xlabel("Residual")
        ax_res.set_title(f"{place} residual comparison ({model_id})")
        ax_res.legend(frameon=False)
        ax_res.grid(alpha=0.25, linewidth=0.8)
        save_dual(fig_res, f"Fig_residual_comparison_CS_{place}_{model_id}")
        plt.close(fig_res)


def plot_category_iii(df: pd.DataFrame, cat1: Dict[str, pd.DataFrame], ml_outputs: Dict[str, pd.DataFrame], cat2: Dict[str, pd.DataFrame] | None = None) -> None:
    metrics_df = ml_outputs.get("metrics", pd.DataFrame())
    metrics_cs_df = ml_outputs.get("metrics_cs", pd.DataFrame())
    feature_df = ml_outputs.get("features", pd.DataFrame())
    best_predictions = ml_outputs.get("best_predictions", {})
    all_predictions = ml_outputs.get("all_predictions", {})
    all_predictions_cs = ml_outputs.get("all_predictions_cs", {})
    cs_preds = {} if cat2 is None else cat2.get("predictions", {})
    palette_cfg = _default_plot_labels()
    palette_extra_defaults = {
        "line": "#4e79a7",
        "cs": "#d95f02",
        "temp": "#7570b3",
        "cs_m1": "#7b3294",
        "cs_m2": "#e79f00",
        "cs_m3": "#3182bd",
    }
    palette_extra = {**palette_extra_defaults, **palette_cfg.get("palette_extra", {})}
    marker_map_defaults = {"1_cs": "s", "2_cs": "o", "3_cs": "^"}
    panel_fontsize = 14
    title_fontsize = 14
    label_fontsize = 13
    tick_fontsize = 12
    xlabel_text = r"$\mathbf{SOA}_{\mathrm{obs}}$"
    ylabel_text = r"$\mathbf{SOA}_{\mathrm{pred}}$"
    def fetch_cs_pred(place: str, mid: str = "2"):
        keys = [(place, f"{mid}_cs"), (place, mid), (place, f"{mid}_CS"), (place, "CS"), place]
        for key in keys:
            if isinstance(cs_preds, dict) and key in cs_preds:
                return cs_preds.get(key)
        return None
    def fmt_r_text(r_center: float, r_uc: float) -> str:
        show_rc = pd.notna(r_center) and r_center >= 0
        show_ruc = pd.notna(r_uc)
        if show_rc and show_ruc:
            return f"$R^2_c$={r_center:.2f}, $R^2_{{uc}}$={r_uc:.2f}"
        if show_rc:
            return f"$R^2_c$={r_center:.2f}"
        if show_ruc:
            return f"$R^2_{{uc}}$={r_uc:.2f}"
        return ""

    def make_legend_opaque(legend_obj) -> None:
        if legend_obj is None:
            return
        handles = getattr(legend_obj, "legendHandles", None)
        if handles is None:
            handles = getattr(legend_obj, "legend_handles", None)
        if handles is None:
            return
        for handle in handles:
            try:
                handle.set_alpha(1.0)
            except Exception:
                continue
    if metrics_df.empty:
        return
    # Multi-model vs linear scatter (Y1 target) in 2x2 layout: rows=place (CM/JH), cols=without CS vs with CS
    if all_predictions:
        places_all = list(df["place"].dropna().unique())
        place_priority = ["CM", "JH"]
        places = [p for p in place_priority if p in places_all] + [p for p in places_all if p not in place_priority]
        if not places:
            places = places_all
        fig_multi, axes_multi = plt.subplots(len(places), 2, figsize=(13.2, 4.9 * len(places)), sharex=False, sharey=False)
        if len(places) == 1:
            axes_multi = np.array([axes_multi])
        model_names_all = sorted({key[2] for key in all_predictions.keys() if key[1] == "Y1"})
        model_names_cs = sorted({key[2] for key in all_predictions_cs.keys() if key[1] == "Y1_cs"})
        all_models_union = sorted(set(model_names_all + model_names_cs))
        palette_nature = [
            "#264653",
            "#2a9d8f",
            "#e9c46a",
            "#f4a261",
            "#e76f51",
            "#6a4c93",
            "#457b9d",
            "#b56576",
        ]
        color_map = {name: palette_nature[idx % len(palette_nature)] for idx, name in enumerate(all_models_union)}
        legend_kwargs_top = {
            "loc": "upper left",
            "frameon": False,
            "ncol": 1,
            "columnspacing": 0.6,
            "borderaxespad": 0.25,
            "handletextpad": 0.4,
            "labelspacing": 0.7,
            "markerscale": 1.0,
        }
        legend_kwargs_bottom = {
            "loc": "lower right",
            "frameon": False,
            "ncol": 1,
            "columnspacing": 0.6,
            "borderaxespad": 0.25,
            "handletextpad": 0.4,
            "labelspacing": 0.7,
            "markerscale": 1.0,
        }
        color_linear_base = "#274060"
        color_linear_cs = palette_extra.get("cs_m2", palette_extra.get("cs", "#c44536"))

        def style_axis(ax):
            ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.2, length=5, pad=4)
            for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                label.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_linewidth(1.1)
            ax.grid(alpha=0.28, linewidth=0.9)

        def get_ml_cs_entry(place_name: str, mdl: str):
            entry_direct = all_predictions_cs.get((place_name, "Y1_cs", mdl))
            if entry_direct is not None:
                return entry_direct
            base_entry = all_predictions.get((place_name, "Y1", mdl))
            if base_entry is None:
                return None
            y_true_series, y_pred_series = base_entry["y_true"], base_entry["y_pred"]
            ratio_series = None
            base_lin = cat1["predictions"].get((place_name, "2"))
            cs_lin = fetch_cs_pred(place_name, "2")
            if base_lin is not None and cs_lin is not None:
                ratio_series = pd.Series(cs_lin).reindex(y_pred_series.index) / pd.Series(base_lin).reindex(y_pred_series.index)
            if ratio_series is None:
                ratio_series = pd.Series(1.0, index=y_pred_series.index)
            ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            y_pred_cs = y_pred_series.reindex(ratio_series.index) * ratio_series
            y_true_cs = y_true_series.reindex(ratio_series.index)
            r2_cs_c = r2_score(y_true_cs, y_pred_cs) if y_true_cs.notna().any() and y_pred_cs.notna().any() else np.nan
            r2_cs_uc_val = _r2_uncentered(y_true_cs, y_pred_cs) if y_true_cs.notna().any() and y_pred_cs.notna().any() else np.nan
            return {"y_true": y_true_cs, "y_pred": y_pred_cs, "R2_test": r2_cs_c, "R2_test_uncentered": r2_cs_uc_val}
        if not model_names_cs:
            model_names_cs = model_names_all
        for row_idx, place in enumerate(places):
            ax_base = axes_multi[row_idx, 0]
            ax_cs = axes_multi[row_idx, 1]
            y_true_full = df.loc[df["place"] == place, "SOA"]
            base_pred = cat1["predictions"].get((place, "2"))
            base_pred = base_pred.reindex(y_true_full.index) if base_pred is not None else None
            cs_pred = fetch_cs_pred(place, "2")
            cs_pred = pd.Series(cs_pred).reindex(y_true_full.index) if cs_pred is not None else None
            # Without CS column
            lim_candidates_base = []
            if base_pred is not None:
                mask_base = y_true_full.notna() & base_pred.notna()
                if mask_base.any():
                    lim_candidates_base.append(y_true_full[mask_base])
                    lim_candidates_base.append(base_pred[mask_base])
                    r2_base = r2_score(y_true_full[mask_base], base_pred[mask_base])
                    r2_base_uc = _r2_uncentered(y_true_full[mask_base], base_pred[mask_base])
                    ax_base.scatter(
                        y_true_full[mask_base],
                        base_pred[mask_base],
                        s=26,
                        alpha=0.38,
                        color=color_linear_base,
                        label=f"Linear Model 2 ({fmt_r_text(r2_base, r2_base_uc)})",
                        edgecolors="none",
                        linewidth=0.0,
                    )
            for model_name in model_names_all:
                entry = all_predictions.get((place, "Y1", model_name))
                if entry is None:
                    continue
                y_true_series, y_pred_series = entry["y_true"], entry["y_pred"]
                y_true_aligned, y_pred_aligned = y_true_series.align(y_pred_series, join="inner")
                mask_model = y_true_aligned.notna() & y_pred_aligned.notna()
                if not mask_model.any():
                    continue
                y_true_aligned = y_true_aligned[mask_model]
                y_pred_aligned = y_pred_aligned[mask_model]
                lim_candidates_base.append(y_true_aligned)
                lim_candidates_base.append(y_pred_aligned)
                r2_c = r2_score(y_true_aligned, y_pred_aligned) if len(y_true_aligned) else np.nan
                r2_uc = _r2_uncentered(y_true_aligned, y_pred_aligned) if len(y_true_aligned) else np.nan
                ax_base.scatter(
                    y_true_aligned,
                    y_pred_aligned,
                    s=24,
                    alpha=0.45,
                    color=color_map.get(model_name, "#7570b3"),
                    marker="o",
                    label=f"{model_name} ({fmt_r_text(r2_c, r2_uc)})",
                    edgecolors="none",
                    linewidth=0.0,
                )
            if lim_candidates_base:
                concat_vals = pd.concat(lim_candidates_base)
                v_min = concat_vals.min()
                v_max = concat_vals.max()
                pad = 0.05 * (v_max - v_min) if v_max > v_min else 0.1
                x_lim = (v_min - pad, v_max + pad)
                ax_base.plot([x_lim[0], x_lim[1]], [x_lim[0], x_lim[1]], linestyle="--", color="#3b3b3b", linewidth=1.3)
                ax_base.set_xlim(x_lim)
                ax_base.set_ylim(x_lim)
            ax_base.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight="bold")
            ax_base.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight="bold")
            ax_base.set_title(f"{place} - without CS", fontsize=title_fontsize, fontweight="bold")
            # split legend: first 5 entries top-left, remainder bottom-right
            handles_all, labels_all = ax_base.get_legend_handles_labels()
            if handles_all:
                if len(handles_all) > 5:
                    leg_top = ax_base.legend(handles_all[:5], labels_all[:5], **legend_kwargs_top)
                    ax_base.add_artist(leg_top)
                    leg_bottom = ax_base.legend(handles_all[5:], labels_all[5:], **legend_kwargs_bottom)
                    leg_list = [leg_top, leg_bottom]
                else:
                    leg_single = ax_base.legend(handles_all, labels_all, **legend_kwargs_top)
                    leg_list = [leg_single]
                for leg_obj in leg_list:
                    if leg_obj:
                        for text in leg_obj.get_texts():
                            text.set_fontsize(12)
                            text.set_fontweight("bold")
                        make_legend_opaque(leg_obj)
            style_axis(ax_base)
            ax_base.xaxis.set_major_locator(MaxNLocator(6))
            ax_base.yaxis.set_major_locator(MaxNLocator(6))
            # With CS column
            lim_candidates_cs = []
            if cs_pred is not None:
                mask_cs = y_true_full.notna() & cs_pred.notna()
                if mask_cs.any():
                    lim_candidates_cs.append(y_true_full[mask_cs])
                    lim_candidates_cs.append(cs_pred[mask_cs])
                    r2_cs = r2_score(y_true_full[mask_cs], cs_pred[mask_cs])
                    r2_cs_uc = _r2_uncentered(y_true_full[mask_cs], cs_pred[mask_cs])
                    cs_marker = marker_map_defaults.get("2_cs", "D")
                    ax_cs.scatter(
                        y_true_full[mask_cs],
                        cs_pred[mask_cs],
                        s=28,
                        alpha=0.5,
                        color=color_linear_cs,
                        marker=cs_marker,
                        label=f"Linear Model 2 + CS ({fmt_r_text(r2_cs, r2_cs_uc)})",
                        edgecolors="none",
                        linewidth=0.0,
                    )
            for model_name in model_names_cs:
                entry_cs = get_ml_cs_entry(place, model_name)
                if entry_cs is None:
                    continue
                y_true_series_cs, y_pred_series_cs = entry_cs["y_true"], entry_cs["y_pred"]
                y_true_aligned_cs, y_pred_aligned_cs = y_true_series_cs.align(y_pred_series_cs, join="inner")
                mask_model_cs = y_true_aligned_cs.notna() & y_pred_aligned_cs.notna()
                if not mask_model_cs.any():
                    continue
                y_true_aligned_cs = y_true_aligned_cs[mask_model_cs]
                y_pred_aligned_cs = y_pred_aligned_cs[mask_model_cs]
                lim_candidates_cs.append(y_true_aligned_cs)
                lim_candidates_cs.append(y_pred_aligned_cs)
                r2_cs_c = r2_score(y_true_aligned_cs, y_pred_aligned_cs) if len(y_true_aligned_cs) else np.nan
                r2_cs_uc_val = _r2_uncentered(y_true_aligned_cs, y_pred_aligned_cs) if len(y_true_aligned_cs) else np.nan
                ax_cs.scatter(
                    y_true_aligned_cs,
                    y_pred_aligned_cs,
                    s=26,
                    alpha=0.5,
                    color=color_map.get(model_name, "#7570b3"),
                    marker="D",
                    label=f"{model_name} + CS ({fmt_r_text(r2_cs_c, r2_cs_uc_val)})",
                    edgecolors="none",
                    linewidth=0.0,
                )
            if lim_candidates_cs:
                concat_vals_cs = pd.concat(lim_candidates_cs)
                v_min_cs = concat_vals_cs.min()
                v_max_cs = concat_vals_cs.max()
                pad_cs = 0.05 * (v_max_cs - v_min_cs) if v_max_cs > v_min_cs else 0.1
                x_lim_cs = (v_min_cs - pad_cs, v_max_cs + pad_cs)
                ax_cs.plot([x_lim_cs[0], x_lim_cs[1]], [x_lim_cs[0], x_lim_cs[1]], linestyle="--", color="#3b3b3b", linewidth=1.3)
                ax_cs.set_xlim(x_lim_cs)
                ax_cs.set_ylim(x_lim_cs)
            ax_cs.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight="bold")
            ax_cs.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight="bold")
            ax_cs.set_title(f"{place} - with CS", fontsize=title_fontsize, fontweight="bold")
            handles_cs, labels_cs = ax_cs.get_legend_handles_labels()
            if handles_cs:
                if len(handles_cs) > 5:
                    leg_top_cs = ax_cs.legend(handles_cs[:5], labels_cs[:5], **legend_kwargs_top)
                    ax_cs.add_artist(leg_top_cs)
                    leg_bottom_cs = ax_cs.legend(handles_cs[5:], labels_cs[5:], **legend_kwargs_bottom)
                    leg_list_cs = [leg_top_cs, leg_bottom_cs]
                else:
                    leg_single_cs = ax_cs.legend(handles_cs, labels_cs, **legend_kwargs_top)
                    leg_list_cs = [leg_single_cs]
                for leg_obj in leg_list_cs:
                    if leg_obj:
                        for text in leg_obj.get_texts():
                            text.set_fontsize(12)
                            text.set_fontweight("bold")
                        make_legend_opaque(leg_obj)
            style_axis(ax_cs)
            ax_cs.xaxis.set_major_locator(MaxNLocator(6))
            ax_cs.yaxis.set_major_locator(MaxNLocator(6))
        fig_multi.subplots_adjust(right=0.85, wspace=0.32, hspace=0.32)
        fig_multi.tight_layout()
        save_dual(fig_multi, "SOA_ML_vs_linear_scatter")
        plt.close(fig_multi)


def _feature_math_labels() -> dict:
    return {
        "O3": r"$\mathbf{O}_{3}$",
        "NOx": r"$\mathbf{NO}_{\mathbf{x}}$",
        "SO2": r"$\mathbf{SO}_{\mathbf{2}}$",
        "Radiation": r"$\mathbf{h}\nu$",
        "Temperature": r"$\mathbf{T}$",
        "Humidity": r"$\mathbf{RH}$",
        "HNO3": r"$\mathbf{RH}\cdot\mathbf{NO}_{\mathbf{x}}$",
        "H2SO4": r"$\mathbf{RH}\cdot\mathbf{SO}_{\mathbf{2}}$",
        "HNO3O3": r"$\mathbf{RH}\cdot\mathbf{NO}_{\mathbf{x}}\cdot\mathbf{O}_{\mathbf{3}}$",
        "H2SO4O3": r"$\mathbf{RH}\cdot\mathbf{SO}_{\mathbf{2}}\cdot\mathbf{O}_{\mathbf{3}}$",
        "O3hv": r"$\mathbf{O}_{\mathbf{3}}\cdot \mathbf{h}\nu$",
        "C_T_hat": r"$\mathbf{BVOCs}(T)$",
        "k_env_cs": r"$\mathbf{k}_{\mathbf{env}}$",
        "doy_sin": r"$\mathbf{\sin}(2\pi\ \mathbf{doy}/365)$",
        "doy_cos": r"$\mathbf{\cos}(2\pi\ \mathbf{doy}/365)$",
        "hour_sin": r"$\mathbf{\sin}(2\pi\ \mathbf{hour}/24)$",
        "hour_cos": r"$\mathbf{\cos}(2\pi\ \mathbf{hour}/24)$",
    }


def _morandi_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    # Nature-inspired muted diverging palette (deep teal to warm ochre).
    colors = ["#0b4f6c", "#3a7ca5", "#7fb685", "#f2cc8f", "#d9822b"]
    return LinearSegmentedColormap.from_list("nature_soft", colors)


def _morandi_colors(n: int) -> list[str]:
    base = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#e41a1c", "#a65628", "#f781bf", "#999999"]
    if n <= len(base):
        return base[:n]
    # Repeat with slight cycling if more colors are needed.
    return [base[i % len(base)] for i in range(n)]


def _nature_single_hue(n: int, base_color: str = "#377eb8") -> list[str]:
    base_rgb = np.array(mcolors.to_rgb(base_color))
    shades = []
    for i in range(n):
        factor = 0.35 + 0.55 * (1 - i / max(n - 1, 1))
        rgb = 1 - (1 - base_rgb) * factor
        shades.append(mcolors.to_hex(rgb))
    return shades


def _shap_beeswarm_panel(ax, shap_values, X: pd.DataFrame, title: str, top_k: int = 10) -> None:
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[0]
    if shap_arr.shape[0] != X.shape[0]:
        # Align rows if needed
        min_len = min(shap_arr.shape[0], X.shape[0])
        shap_arr = shap_arr[:min_len, :]
        X = X.iloc[:min_len, :]

    mean_abs = np.abs(shap_arr).mean(axis=0)
    order_idx = np.argsort(mean_abs)[::-1][: min(top_k, shap_arr.shape[1])]
    feature_labels = _feature_math_labels()
    rng = np.random.default_rng(42)
    cmap = _morandi_cmap()
    norm_global = mcolors.Normalize(0.0, 1.0)
    scatter_size = 18
    jitter_scale = 0.07

    max_abs_val = np.nanmax(np.abs(shap_arr[:, order_idx])) if order_idx.size else 1.0
    x_lim = max_abs_val * 1.1 if max_abs_val > 0 else 1.0

    for pos, idx in enumerate(order_idx):
        vals = shap_arr[:, idx]
        jitter = rng.normal(scale=jitter_scale, size=len(vals))
        feat_vals = X.iloc[:, idx]
        fmin = np.nanmin(feat_vals)
        fmax = np.nanmax(feat_vals)
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin == fmax:
            fmin, fmax = -1.0, 1.0
        feat_norm = (feat_vals - fmin) / (fmax - fmin)
        colors = cmap(norm_global(np.clip(feat_norm, 0.0, 1.0)))
        ax.scatter(
            vals,
            np.full_like(vals, pos, dtype=float) + jitter,
            c=colors,
            alpha=0.6,
            s=scatter_size,
            edgecolors="none",
            linewidth=0.0,
        )
    ax.axvline(0.0, color="#404040", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_yticks(range(len(order_idx)))
    ax.set_yticklabels([feature_labels.get(X.columns[i], X.columns[i]) for i in order_idx], fontweight="bold")
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel("SHAP value", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.28, linewidth=1.0, color="#d0d0d0")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=12, width=1.1, length=5, direction="out")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_global)
    sm.set_array([])
    cb_ax = ax.inset_axes([0.60, 0.80, 0.36, 0.035], transform=ax.transAxes)
    cbar = plt.colorbar(sm, cax=cb_ax, orientation="horizontal")
    tick_pos = [0.0, 0.5, 1.0]
    cbar.set_ticks(tick_pos)
    cbar.ax.set_xticklabels(["Low", "Mid", "High"], fontsize=11, fontweight="bold")
    cbar.ax.tick_params(width=1.2, length=4, direction="out", labelsize=11)
    for tick in tick_pos:
        cbar.ax.plot(
            [tick, tick],
            [0, 1],
            color="white",
            linestyle="--",
            linewidth=1.2,
            transform=cbar.ax.get_xaxis_transform(),
            alpha=0.8,
        )
    cb_ax.set_xlabel("Normalized feature value", fontsize=12, fontweight="bold", labelpad=6)


def _best_gbdt_entries(metrics_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    sub = metrics_df[(metrics_df["Model"] == "GBDT") & (metrics_df["Target"] == target_name)]
    if sub.empty or "Place" not in sub:
        return pd.DataFrame()
    idx = sub.groupby("Place")["R2_test"].idxmax()
    return sub.loc[idx]


def plot_gbdt_shap_from_cache(cache_dir: str | None = CHECKPOINT_DIR, tables_dir: str | None = TABLES_DIR) -> None:
    cfg = default_config()
    tables_path = Path(tables_dir) if tables_dir is not None else TABLES_DIR
    df_sde, cat1_outputs, cat2_outputs, ml_outputs, labels_cfg = load_cached_results(cache_dir, tables_path)
    set_plot_style()
    metrics_df = ml_outputs.get("metrics", pd.DataFrame())
    if metrics_df.empty:
        return

    cs_params = cat2_outputs.get("params", pd.DataFrame())
    cs_series_full = cat2_outputs.get("cs", pd.Series(dtype=float))
    shap_frames: list[pd.DataFrame] = []
    runmeta_rows: list[dict] = []
    shap_payloads: list[dict] = []

    targets = [("Y1", "base"), ("Y2", "cs")]
    for target_name, variant_label in targets:
        best_rows = _best_gbdt_entries(metrics_df, target_name)
        if best_rows.empty:
            continue
        for _, row in best_rows.iterrows():
            place = row.get("Place")
            if place is None:
                continue
            df_place = df_sde[df_sde["place"] == place]
            cs_place = cs_series_full.reindex(df_place.index)
            beta_max = None
            cs0 = None
            if isinstance(cs_params, pd.DataFrame) and not cs_params.empty:
                beta_row = cs_params[(cs_params["Place"] == place) & (cs_params["Parameter"] == "beta_max")]
                cs0_row = cs_params[(cs_params["Place"] == place) & (cs_params["Parameter"] == "CS0")]
                if not beta_row.empty and not cs0_row.empty:
                    beta_max = beta_row["Estimate"].iloc[0]
                    cs0 = cs0_row["Estimate"].iloc[0]

            X_target, y_target, multiplier, variant_label_effective, ct_series, k_env_series = build_ml_training_set(
                df_place,
                cs_place,
                beta_max,
                cs0,
                target_name,
            )
            if X_target.empty or len(y_target) < 5:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_target, y_target, test_size=0.2, random_state=42, shuffle=True
            )
            metrics_local, pred_test, model, search_obj, space, pred_train = _train_ml_model(
                X_train,
                X_test,
                y_train,
                y_test,
                "GBDT",
            )

            y_test_mult = multiplier.reindex(y_test.index)
            y_test_soa = df_place["SOA"].reindex(y_test.index)
            soa_pred_test = pd.Series(pred_test, index=y_test.index) * y_test_mult
            r2_real = r2_score(y_test_soa, soa_pred_test) if y_test_soa.notna().any() else np.nan

            stem_base = f"Fig_ML_GBDT_SHAP_{place}_{variant_label_effective}"
            try:
                import shap

                sample_for_shap = X_train
                if sample_for_shap.shape[0] > cfg.shap_max_points:
                    sample_for_shap = sample_for_shap.sample(n=cfg.shap_max_points, random_state=42)
                explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(sample_for_shap)
                shap_arr = np.array(shap_values)
                if shap_arr.ndim == 3:
                    shap_arr_use = shap_arr[0]
                else:
                    shap_arr_use = shap_arr
                abs_mean = np.abs(shap_arr_use).mean(axis=0)
                shap_df = pd.DataFrame({"feature": sample_for_shap.columns, "abs_mean_shap": abs_mean})
                shap_df["Place"] = place
                shap_df["Target"] = target_name
                shap_df["Variant"] = variant_label_effective
                shap_frames.append(shap_df)
                shap_payloads.append(
                    {
                        "place": place,
                        "variant": variant_label_effective,
                        "target": target_name,
                        "shap_values": shap_values,
                        "X": sample_for_shap,
                        "shap_df": shap_df,
                    }
                )
            except Exception:
                shap_values = None

            shap_df = shap_frames[-1] if shap_frames else pd.DataFrame()

            runmeta_rows.append(
                {
                    "Place": place,
                    "Target": target_name,
                    "Variant": variant_label_effective,
                    "R2_cached": row.get("R2_test", np.nan),
                    "R2_recomputed": metrics_local.get("R2_test", np.nan),
                    "R2_real_SOA": r2_real,
                    "n_samples": len(y_target),
                    "n_features": X_target.shape[1],
                }
            )

    if shap_frames:
        shap_all = pd.concat(shap_frames, ignore_index=True)
        tables_path.mkdir(parents=True, exist_ok=True)
        shap_all.to_csv(tables_path / "Table_ML_GBDT_SHAP_importance.csv", index=False)
    if runmeta_rows:
        runmeta = pd.DataFrame(runmeta_rows)
        tables_path.mkdir(parents=True, exist_ok=True)
        runmeta.to_csv(tables_path / "Table_ML_GBDT_SHAP_runmeta.csv", index=False)

    if shap_payloads:
        def variant_label_txt(variant: str) -> str:
            return "Y1" if variant == "base" else "Y1 + CS"

        feature_labels = _feature_math_labels()
        for payload in shap_payloads:
            place = payload["place"]
            variant = payload["variant"]
            shap_vals = payload["shap_values"]
            X_cur = payload["X"]
            shap_df_cur = payload.get("shap_df", pd.DataFrame())
            exclude_feats = {"doy_sin", "doy_cos", "hour_sin", "hour_cos"}

            shap_arr = np.array(shap_vals)
            if shap_arr.ndim == 3:
                shap_arr_use = shap_arr[0]
            else:
                shap_arr_use = shap_arr
            if shap_arr_use.shape[0] != X_cur.shape[0]:
                min_len = min(shap_arr_use.shape[0], X_cur.shape[0])
                shap_arr_use = shap_arr_use[:min_len, :]
                X_cur = X_cur.iloc[:min_len, :]

            if shap_df_cur.empty:
                abs_mean = np.abs(shap_arr_use).mean(axis=0)
                shap_df_cur = pd.DataFrame({"feature": X_cur.columns, "abs_mean_shap": abs_mean})

            keep_cols = [c for c in X_cur.columns if c not in exclude_feats]
            if keep_cols:
                col_idx = [X_cur.columns.get_loc(c) for c in keep_cols]
                shap_arr_use = shap_arr_use[:, col_idx]
                X_cur = X_cur[keep_cols]
                shap_df_cur = shap_df_cur[shap_df_cur["feature"].isin(keep_cols)]
            else:
                shap_df_cur = shap_df_cur[~shap_df_cur["feature"].isin(exclude_feats)]

            shap_df_sorted = shap_df_cur.sort_values("abs_mean_shap", ascending=False)
            top_bar = shap_df_sorted.head(10)
            top_pie = shap_df_sorted.head(7)
            pie_rest = shap_df_sorted.iloc[7:]["abs_mean_shap"].sum()
            if pie_rest > 0:
                top_pie = pd.concat([top_pie, pd.DataFrame({"feature": ["Others"], "abs_mean_shap": [pie_rest]})], ignore_index=True)

            fig = plt.figure(figsize=(11.8, 5.4))
            fig.patch.set_facecolor("white")
            gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.24)
            ax_bee = fig.add_subplot(gs[0, 0])
            _shap_beeswarm_panel(
                ax_bee,
                shap_arr_use,
                X_cur,
                title=f"{place} GBDT ({variant_label_txt(variant)})",
                top_k=10,
            )

            ax_bar = fig.add_subplot(gs[0, 1])
            bar_colors = _nature_single_hue(len(top_bar), base_color="#0b4f6c")
            ax_bar.barh(
                [feature_labels.get(f, f) for f in top_bar["feature"]],
                top_bar["abs_mean_shap"],
                color=bar_colors,
                edgecolor="#0f1f2d",
                linewidth=0.6,
            )
            ax_bar.tick_params(axis="y", labelsize=11, width=1.0, length=4, direction="out")
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Mean |SHAP|", fontsize=11, fontweight="bold")
            ax_bar.set_title("Top SHAP importance", fontsize=12, fontweight="bold")
            ax_bar.grid(axis="x", alpha=0.32, linewidth=1.0, color="#d0d0d0")
            for lbl in ax_bar.get_yticklabels():
                lbl.set_fontweight("bold")
            for lbl in ax_bar.get_xticklabels():
                lbl.set_fontweight("bold")
            ax_bar.tick_params(axis="x", labelsize=11, width=1.0, length=5, direction="out")
            for bar, val in zip(ax_bar.patches, top_bar["abs_mean_shap"]):
                x = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                txt = f"{val:.2f}"
                text_obj = ax_bar.text(
                    x,
                    y,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
                text_obj.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=1.3, foreground="#0f1f2d"),
                        patheffects.Normal(),
                    ]
                )

            pie_scale = 0.88 if variant == "cs" else 0.75
            pie_fontsize = 10 if variant == "cs" else 8
            ax_pie = inset_axes(
                ax_bar,
                width=f"{pie_scale*100:.0f}%",
                height=f"{pie_scale*100:.0f}%",
                loc="center right",
                bbox_to_anchor=(0.25, -0.05, 0.65, 0.65),
                bbox_transform=ax_bar.transAxes,
                borderpad=0.6,
            )
            pie_colors = _nature_single_hue(len(top_pie), base_color="#2f755c")
            wedges, texts, autotexts = ax_pie.pie(
                top_pie["abs_mean_shap"],
                labels=[feature_labels.get(f, f) for f in top_pie["feature"]],
                colors=pie_colors,
                startangle=90,
                autopct="%1.1f%%",
                pctdistance=0.8,
                wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 0.6},
                textprops={"fontsize": pie_fontsize, "fontweight": "bold"},
            )
            for t in autotexts:
                t.set_fontsize(pie_fontsize)
                t.set_fontweight("bold")
                t.set_path_effects([
                    patheffects.Stroke(linewidth=1.2, foreground="white"),
                    patheffects.Normal(),
                ])
            ax_pie.axis("equal")
            for label_obj in texts + autotexts:
                label_obj.set_path_effects([
                    patheffects.Stroke(linewidth=1.2, foreground="white"),
                    patheffects.Normal(),
                ])
                x_pos, y_pos = label_obj.get_position()
                if x_pos > 0.0 and y_pos > 0.0:
                    label_obj.set_position((x_pos * 1.12, y_pos * 1.12))
                    label_obj.set_rotation(45)
                    label_obj.set_rotation_mode("anchor")
                    label_obj.set_horizontalalignment("left")
                    label_obj.set_verticalalignment("bottom")
            # Keep percentages fixed; no global text adjustment to avoid shifting values.

            fig.tight_layout()
            variant_label = "with_CS" if variant == "cs" else variant
            save_dual(fig, f"SOA_SHAP_{place}_{variant_label}")
            plt.close(fig)


def plot_from_cache(cache_dir: str | None = CHECKPOINT_DIR, tables_dir: str | None = TABLES_DIR) -> None:
    tables_path = Path(tables_dir) if tables_dir is not None else TABLES_DIR
    df_sde, cat1_outputs, cat2_outputs, ml_outputs, labels_cfg = load_cached_results(cache_dir, tables_path)
    set_plot_style()
    plot_category_i(df_sde, cat1_outputs, cat2_outputs, labels_cfg)
    plot_category_iii(df_sde, cat1_outputs, ml_outputs, cat2_outputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot cached results.")
    parser.add_argument("--gbdt-shap", action="store_true", help="If set, draw SHAP for top GBDT models (with and without CS).")
    args = parser.parse_args()

    plot_from_cache()
    if args.gbdt_shap:
        plot_gbdt_shap_from_cache()
