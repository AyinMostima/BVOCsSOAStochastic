from __future__ import annotations

import os
import sys
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontEntry, FontProperties
from matplotlib.text import Text
from sklearn.metrics import r2_score

THIS_BUNDLE_ROOT = Path(__file__).resolve().parents[3]  # .../paper_bundle
REPO_ROOT = Path(__file__).resolve().parents[4]
DOT = "\u00b7"
FP_HELV_BOLD = FontProperties(family="Helvetica", weight="bold")
if str(THIS_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_BUNDLE_ROOT))

from paper.workflow.lib.modeling_framework_paper import (  # noqa: E402
    _build_env_features,
    _r2_uncentered,
    load_cached_results,
)
from paper.workflow.lib.paper_paths import CHECKPOINT_DIR, FIGURE_DIR, TABLES_DIR  # noqa: E402


def _register_helvetica_fonts_strict() -> str:
    """
    Register local Helvetica-family font files and return an available family name.

    Preference order is: Helvetica -> Helvetica Neue -> Helvetica Neue LT Pro.

    This function is intentionally strict: it avoids falling back to Arial/DejaVu to
    satisfy Science/Nature style constraints.
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

    # Enforce Helvetica by registering local faces under a single family name ("Helvetica")
    # with correct style/weight so Matplotlib can consistently pick bold/italic variants.
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
        # Clear font lookup cache after updating ttflist.
        try:
            font_manager.fontManager._findfont_cached.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            fp = font_manager.FontProperties(family="Helvetica")
            font_manager.findfont(fp, fallback_to_default=False)
            return "Helvetica"
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Helvetica font is required but not available. "
                "Provide a system Helvetica or keep local HelveticaNeueLTPro-*.otf files in the repo root."
            ) from exc


def _set_science_style_helvetica() -> str:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _register_helvetica_fonts_strict()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 12,
            "font.weight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.labelcolor": "#111111",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d0d0d0",
            "grid.alpha": 0.12,
            "grid.linewidth": 0.5,
            "savefig.dpi": 600,
            "figure.dpi": 200,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.04,
            "text.usetex": False,
            "mathtext.fontset": "custom",
            "mathtext.default": "bf",
            "mathtext.fallback": "stixsans",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica:bold",
        }
    )
    return "Helvetica"


def _apply_minimal_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major")
    ax.tick_params(width=0.8, length=3.0, colors="#111111", labelcolor="#111111")
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")
    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.9)


def _format_value_for_label(val: float) -> str:
    """
    Format numeric value for bar-end labels.

    Rules (user-specified):
    - If 1 < |val| < 10: show fixed decimal.
    - Otherwise, use scientific notation with base 10 exponents when |val| >= 10,
      and use compact decimals for |val| <= 1.
    """
    if not np.isfinite(val):
        return ""
    abs_v = float(abs(val))
    if 1.0 < abs_v < 10.0:
        return f"{val:.2f}"
    if abs_v >= 10.0:
        exp = int(np.floor(np.log10(abs_v)))
        mant = val / (10.0 ** exp)
        return f"${mant:.2f}{DOT} 10^{{{exp}}}$"
    # abs_v <= 1
    return f"{val:.3f}"


def _force_all_text_helvetica_bold(fig: plt.Figure) -> None:
    for txt in fig.findobj(Text):
        try:
            txt.set_fontproperties(FP_HELV_BOLD)
        except Exception:
            continue
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        try:
            leg.get_title().set_fontproperties(FP_HELV_BOLD)
        except Exception:
            pass
        for t in leg.get_texts():
            try:
                t.set_fontproperties(FP_HELV_BOLD)
            except Exception:
                continue


def _select_typical_window(
    df: pd.DataFrame, hours: float = 36.0, min_points: int = 30
) -> pd.DataFrame:
    """
    Select a typical high-variability window for an inset time-series plot.

    The intent is to show that the model captures short-term fluctuations,
    complementing the pooled scatter R^2 view.
    """
    if df.empty:
        return df
    df = df.copy()
    # Prefer datetime index if available; otherwise use positional windowing.
    t = pd.to_datetime(df.index, errors="coerce")
    use_time = bool(t.notna().sum() >= max(min_points, 10))
    if use_time:
        df = df.assign(_t=t).sort_values("_t")
        dt = df["_t"].diff().dropna()
        dt_sec = float(np.nanmedian(dt.dt.total_seconds().to_numpy(dtype=float))) if not dt.empty else float("nan")
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            use_time = False
        else:
            n = int(max(min_points, round(hours * 3600.0 / dt_sec)))
            n = min(n, int(df.shape[0]))
            if n < min_points:
                return df.drop(columns=["_t"], errors="ignore")
            obs = pd.to_numeric(df["obs"], errors="coerce").to_numpy(dtype=float)
            obs = np.where(np.isfinite(obs), obs, np.nan)
            roll = pd.Series(obs).rolling(window=n, min_periods=max(10, n // 3)).std()
            if roll.dropna().empty:
                i0 = max(0, df.shape[0] - n)
            else:
                i0 = int(np.nanargmax(roll.to_numpy(dtype=float)) - n + 1)
                i0 = int(np.clip(i0, 0, max(0, df.shape[0] - n)))
            out = df.iloc[i0 : i0 + n].copy()
            return out.drop(columns=["_t"], errors="ignore")

    # Fallback: positional selection using rolling std on observed series.
    obs = pd.to_numeric(df["obs"], errors="coerce").to_numpy(dtype=float)
    obs = np.where(np.isfinite(obs), obs, np.nan)
    n = min(max(min_points, 48), int(df.shape[0]))
    if n < min_points:
        return df
    roll = pd.Series(obs).rolling(window=n, min_periods=max(10, n // 3)).std()
    if roll.dropna().empty:
        i0 = max(0, df.shape[0] - n)
    else:
        i0 = int(np.nanargmax(roll.to_numpy(dtype=float)) - n + 1)
        i0 = int(np.clip(i0, 0, max(0, df.shape[0] - n)))
    return df.iloc[i0 : i0 + n].copy()


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#111111",
    )


def _draw_stats_box(
    ax: plt.Axes,
    header: str,
    rows: list[tuple[str, str]],
    anchor: tuple[float, float] = (0.98, 0.02),
    box_alpha: float = 0.80,
) -> None:
    """
    Draw a two-column stats box with right-aligned values.

    Parameters
    ----------
    - header: title line shown at top of the box.
    - rows: list of (label, value) pairs.
    - anchor: (x, y) in axes fraction for the bottom-right of the box.
    """
    x_right, y_bottom = anchor
    line_h = 0.048
    n_lines = 1 + max(1, len(rows))
    height = line_h * n_lines + 0.018
    width = 0.50
    x0 = x_right - width
    y0 = y_bottom

    if box_alpha > 0:
        patch = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.015,rounding_size=0.01",
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.6,
            alpha=box_alpha,
            zorder=10,
        )
        ax.add_patch(patch)
    ax.text(
        x0 + 0.02,
        y0 + height - 0.02,
        header,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        zorder=11,
    )
    y = y0 + height - 0.02 - line_h
    for lab, val in rows:
        ax.text(
            x0 + 0.02,
            y,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            zorder=11,
        )
        ax.text(
            x0 + width - 0.02,
            y,
            val,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            zorder=11,
        )
        y -= line_h


def _r2_pooled(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(r2_score(y_true[mask], y_pred[mask]))


def _r2_uc_pooled(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(_r2_uncentered(y_true[mask].to_numpy(dtype=float), y_pred[mask].to_numpy(dtype=float)))


def _feature_category(feature: str) -> str:
    feature = str(feature).strip()
    if feature in {"Temperature", "Humidity", "Radiation", "hour_cos", "hour_sin", "doy_cos", "doy_sin"}:
        return "meteorology"
    if feature in {"O3", "NOx", "SO2", "O3hv"}:
        return "oxidants"
    if feature in {"HNO3", "H2SO4", "HNO3O3", "H2SO4O3"}:
        return "acids"
    if feature in {"k_env_cs"}:
        return "cs_gate"
    if feature in {"C_T_hat"}:
        return "bvoc_baseline"
    return "other"


def _category_palette() -> dict[str, str]:
    # Category colors (muted, Nature-friendly; aligned with Seaborn "deep" tones).
    return {
        "meteorology": "#DD8452",
        "oxidants": "#4C72B0",
        "acids": "#8172B2",
        "cs_gate": "#64B5CD",
        "bvoc_baseline": "#55A868",
        "other": "#8C8C8C",
    }


def _adjust_color_lightness(color_hex: str, amount: float = 0.82) -> str:
    base_rgb = np.array(mcolors.to_rgb(color_hex))
    new_rgb = 1.0 - (1.0 - base_rgb) * float(amount)
    new_rgb = np.clip(new_rgb, 0.0, 1.0)
    return mcolors.to_hex(new_rgb)


def _format_chem_token(token: str) -> str:
    token = str(token).strip()
    if token.lower() == "hv":
        return r"h\nu"
    if token == "NOx":
        return r"\mathrm{NO_{x}}"
    if token == "SO2":
        return r"\mathrm{SO_{2}}"
    if token == "O3":
        return r"\mathrm{O_{3}}"
    token_tex = re.sub(r"(\d+)", r"_{\1}", token)
    return rf"\mathrm{{{token_tex}}}"


def _format_feature_label(name: str) -> str:
    """
    Format feature labels with LaTeX for chemical tokens.

    The formatting is conservative: only chemical-like tokens are typeset.
    """
    name = str(name).strip()
    if name in {"hour_cos", "hour_sin", "doy_cos", "doy_sin"}:
        return name.replace("_", r"\_")
    # Strict label mapping aligned with:
    # - paper_bundle\paper\figure\SOA_SHAP_CM_base.png
    # - paper_bundle\paper\figure\SOA_linear_model_benchmark.png
    # Notes:
    # - HNO3 and H2SO4 are used as NOx/SO2 proxies in the paper figures and are
    #   displayed as RH-multiplied terms.
    # - Mechanistic correspondence (paper notation, schematic):
    #   C_SOA = C(T) * k_eff, where C(T) is BVOCs(T) and
    #   k_eff ~ k_O3hv * [O3] * (h nu)
    #         + k_NOx,O3 * [RH] * [NOx] * [O3]
    #         + k_SO2,O3 * [RH] * [SO2] * [O3]
    #         + k_NOx * [RH] * [NOx]
    #         + k_SO2 * [RH] * [SO2]
    #         + k_c + k_hv * (h nu)
    #   Here, the linear regressor "K" is constant (K=1), so its coefficient is the
    #   constant term k_c (intercept-like baseline contribution).
    feature_map = {
        "Temperature": rf"$\mathrm{{T}}$",
        "Humidity": rf"$\mathrm{{RH}}$",
        "RH": rf"$\mathrm{{RH}}$",
        "NOx": rf"$\mathrm{{NO_x}}$",
        "SO2": rf"$\mathrm{{SO_2}}$",
        "O3": rf"$\mathrm{{O_3}}$",
        "Radiation": rf"$h\nu$",
        "hv": rf"$h\nu$",
        "O3hv": rf"$\mathrm{{O_3}}{DOT}h\nu$",
        "k_env_cs": rf"$k_{{\mathrm{{env}}}}$",
        # K is a constant regressor (env["K"]=1.0), i.e., the intercept / constant term (k_c).
        "K": rf"$k_{{c}}$",
        "k_env": rf"$k_{{\mathrm{{env}}}}$",
        "C_T_hat": rf"$\mathrm{{BVOCs}}(T)$",
        "BVOCs(T)": rf"$\mathrm{{BVOCs}}(T)$",
        "HNO3": rf"$\mathrm{{RH}}{DOT}\mathrm{{NO_x}}$",
        "H2SO4": rf"$\mathrm{{RH}}{DOT}\mathrm{{SO_2}}$",
        "HNO3O3": rf"$\mathrm{{RH}}{DOT}\mathrm{{NO_x}}{DOT}\mathrm{{O_3}}$",
        "H2SO4O3": rf"$\mathrm{{RH}}{DOT}\mathrm{{SO_2}}{DOT}\mathrm{{O_3}}$",
    }
    if name in feature_map:
        return feature_map[name]

    def _fmt_tok(tok: str) -> str:
        tok = str(tok).strip()
        if tok in {"C_T_hat", "BVOCs(T)"}:
            return r"\mathrm{BVOCs}(T)"
        if tok in {"k_env_cs", "k_env"}:
            return r"k_{\mathrm{env}}"
        if tok == "K":
            return r"k_{c}"
        if tok in {"Humidity", "RH"}:
            return r"\mathrm{RH}"
        if tok == "Temperature":
            return r"\mathrm{T}"
        return _format_chem_token(tok)

    # Common interaction shorthands used in this project (render as products).
    if name.endswith("hv") and name != "hv":
        left = name[: -len("hv")].strip()
        if left:
            return f"${_fmt_tok(left)}{DOT}\\,{_fmt_tok('hv')}$"
    if name.endswith("O3") and name != "O3":
        left = name[: -len("O3")].strip()
        if left:
            return f"${_fmt_tok(left)}{DOT}\\,{_fmt_tok('O3')}$"
    if "*" in name:
        parts = [p.strip() for p in name.split("*") if p.strip()]
        joined = f"{DOT}\\,".join([_fmt_tok(p) for p in parts])
        return f"${joined}$"
    return f"${_fmt_tok(name)}$"


def _standardize_model2_params(
    df_sde: pd.DataFrame, cat1_params: pd.DataFrame, top_n: int = 7
) -> pd.DataFrame:
    """
    References
    ----------
    - Schielzeth, H. (2010) Methods in Ecology and Evolution.
    - Gelman, A. (2008) Statistics in Medicine.

    Mathematical expression
    -----------------------
    beta*_i = beta_i * sd(x_i) / sd(y)
    se*_i   = se_i   * sd(x_i) / sd(y)

    Parameter meanings
    ------------------
    - beta_i: raw linear coefficient from Model 2.
    - se_i: standard error of beta_i.
    - x_i: predictor used in Model 2, here x_i = env_i * BVOC_mu_hat.
    - y: observed SOA.
    """
    rows = []
    params_m2 = cat1_params.copy()
    params_m2 = params_m2[params_m2["ModelID"].astype(str) == "2"].copy()
    for place in ["JH", "CM"]:
        sub = df_sde[df_sde["place"] == place].copy()
        if sub.empty:
            continue
        y = pd.to_numeric(sub["SOA"], errors="coerce")
        env = _build_env_features(sub)
        mu = pd.to_numeric(sub.get("bvoc_mu_hat"), errors="coerce")
        y_sd = float(np.nanstd(y.to_numpy(dtype=float), ddof=1))
        if not np.isfinite(y_sd) or y_sd <= 0:
            continue
        params_place = params_m2[params_m2["Place"] == place]
        for _, row in params_place.iterrows():
            param = str(row["Parameter"])
            if param not in env.columns:
                continue
            x = pd.to_numeric(env[param], errors="coerce") * mu
            x_sd = float(np.nanstd(x.to_numpy(dtype=float), ddof=1))
            if not np.isfinite(x_sd) or x_sd <= 0:
                continue
            beta = float(row["Estimate"])
            se = float(row["StdErr"]) if pd.notna(row.get("StdErr")) else float("nan")
            rows.append(
                {
                    "Place": place,
                    "Parameter": param,
                    "beta_std": beta * x_sd / y_sd,
                    "se_std": se * x_sd / y_sd if np.isfinite(se) else float("nan"),
                    "p_value": float(row.get("p_value")) if pd.notna(row.get("p_value")) else float("nan"),
                    "Significance": str(row.get("Significance", "")) if pd.notna(row.get("Significance")) else "",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    rank = (
        out.groupby("Parameter")["beta_std"]
        .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
        .sort_values(ascending=False)
    )
    keep = rank.head(int(top_n)).index.tolist()
    return out[out["Parameter"].isin(keep)].copy()


def plot_new_fig4(
    cache_dir: str | Path = CHECKPOINT_DIR,
    tables_dir: str | Path = TABLES_DIR,
    out_stem: str = "New_Fig4_Model_Performance",
) -> Path:
    """
    Create New Fig. 4 (Model Performance & Attribution) from cached results.

    Output is saved to paper\\figure as a PNG, without overwriting existing legacy figures.
    """
    _set_science_style_helvetica()
    cache_dir = Path(cache_dir)
    tables_dir = Path(tables_dir)

    df_sde, cat1, cat2, _ml, _labels = load_cached_results(cache_dir, tables_dir)
    if cat2 is None:
        raise RuntimeError("Cat2 cached outputs are required (CS scaling) but were not found.")

    preds_cat1 = cat1.get("predictions", {})
    preds_cs = cat2.get("predictions", {})
    diag = cat2.get("diag", {})
    cs_params = cat2.get("params", pd.DataFrame()).copy()
    cat1_params = cat1.get("params", pd.DataFrame()).copy()

    # Panel A: best-performing Model 2 + CS correction (pooled across sites, colored by site).
    points = []
    per_site: dict[str, dict[str, pd.Series]] = {}
    for place in ["JH", "CM"]:
        y = df_sde.loc[df_sde["place"] == place, "SOA"]
        y = pd.to_numeric(y, errors="coerce")
        key_cs = (place, "2_cs")
        if key_cs not in preds_cs:
            raise KeyError(f"Missing CS-corrected prediction series for {key_cs}.")
        yhat = pd.to_numeric(preds_cs[key_cs].reindex(y.index), errors="coerce")
        points.append(pd.DataFrame({"Place": place, "SOA_obs": y, "SOA_pred": yhat}))
        try:
            yhat_m1 = pd.to_numeric(preds_cat1[(place, "1")].reindex(y.index), errors="coerce")
        except Exception:
            yhat_m1 = pd.Series(index=y.index, dtype=float)
        try:
            yhat_m3 = pd.to_numeric(preds_cat1[(place, "3")].reindex(y.index), errors="coerce")
        except Exception:
            yhat_m3 = pd.Series(index=y.index, dtype=float)
        per_site[place] = {"obs": y, "pred1": yhat_m1, "pred2": yhat, "pred3": yhat_m3}
    scatter_df = pd.concat(points, ignore_index=False)

    # R2 comparison text box (pooled across sites).
    pooled_obs = pd.concat(
        [pd.to_numeric(df_sde.loc[df_sde["place"] == p, "SOA"], errors="coerce") for p in ["JH", "CM"]]
    )
    pooled_pred_m1 = pd.concat(
        [
            pd.to_numeric(preds_cat1[(p, "1")].reindex(df_sde.loc[df_sde["place"] == p].index), errors="coerce")
            for p in ["JH", "CM"]
        ]
    )
    pooled_pred_m3 = pd.concat(
        [
            pd.to_numeric(preds_cat1[(p, "3")].reindex(df_sde.loc[df_sde["place"] == p].index), errors="coerce")
            for p in ["JH", "CM"]
        ]
    )
    pooled_pred_m2 = pd.concat(
        [
            pd.to_numeric(preds_cs[(p, "2_cs")].reindex(df_sde.loc[df_sde["place"] == p].index), errors="coerce")
            for p in ["JH", "CM"]
        ]
    )
    r2_m1 = _r2_pooled(pooled_obs, pooled_pred_m1)
    r2_m3 = _r2_pooled(pooled_obs, pooled_pred_m3)
    r2_m2 = _r2_pooled(pooled_obs, pooled_pred_m2)
    r2uc_m1 = _r2_uc_pooled(pooled_obs, pooled_pred_m1)
    r2uc_m3 = _r2_uc_pooled(pooled_obs, pooled_pred_m3)
    r2uc_m2 = _r2_uc_pooled(pooled_obs, pooled_pred_m2)

    # Panel B: CS modulation curve (Michaelis-Menten saturation) for Model 2.
    cs_fit = {}
    for place in ["JH", "CM"]:
        beta = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "beta_max"), "Estimate"]
        cs0 = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "CS0"), "Estimate"]
        beta_se = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "beta_max"), "StdErr"]
        cs0_se = cs_params.loc[(cs_params["Place"] == place) & (cs_params["Parameter"] == "CS0"), "StdErr"]
        if beta.empty or cs0.empty:
            continue
        se_beta = float(beta_se.mean()) if not beta_se.empty else float("nan")
        se_cs0 = float(cs0_se.mean()) if not cs0_se.empty else float("nan")
        cs_fit[place] = (float(beta.mean()), float(cs0.mean()), se_beta, se_cs0)

    # Panel C1: standardized coefficients from Model 2 (top drivers).
    coef_df = _standardize_model2_params(df_sde, cat1_params, top_n=7)

    # Panel C2: SHAP feature importance (GBDT, with CS) from the bundle tables.
    shap_path = tables_dir / "Table_ML_GBDT_SHAP_importance.csv"
    if not shap_path.exists():
        raise FileNotFoundError(f"Missing SHAP importance table: {shap_path}")
    shap_df = pd.read_csv(shap_path)
    shap_df = shap_df[(shap_df["Variant"] == "cs") & (shap_df["Target"] == "Y2")].copy()
    shap_df = shap_df[~shap_df["feature"].astype(str).str.contains("hour", case=False, na=False)].copy()
    if shap_df.empty:
        raise RuntimeError("No SHAP rows found for Variant='cs' and Target='Y2'.")
    shap_all = shap_df.copy()
    shap_rank = (
        shap_df.groupby("feature")["abs_mean_shap"]
        .mean()
        .sort_values(ascending=False)
    )
    # Keep feature names consistent between Panel C1 (linear) and Panel C2 (SHAP):
    # linear Model 2 uses proxy/environment terms (HNO3, H2SO4, ...), while GBDT SHAP
    # may rank a slightly different subset. Enforce inclusion of the linear drivers in
    # SHAP bars so readers can compare importance vs coefficient direction.
    must_include = set(coef_df["Parameter"].astype(str).tolist()) if (isinstance(coef_df, pd.DataFrame) and not coef_df.empty and "Parameter" in coef_df.columns) else set()
    # K is an intercept term (constant regressor) and will not appear as a SHAP feature.
    must_include.discard("K")
    # Always include key RH-multiplied proxy terms used in the paper's SHAP naming.
    must_include |= {"H2SO4", "H2SO4O3"}

    ranked = shap_rank.index.tolist()
    # Start from SHAP top ranks, then append missing linear terms.
    top_features: list[str] = []
    for f in ranked:
        if f not in top_features:
            top_features.append(f)
        if len(top_features) >= 9:
            break
    for f in sorted(must_include):
        if f in ranked and f not in top_features:
            top_features.append(f)
    # Cap to a compact set while guaranteeing inclusion of must_include if present.
    max_keep = 11
    if len(top_features) > max_keep:
        protected = [f for f in top_features if f in must_include]
        remainder = [f for f in top_features if f not in must_include]
        top_features = remainder[: max(0, max_keep - len(protected))] + protected
        # Preserve SHAP ranking order where possible.
        top_features = [f for f in ranked if f in top_features] + [f for f in protected if f not in ranked]

    shap_df = shap_df[shap_df["feature"].isin(top_features)].copy()

    # --- Figure layout (A,B on top; C spanning full width) ---
    legend_anchor = (0.98, 0.02)
    legend_kw = {
        "frameon": False,
        "loc": "lower right",
        "bbox_to_anchor": legend_anchor,
        "borderaxespad": 0.0,
        "handletextpad": 0.3,
        "prop": FP_HELV_BOLD,
    }

    fig = plt.figure(figsize=(12.8, 8.4), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.28)
    # Panel A is split into a 2x1 stack: scatter (top) + time series (bottom).
    # Give the time-series more vertical room so line styles are distinguishable.
    # Also increase hspace so the scatter x-label (SOA_obs) is visible between the stacked axes.
    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[0.62, 0.38], hspace=0.30)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_a_ts = fig.add_subplot(gs_a[1, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.075, right=0.99, top=0.97, bottom=0.09, hspace=0.30, wspace=0.28)

    # Site colors (Nature-friendly, colorblind-safe; Okabe-Ito inspired).
    colors = {"JH": "#0072B2", "CM": "#D55E00"}
    markers = {"JH": "o", "CM": "^"}

    pooled_mask = scatter_df["SOA_obs"].notna() & scatter_df["SOA_pred"].notna()
    x_all = scatter_df.loc[pooled_mask, "SOA_obs"].to_numpy(dtype=float)
    y_all = scatter_df.loc[pooled_mask, "SOA_pred"].to_numpy(dtype=float)
    if int(pooled_mask.sum()) >= 50:
        ax_a.hexbin(
            x_all,
            y_all,
            gridsize=42,
            cmap="Greys",
            mincnt=1,
            linewidths=0.0,
            alpha=0.45,
        )
    # Identify low-density edge points using a coarse 2D histogram.
    if x_all.size >= 50:
        bins = 40
        hist, xedges, yedges = np.histogram2d(x_all, y_all, bins=bins)
        xi = np.clip(np.searchsorted(xedges, x_all, side="right") - 1, 0, bins - 1)
        yi = np.clip(np.searchsorted(yedges, y_all, side="right") - 1, 0, bins - 1)
        # local density threshold is set using pooled occupied bins only.
        occupied = hist[hist > 0]
        thr = np.nanquantile(occupied, 0.35) if occupied.size else 1.0
    for place in ["JH", "CM"]:
        sub = scatter_df[scatter_df["Place"] == place]
        sub_mask = sub["SOA_obs"].notna() & sub["SOA_pred"].notna()
        x = sub.loc[sub_mask, "SOA_obs"].to_numpy(dtype=float)
        y = sub.loc[sub_mask, "SOA_pred"].to_numpy(dtype=float)
        if x.size == 0:
            continue
        # Recreate edge mask for this subset (same binning as pooled).
        if x_all.size >= 50:
            xi = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, bins - 1)
            yi = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, bins - 1)
            local = hist[xi, yi]
            keep = local <= thr
        else:
            keep = np.ones_like(x, dtype=bool)
        ax_a.scatter(
            x[keep],
            y[keep],
            s=26,
            alpha=0.62,
            color=colors[place],
            marker=markers[place],
            edgecolors="none",
            label=place,
        )
    lims = pd.concat([scatter_df["SOA_obs"], scatter_df["SOA_pred"]]).dropna()
    if lims.empty:
        raise RuntimeError("No valid points for Panel A (observed/predicted SOA).")
    vmin, vmax = float(lims.min()), float(lims.max())
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.1
    lo, hi = vmin - pad, vmax + pad
    xs = np.linspace(max(lo, 0.0), hi, 250)
    # Empirical 95% prediction interval in multiplicative error (log10 ratio).
    eps = np.log10(np.clip(y_all, 1e-12, None) / np.clip(x_all, 1e-12, None))
    eps = eps[np.isfinite(eps)]
    if eps.size >= 20:
        q_lo, q_hi = np.nanquantile(eps, [0.025, 0.975])
        f_lo, f_hi = 10.0 ** float(q_lo), 10.0 ** float(q_hi)
        ax_a.fill_between(xs, xs * f_lo, xs * f_hi, color="#bdbdbd", alpha=0.18, linewidth=0.0, zorder=0)
    ax_a.plot([lo, hi], [lo, hi], color="#4f4f4f", linestyle="--", linewidth=1.8, zorder=2)
    ax_a.set_xlim(lo, hi)
    ax_a.set_ylim(lo, hi)
    ax_a.set_aspect("auto")
    # Panel A (top) x-label is drawn at figure-level to avoid being covered by the
    # lower stacked axis patch.
    ax_a.set_xlabel(f"$\\mathbf{{SOA}}_{{\\mathbf{{obs}}}}\\ (\\mu g {DOT} m^{{-3}})$", labelpad=9.0)
    ax_a.set_ylabel(f"$\\mathbf{{SOA}}_{{\\mathbf{{pred}}}}\\ (\\mu g {DOT} m^{{-3}})$")
    # Inset title (inside axes for tighter top spacing).
    ax_a.text(
        0.02,
        0.98,
        "Benchmark: Model 2 (Temp-driven SDE + CS)",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    # Stats annotation in upper-left (no box) to match Nature-like minimal style.
    stats_lines = [
        r"$\bf{Goodness\ of\ fit\ (uncentered)}$",
        rf"Model 2 pooled: $R^2_{{uc}}={r2uc_m2:.2f}$",
        rf"Ref. Model 1 pooled: $R^2_{{uc}}={r2uc_m1:.2f}$",
        rf"Ref. Model 3 pooled: $R^2_{{uc}}={r2uc_m3:.2f}$",
    ]
    ax_a.text(
        0.02,
        0.86,
        "\n".join(stats_lines),
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111111",
    )

    # Panel A (bottom): typical 24-48 h window (here 36 h) time-series to illustrate
    # how Model 2 captures short-term variability beyond pooled scatter R^2.
    place_inset = "JH" if "JH" in per_site else "CM"
    inset_series = per_site.get(place_inset, {})
    df_in = pd.DataFrame(
        {
            "obs": pd.to_numeric(inset_series.get("obs"), errors="coerce"),
            "m1": pd.to_numeric(inset_series.get("pred1"), errors="coerce"),
            "m2": pd.to_numeric(inset_series.get("pred2"), errors="coerce"),
            "m3": pd.to_numeric(inset_series.get("pred3"), errors="coerce"),
        }
    )
    df_in = df_in.replace([np.inf, -np.inf], np.nan).dropna(subset=["obs", "m1", "m2", "m3"], how="any")
    df_win = _select_typical_window(df_in, hours=36.0, min_points=36)
    if not df_win.empty and df_win.shape[0] >= 20:
        t = pd.to_datetime(pd.Index(df_win.index), errors="coerce")
        mask_t = pd.notna(t)
        if int(mask_t.sum()) >= 3:
            t0 = t[mask_t][0]
            df_win = df_win.loc[mask_t].copy()
            dt_h = (t[mask_t] - t0) / np.timedelta64(1, "h")
            xh = np.asarray(dt_h, dtype=float)
        else:
            xh = np.arange(df_win.shape[0], dtype=float)

        # Layering requirement:
        # - Observed baseline at the bottom
        # - Model 2 above Observed
        # - Model 3 on the top
        model_colors = {
            "obs": "#111111",
            "m1": "#E69F00",  # orange
            "m2": "#0072B2",  # blue
            "m3": "#D81B60",  # magenta for contrast
        }
        ax_a_ts.plot(
            xh,
            df_win["obs"].to_numpy(dtype=float),
            color=model_colors["obs"],
            lw=2.8,
            label="Observed",
            zorder=1,
            alpha=1.0,
        )
        ax_a_ts.plot(
            xh,
            df_win["m1"].to_numpy(dtype=float),
            color=model_colors["m1"],
            lw=1.25,
            linestyle=(0, (1.2, 1.2)),
            label="Model 1",
            zorder=2,
            alpha=0.85,
        )
        ax_a_ts.plot(
            xh,
            df_win["m3"].to_numpy(dtype=float),
            color=model_colors["m3"],
            lw=2.00,
            linestyle=(0, (3.2, 1.8)),
            label="Model 3",
            zorder=5,
            alpha=1.0,
        )
        ax_a_ts.plot(
            xh,
            df_win["m2"].to_numpy(dtype=float),
            color=model_colors["m2"],
            lw=1.90,
            label="Model 2",
            zorder=4,
            alpha=0.97,
        )[0].set_path_effects(
            [
                pe.Stroke(linewidth=3.2, foreground="white"),
                pe.Normal(),
            ]
        )
        # Transparent axis patch so Panel A top x-label is not covered.
        ax_a_ts.set_facecolor("none")
        ax_a_ts.patch.set_alpha(0.0)
        ax_a_ts.grid(True, alpha=0.10, linewidth=0.5)
        ax_a_ts.tick_params(width=0.7, length=2.8, labelsize=9, colors="#111111", pad=1.0)
        ax_a_ts.set_xlabel("Time (h)", fontsize=10, fontweight="bold", color="#111111", labelpad=1.0)
        ax_a_ts.set_ylabel(f"$\\mathbf{{SOA}}\\ (\\mu g {DOT} m^{{-3}})$", fontsize=10, fontweight="bold", color="#111111", labelpad=1.0)
        # Legend order matches the visual hierarchy (top-to-bottom emphasis).
        handles_ts, labels_ts = ax_a_ts.get_legend_handles_labels()
        map_ts = {lab: h for h, lab in zip(handles_ts, labels_ts)}
        # Matplotlib fills legend entries column-major when ncol>1. Use an order that
        # visually reads (top row) Model 3, Model 2; (bottom row) Model 1, Observed.
        order_ts = ["Model 3", "Model 1", "Model 2", "Observed"]
        handles_ord = [map_ts[k] for k in order_ts if k in map_ts]
        labels_ord = [k for k in order_ts if k in map_ts]
        leg_ts = ax_a_ts.legend(
            handles_ord,
            labels_ord,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.00, 0.98),
            borderaxespad=0.0,
            ncol=2,
            fontsize=9,
            prop=FP_HELV_BOLD,
            handlelength=2.0,
            columnspacing=0.9,
            labelspacing=0.2,
        )
        if leg_ts is not None:
            for ttxt in leg_ts.get_texts():
                ttxt.set_fontweight("bold")
    _apply_minimal_axis(ax_a_ts)

    handles_a = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.5, markerfacecolor=colors["JH"], markeredgecolor="none", alpha=1.0, label="JH"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=8.0, markerfacecolor=colors["CM"], markeredgecolor="none", alpha=1.0, label="CM"),
    ]
    leg_a = ax_a.legend(
        handles=handles_a,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.2),
        borderaxespad=0.0,
        handletextpad=0.4,
        prop=FP_HELV_BOLD,
    )
    _apply_minimal_axis(ax_a)

    # Panel B

    def _binned_mean_sem(cs_vals: pd.Series, ratio_vals: pd.Series, n_bins: int = 12) -> pd.DataFrame:
        """
        References
        ----------
        - Wilks, D. S. (2011) Statistical Methods in the Atmospheric Sciences.

        Mathematical expression
        -----------------------
        For each CS bin j:
        mean_j = (1/n_j) * sum_i y_i
        sem_j  = sd_j / sqrt(n_j)

        Parameter meanings
        ------------------
        - CS: condensation sink (x-axis).
        - y: modulation factor k_env = SOA / SOA_lin (y-axis).
        - n_j: number of points in bin j.
        """
        df_bin = pd.DataFrame({"cs": pd.to_numeric(cs_vals, errors="coerce"), "ratio": pd.to_numeric(ratio_vals, errors="coerce")})
        df_bin = df_bin.dropna()
        if df_bin.shape[0] < 20:
            return pd.DataFrame()
        # Prefer quantile bins to avoid arbitrary CS units and keep stable sampling.
        n_bins_eff = int(min(max(6, n_bins), max(6, df_bin.shape[0] // 40)))
        try:
            df_bin["qbin"] = pd.qcut(df_bin["cs"], q=n_bins_eff, duplicates="drop")
        except Exception:
            return pd.DataFrame()
        g = df_bin.groupby("qbin", observed=True)
        n = g["ratio"].count().astype(float)
        mean_cs = g["cs"].mean()
        mean_ratio = g["ratio"].mean()
        sd = g["ratio"].std(ddof=1)
        sem = sd / np.sqrt(n)
        out = pd.DataFrame({"cs_mean": mean_cs, "ratio_mean": mean_ratio, "ratio_sem": sem, "n": n})
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["cs_mean", "ratio_mean"])
        return out.sort_values("cs_mean").reset_index(drop=True)

    # Collect pooled CS range for regime shading.
    cs_pool = []
    for place in ["JH", "CM"]:
        entry = diag.get((place, "2_cs"), {})
        cs_v = entry.get("cs")
        ratio_v = entry.get("ratio")
        if cs_v is None or ratio_v is None:
            continue
        cs_pool.append(pd.to_numeric(cs_v, errors="coerce"))
    cs_all = pd.concat(cs_pool, ignore_index=False) if cs_pool else pd.Series(dtype=float)
    cs_all = cs_all.replace([np.inf, -np.inf], np.nan)
    cs_all = cs_all.dropna()
    if cs_all.size >= 50:
        cs_q01, cs_q99 = np.nanquantile(cs_all.to_numpy(dtype=float), [0.02, 0.98])
        cs_q20, cs_q80 = np.nanquantile(cs_all.to_numpy(dtype=float), [0.20, 0.80])
    else:
        cs_q01, cs_q99 = np.nan, np.nan
        cs_q20, cs_q80 = np.nan, np.nan

    # Title inside axes to match Panel A vertical placement.
    ax_b.text(
        0.02,
        0.98,
        "Nonlinear CS modulation (saturation)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )

    # Draw raw scatter (low alpha) + binned means + fitted saturation curves (trim extreme values).
    pooled_ratio_trimmed = []
    pooled_beta_max = []
    cs0_by_place: dict[str, float] = {}
    for place in ["JH", "CM"]:
        key = (place, "2_cs")
        entry = diag.get(key, {})
        cs = entry.get("cs")
        ratio = entry.get("ratio")
        if cs is None or ratio is None:
            continue
        cs = pd.to_numeric(cs, errors="coerce")
        ratio = pd.to_numeric(ratio, errors="coerce")
        m = cs.notna() & ratio.notna()
        if int(m.sum()) < 5:
            continue
        cs_m = cs[m]
        ratio_m = ratio[m]
        if int(cs_m.size) >= 30:
            cs_lo, cs_hi = np.nanquantile(cs_m.to_numpy(dtype=float), [0.02, 0.98])
            r_lo, r_hi = np.nanquantile(ratio_m.to_numpy(dtype=float), [0.02, 0.98])
            m2 = (cs_m >= cs_lo) & (cs_m <= cs_hi) & (ratio_m >= r_lo) & (ratio_m <= r_hi)
            cs_m = cs_m[m2]
            ratio_m = ratio_m[m2]
        if int(cs_m.size) < 5:
            continue
        pooled_ratio_trimmed.append(ratio_m.to_numpy(dtype=float, na_value=np.nan))
        ax_b.scatter(
            cs_m,
            ratio_m,
            s=12,
            alpha=0.16,
            color=colors[place],
            edgecolors="none",
        )
        binned = _binned_mean_sem(cs_m, ratio_m, n_bins=12)
        if not binned.empty:
            ax_b.errorbar(
                binned["cs_mean"].to_numpy(dtype=float),
                binned["ratio_mean"].to_numpy(dtype=float),
                yerr=binned["ratio_sem"].fillna(0.0).to_numpy(dtype=float),
                fmt="o" if place == "JH" else "^",
                color=colors[place],
                ecolor=colors[place],
                elinewidth=1.2,
                capsize=2.5,
                markersize=5.0,
                markerfacecolor=colors[place],
                markeredgecolor="#ffffff",
                markeredgewidth=0.8,
                alpha=0.95,
                zorder=6,
            )
        if place in cs_fit:
            beta_max, cs0, beta_se, cs0_se = cs_fit[place]
            pooled_beta_max.append(float(beta_max))
            cs0_by_place[place] = float(cs0)
            xs = np.linspace(float(cs_m.min()), float(cs_m.max()), 240)
            k_curve = beta_max * xs / (xs + cs0)
            # References
            # ----------
            # - Bevington, P. R. and Robinson, D. K. (2003) Data Reduction and Error Analysis.
            #
            # Mathematical expression
            # -----------------------
            # k(CS) = beta_max * CS / (CS + CS0)
            # Var[k] ~= (dk/dbeta)^2 Var[beta] + (dk/dCS0)^2 Var[CS0], assuming Cov(beta,CS0)=0
            # dk/dbeta = CS / (CS + CS0)
            # dk/dCS0  = -beta_max * CS / (CS + CS0)^2
            #
            # Parameter meanings
            # ------------------
            # - beta_max: saturation level of k_env.
            # - CS0: half-saturation condensation sink.
            # - StdErr: standard error of parameter estimate from the fit.
            if np.isfinite(beta_se) and np.isfinite(cs0_se) and beta_se > 0 and cs0_se > 0:
                dk_db = xs / (xs + cs0)
                dk_dcs0 = -beta_max * xs / np.square(xs + cs0)
                var_k = np.square(dk_db * beta_se) + np.square(dk_dcs0 * cs0_se)
                se_k = np.sqrt(np.maximum(var_k, 0.0))
                ci = 1.96 * se_k
                lo_ci = np.maximum(k_curve - ci, 0.0)
                hi_ci = k_curve + ci
                ax_b.fill_between(xs, lo_ci, hi_ci, color=colors[place], alpha=0.18, linewidth=0.0, zorder=3)
            ax_b.plot(xs, k_curve, color=colors[place], linewidth=2.5, alpha=0.98, zorder=5)
            sat_ls = "--" if place == "JH" else "-."
            ax_b.axhline(beta_max, color="#3a3a3a", linestyle=sat_ls, linewidth=1.8, alpha=0.85, zorder=4)
            if np.isfinite(cs0):
                pass
    ax_b.set_xlabel("CS")
    ax_b.set_ylabel(f"$\\mathbf{{k}}_{{\\mathbf{{env}}}}=\\mathbf{{SOA}}{DOT}\\mathbf{{SOA}}_{{\\mathbf{{lin}}}}^{{-1}}$")
    # Axis limits from pooled trimmed quantiles to reduce outlier-driven whitespace.
    if np.isfinite(cs_q01) and np.isfinite(cs_q99) and cs_q99 > cs_q01:
        ax_b.set_xlim(max(0.0, float(cs_q01)), float(cs_q99))
    if pooled_ratio_trimmed:
        r_all = np.concatenate(pooled_ratio_trimmed)
        r_all = r_all[np.isfinite(r_all)]
        if r_all.size >= 30:
            # References: Wilks (2011). Quantile-based trimming for robust axis limits.
            # Mathematical expression: q_p = Quantile(r, p), ylim = [q_0.02 - m, q_0.98 + m].
            # Parameter meanings: r is k_env values; m is a small padding proportional to the inter-quantile range.
            q_lo, q_hi = np.nanquantile(r_all, [0.02, 0.98])
            span = float(max(q_hi - q_lo, 0.10))
            y_lo = float(max(0.0, q_lo - 0.08 * span))
            y_hi = float(q_hi + 0.08 * span)
            if pooled_beta_max:
                y_hi = max(y_hi, float(max(pooled_beta_max)) * 1.08)
            if np.isfinite(y_hi) and y_hi > y_lo:
                ax_b.set_ylim(y_lo, y_hi)

    # Legend includes fitted curves and saturation levels (beta_max).
    handles_b = [
        Line2D([0], [0], color=colors["JH"], lw=2.5, marker="o", markersize=5, linestyle="-", label="JH fit"),
        Line2D([0], [0], color=colors["CM"], lw=2.5, marker="^", markersize=6, linestyle="-", label="CM fit"),
        Line2D([0], [0], color="#3a3a3a", lw=1.8, linestyle="--", label=r"JH $\boldsymbol{\beta}_{\mathbf{max}}$"),
        Line2D([0], [0], color="#3a3a3a", lw=1.8, linestyle="-.", label=r"CM $\boldsymbol{\beta}_{\mathbf{max}}$"),
    ]
    leg_b = ax_b.legend(handles=handles_b, ncol=2, columnspacing=0.9, **legend_kw)
    if leg_b is not None:
        for t in leg_b.get_texts():
            t.set_fontweight("bold")
    # Parameter annotation above legend (requested).
    param_lines = []
    for place in ["JH", "CM"]:
        if place not in cs_fit:
            continue
        beta_max, cs0, _beta_se, _cs0_se = cs_fit[place]
        param_lines.append(rf"{place}: $\boldsymbol{{\beta}}_{{\mathbf{{max}}}}={beta_max:.2f}$, $\mathbf{{CS_0}}={cs0:.1f}$")
    if param_lines:
        ax_b.text(
            0.98,
            0.93,
            "\n".join(param_lines),
            transform=ax_b.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.6},
        )
    _apply_minimal_axis(ax_b)

    # Panel C1 (Linear drivers)
    ax_c1.set_title("Linear drivers (Model 2 standardized coefficients)", pad=2)
    if coef_df.empty:
        ax_c1.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
        ax_c1.axis("off")
    else:
        # Use slightly lightened site colors for a smoother, less saturated look.
        colors_c1 = {k: _adjust_color_lightness(v, amount=0.78) for k, v in colors.items()}
        coef_df = coef_df.copy()
        coef_df["Label"] = coef_df["Parameter"].apply(_format_feature_label)
        order = (
            coef_df.groupby("Label")["beta_std"]
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .sort_values(ascending=False)
            .index.tolist()
        )
        y_pos = np.arange(len(order))
        offsets = {"JH": -0.18, "CM": 0.18}
        coef_map = coef_df.set_index(["Place", "Label"])
        for place in ["JH", "CM"]:
            sub = coef_df[coef_df["Place"] == place].set_index("Label").reindex(order)
            ax_c1.barh(
                y_pos + offsets[place],
                sub["beta_std"].to_numpy(dtype=float),
                height=0.32,
                color=colors_c1[place],
                alpha=0.98,
                label=place,
                xerr=sub["se_std"].to_numpy(dtype=float),
                error_kw={"elinewidth": 0.9, "ecolor": "#4a4a4a", "capsize": 2.3, "capthick": 0.9},
                edgecolor=colors[place],
                linewidth=0.55,
                hatch=None,
            )
        # Symmetric x-limits for direct comparison of positive vs negative effects.
        max_abs = float(
            np.nanmax(np.abs(coef_df["beta_std"].to_numpy(dtype=float)) + np.abs(coef_df["se_std"].fillna(0.0).to_numpy(dtype=float)))
        )
        max_abs = max_abs if np.isfinite(max_abs) and max_abs > 0 else 1.0
        ax_c1.set_xlim(-1.28 * max_abs, 1.28 * max_abs)
        ax_c1.axvline(0.0, color="#333333", linewidth=1.4)
        ax_c1.set_yticks(y_pos)
        ax_c1.set_yticklabels(order)
        ax_c1.invert_yaxis()
        ax_c1.set_xlabel(r"Standardized coefficient $\boldsymbol{\beta}^{\ast}$ (-)")
        # Numeric labels at bar ends (requested).
        span_x = ax_c1.get_xlim()[1] - ax_c1.get_xlim()[0]
        dx_num = 0.045 * span_x
        for place in ["JH", "CM"]:
            sub = coef_df[coef_df["Place"] == place].set_index("Label").reindex(order)
            vals = sub["beta_std"].to_numpy(dtype=float)
            for i, v in enumerate(vals):
                if not np.isfinite(v):
                    continue
                txt = _format_value_for_label(float(v))
                if not txt:
                    continue
                ax_c1.text(
                    float(v + (dx_num if v >= 0 else -dx_num)),
                    float(y_pos[i] + offsets[place]),
                    txt,
                    ha="left" if v >= 0 else "right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#111111",
                    clip_on=False,
                )
        # Significance stars next to bars (from Model 2 regression p-values).
        dx = 0.015 * span_x
        for idx, label in enumerate(order):
            for place in ["JH", "CM"]:
                try:
                    row = coef_map.loc[(place, label)]
                except Exception:
                    continue
                star = str(row.get("Significance", "")).strip()
                if not star:
                    continue
                beta_val = float(row.get("beta_std"))
                y0 = float(y_pos[idx] + offsets[place])
                x0 = beta_val + (dx if beta_val >= 0 else -dx)
                ax_c1.text(
                    x0,
                    y0,
                    star,
                    ha="left" if beta_val >= 0 else "right",
                    va="center",
                    fontsize=10,
                    color="#111111",
                    fontweight="bold",
                )
        leg_c1 = ax_c1.legend(**legend_kw)
        if leg_c1 is not None:
            for t in leg_c1.get_texts():
                t.set_fontweight("bold")
        _apply_minimal_axis(ax_c1)

    # Panel C2 (Nonlinear interactions)
    ax_c2.set_title("Nonlinear importance (GBDT SHAP, with CS)", pad=2)

    shap_pivot = shap_df.pivot_table(index="feature", columns="Place", values="abs_mean_shap", aggfunc="mean")
    shap_overall = shap_pivot.mean(axis=1, skipna=True)
    order_feat = shap_overall.sort_values(ascending=False).index.tolist()
    y2 = np.arange(len(order_feat), dtype=float)
    bar_vals = shap_overall.reindex(order_feat).to_numpy(dtype=float)
    cat_colors = _category_palette()
    bar_colors = [cat_colors.get(_feature_category(f), cat_colors["other"]) for f in order_feat]
    ax_c2.barh(
        y2,
        bar_vals,
        height=0.55,
        color=bar_colors,
        alpha=1.0,
        edgecolor="none",
        linewidth=0.0,
        zorder=2,
    )
    ax_c2.set_yticks(y2)
    ax_c2.set_yticklabels([_format_feature_label(f) for f in order_feat])
    ax_c2.invert_yaxis()
    ax_c2.set_xlabel(r"Mean $|\mathrm{SHAP}|$ (-)")
    # Numeric labels at bar ends (requested).
    if np.isfinite(bar_vals).any():
        vmax = float(np.nanmax(bar_vals))
        if np.isfinite(vmax) and vmax > 0:
            ax_c2.set_xlim(0.0, vmax * 1.18)
        x0, x1 = ax_c2.get_xlim()
        pad = 0.03 * (x1 - x0)
        x_span = float(np.nanmax(bar_vals) - np.nanmin(bar_vals))
        x_span = x_span if np.isfinite(x_span) and x_span > 0 else float(np.nanmax(bar_vals))
        dx_shap = 0.015 * x_span if x_span > 0 else 0.002
        for y_i, v in zip(y2, bar_vals):
            if not np.isfinite(v):
                continue
            txt = _format_value_for_label(float(v))
            if not txt:
                continue
            x_txt = float(v + dx_shap)
            ha = "left"
            if x_txt > (x1 - pad):
                x_txt = float(max(x0 + pad, v - dx_shap))
                ha = "right"
            ax_c2.text(
                x_txt,
                float(y_i),
                txt,
                ha=ha,
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#111111",
                clip_on=True,
            )
    _apply_minimal_axis(ax_c2)

    # Inset donut: Top-5 SHAP share (pooled across sites), label text on wedges.
    shap_all_pivot = shap_all.pivot_table(index="feature", columns="Place", values="abs_mean_shap", aggfunc="mean")
    shap_all_overall = shap_all_pivot.mean(axis=1, skipna=True).sort_values(ascending=False)
    top5 = shap_all_overall.head(5)
    total = float(shap_all_overall.sum())
    other = max(total - float(top5.sum()), 0.0)
    donut_vals = list(top5.to_numpy(dtype=float)) + [other]
    donut_colors = [cat_colors.get(_feature_category(f), cat_colors["other"]) for f in top5.index.tolist()] + ["#bdbdbd"]
    donut_labels = [_format_feature_label(f) for f in top5.index.tolist()] + ["Others"]
    ax_in = ax_c2.inset_axes([0.56, 0.01, 0.43, 0.70])
    wedges, _ = ax_in.pie(
        donut_vals,
        labels=None,
        colors=donut_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.0},
    )
    # Put labels directly on wedges to reduce visual clutter.
    for wedge, label in zip(wedges, donut_labels):
        ang = 0.5 * (wedge.theta1 + wedge.theta2)
        ang_rad = np.deg2rad(ang)
        span_deg = float(wedge.theta2 - wedge.theta1)
        r_text = 0.86 if span_deg < 28 else 0.80
        fs = 8 if span_deg < 28 else 9
        ax_in.text(
            r_text * float(np.cos(ang_rad)),
            r_text * float(np.sin(ang_rad)),
            label,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color="#111111",
        )
    ax_in.text(0.0, 0.0, "Top 5", ha="center", va="center", fontsize=10, fontweight="bold")
    ax_in.set_aspect("equal")

    # Use a standard x-label for Panel A (top). The lower stacked axis patch is
    # made transparent so the label remains visible in the inter-panel gap.

    out_dir = FIGURE_DIR / "extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.png"
    # Save with a fixed canvas size to keep width/height consistent across figures.
    _force_all_text_helvetica_bold(fig)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

    if not out_path.exists():
        raise RuntimeError(f"Expected output was not created: {out_path}")
    return out_path


def main() -> None:
    out_path = plot_new_fig4()
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    main()
