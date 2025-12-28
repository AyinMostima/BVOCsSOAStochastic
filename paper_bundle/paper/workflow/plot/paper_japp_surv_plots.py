from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import numpy as np
import pandas as pd

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import CHECKPOINT_DIR, FIGURE_DIR  # noqa: E402
from src.workflow.config import default_config  # noqa: E402

HF_RULE = "10s"
HF_SECONDS = int(pd.to_timedelta(HF_RULE).total_seconds())
COUNT_CMAP = "viridis"
SITE_COLORS = {
    "CM": {"japp": "#E64B35", "surv": "#4DBBD5"},
    "JH": {"japp": "#3C5488", "surv": "#00A087"},
}
SINGLE_COLORS = {"japp": "#E64B35", "surv": "#4DBBD5"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue LT Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.weight": "bold",
        # Keep math glyphs consistent with Helvetica.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica Neue LT Pro",
        "mathtext.it": "Helvetica Neue LT Pro:italic",
        "mathtext.bf": "Helvetica Neue LT Pro:bold",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
        "lines.linewidth": 1.2,
    }
)


def _bolden_legend(legend_obj: Legend | None) -> None:
    if legend_obj is None:
        return
    for text in legend_obj.get_texts():
        text.set_fontweight("bold")


def save_dual(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=500, bbox_inches="tight")


def load_japp_survival_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> pd.DataFrame:
    """
    Load the cached high-frequency formation and survival output.

    Returns
    -------
    pd.DataFrame
        Time-indexed dataframe with at least J_app and S_surv columns.
    """
    output = pd.read_parquet(checkpoint_dir / "step05_japp_survival.parquet")
    if "Time" in output.columns:
        output["Time"] = pd.to_datetime(output["Time"])
        output = output.set_index("Time")
    output = output.sort_index()
    if "place" not in output.columns:
        output["place"] = "ALL"
    return output


def maybe_join_growth_metrics(
    output: pd.DataFrame, checkpoint_dir: Path = CHECKPOINT_DIR
) -> pd.DataFrame:
    """
    Join growth-related metrics onto the step05 output if available.
    """
    growth_path = checkpoint_dir / "step04_growth_metrics_hf.parquet"
    if not growth_path.exists():
        return output
    growth_hf = pd.read_parquet(growth_path)
    if "Time" in growth_hf.columns:
        growth_hf["Time"] = pd.to_datetime(growth_hf["Time"])
        growth_hf = growth_hf.set_index("Time")
    growth_hf = growth_hf.sort_index()
    cols_to_join = [c for c in ["CS_star", "G_abs"] if c in growth_hf.columns]
    if not cols_to_join:
        return output
    return output.join(growth_hf[cols_to_join], how="left")


def compute_survival_gate_fit(
    output: pd.DataFrame, *, fallback_eta: float = 1.0
) -> tuple[pd.DataFrame, float, dict[str, float]]:
    """
    Compute survival gate diagnostics and a single-parameter gate curve.

    References
    ----------
    - Kerminen, V.-M. and Kulmala, M. (2002) J. Aerosol Sci.
    - Kulmala, M. et al. (2007) Atmos. Chem. Phys.

    Mathematical expression
    -----------------------
    S_surv = 1 / (1 + eta * CS_star / |G_abs|).

    Parameter meanings
    ------------------
    - S_surv: survival probability (dimensionless).
    - CS_star: effective scavenging term for the growing mode (units consistent with G_abs).
    - G_abs: absolute growth rate proxy (units consistent with CS_star).
    - eta: dimensionless gate efficiency parameter.
    """
    ratio_df = pd.DataFrame()
    metrics = {"gate_r2": np.nan, "gate_rmse": np.nan, "gate_spearman": np.nan}
    best_eta = float(fallback_eta)
    if {"CS_star", "G_abs", "S_surv"}.issubset(output.columns):
        ratio_df = pd.DataFrame(
            {
                "ratio": (output["CS_star"] / output["G_abs"]).replace([np.inf, -np.inf], np.nan),
                "survival": output["S_surv"],
            }
        ).dropna()
        ratio_df = ratio_df[ratio_df["ratio"] > 0]
        if not ratio_df.empty:
            # References: see docstring.
            # Equation: eta = (1/S_surv - 1) / (CS_star/|G_abs|).
            # Parameters: eta is dimensionless; use the median implied eta for a robust single-parameter curve.
            implied_eta = ((1.0 / ratio_df["survival"]) - 1.0) / ratio_df["ratio"]
            implied_eta = implied_eta.replace([np.inf, -np.inf], np.nan).dropna()
            if not implied_eta.empty:
                best_eta = float(implied_eta.median())

            ratio_log = np.log10(ratio_df["ratio"])
            metrics["gate_spearman"] = float(ratio_log.corr(ratio_df["survival"], method="spearman"))

            # References: see docstring.
            # Equation: S_pred = 1 / (1 + eta * ratio).
            # Parameters: best_eta is the median implied eta across samples.
            s_pred = 1.0 / (1.0 + best_eta * ratio_df["ratio"])
            residuals = ratio_df["survival"] - s_pred
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((ratio_df["survival"] - ratio_df["survival"].mean()) ** 2))
            metrics["gate_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            metrics["gate_rmse"] = float(np.sqrt(np.mean(residuals**2)))

    return ratio_df, best_eta, metrics


def select_japp_surv_window(
    output: pd.DataFrame,
    *,
    window: slice | None = None,
    max_gap: pd.Timedelta = pd.Timedelta(minutes=10),
) -> pd.DataFrame:
    """
    Select a contiguous time window suitable for the J_app vs S_surv time-series panel.
    """
    if window is None:
        window = slice("2022-08-06", "2022-08-08")
    window_data = output.loc[window].copy()
    if window_data.empty:
        window_data = output.copy()
    window_data = window_data.dropna(subset=["J_app", "S_surv"], how="any")
    active_mask = (window_data["J_app"].abs() > 1e-6) & (window_data["S_surv"].abs() > 1e-6)
    window_data = window_data.loc[active_mask]
    if window_data.empty:
        window_data = output.dropna(subset=["J_app", "S_surv"], how="any")

    reference_time = window_data.index.min()
    window_data = window_data.assign(
        elapsed_hours=(window_data.index - reference_time).total_seconds() / 3600.0
    )
    # Mark gaps so downstream plotting can break lines cleanly.
    window_data = window_data.assign(gap=window_data.index.to_series().diff().gt(max_gap).fillna(False))
    return window_data


def main() -> None:
    cfg = default_config()
    output = load_japp_survival_checkpoint(CHECKPOINT_DIR)

    freq_seconds = HF_SECONDS

    output = maybe_join_growth_metrics(output, CHECKPOINT_DIR)

    ratio_df = pd.DataFrame()
    best_eta = None
    gate_r2 = np.nan
    gate_rmse = np.nan
    gate_spearman = np.nan
    ratio_df, inferred_eta, gate_metrics = compute_survival_gate_fit(
        output, fallback_eta=(cfg.eta_grid[0] if cfg.eta_grid else 1.0)
    )
    if not ratio_df.empty:
        best_eta = inferred_eta
        gate_r2 = gate_metrics["gate_r2"]
        gate_rmse = gate_metrics["gate_rmse"]
        gate_spearman = gate_metrics["gate_spearman"]
    if best_eta is None:
        best_eta = cfg.eta_grid[0] if cfg.eta_grid else 1.0
    valid_pairs = output[["I_indicator", "delta_mass"]].dropna()
    ratio_log = np.log10(ratio_df["ratio"]) if not ratio_df.empty else pd.Series(dtype=float)

    window_data = select_japp_surv_window(output)

    fig = plt.figure(figsize=(10.8, 6.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.3)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top2 = ax_top.twinx()
    legend_entries: list[tuple[str, str, str]] = []
    if "place" in window_data.columns and window_data["place"].notna().any():
        for place, sub in window_data.groupby("place"):
            if sub.empty:
                continue
            sub = sub.dropna(subset=["J_app", "S_surv"], how="any")
            if sub.empty:
                continue
            gap = sub.index.to_series().diff().gt(pd.Timedelta(minutes=10))
            j_series = sub["J_app"].copy()
            s_series = sub["S_surv"].copy()
            j_series[gap] = np.nan
            s_series[gap] = np.nan
            ehrs = sub["elapsed_hours"]
            colors = SITE_COLORS.get(place, SINGLE_COLORS)
            (line_j,) = ax_top.plot(ehrs, j_series, color=colors["japp"], label=rf"$J_{{\mathrm{{app}}}}$ ({place})", zorder=2)
            (line_s,) = ax_top2.plot(
                ehrs,
                s_series,
                color=colors["surv"],
                linestyle="--",
                alpha=0.25,
                linewidth=1.0,
                zorder=0,
                label=f"$S_{{surv}}$ {place}",
            )
            legend_entries.extend(
                [
                    (fr"$J_{{\mathrm{{app}}}}$ ({place})", colors["japp"], "-"),
                    (fr"$S_{{\mathrm{{surv}}}}$ ({place})", colors["surv"], "--"),
                ]
            )
    else:
        gap = window_data.index.to_series().diff().gt(pd.Timedelta(minutes=10))
        j_series = window_data["J_app"].copy()
        s_series = window_data["S_surv"].copy()
        j_series[gap] = np.nan
        s_series[gap] = np.nan
        line_j = ax_top.plot(window_data["elapsed_hours"], j_series, color=SINGLE_COLORS["japp"], label=r"$J_{\mathrm{app}}$", zorder=2)[0]
        line_s = ax_top2.plot(
            window_data["elapsed_hours"],
            s_series,
            color=SINGLE_COLORS["surv"],
            linestyle="--",
            alpha=0.25,
            linewidth=1.0,
            zorder=0,
            label="$S_{surv}$",
        )[0]
        legend_entries.extend(
            [
                (r"$J_{\mathrm{app}}$", SINGLE_COLORS["japp"], "-"),
                (r"$S_{\mathrm{surv}}$", SINGLE_COLORS["surv"], "--"),
            ]
        )
    if legend_entries:
        proxies: list[Line2D] = []
        labels: list[str] = []
        seen = set()
        for label, color, linestyle in legend_entries:
            if label in seen:
                continue
            seen.add(label)
            proxies.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5))
            labels.append(label)
        leg_top = ax_top.legend(proxies, labels, frameon=False, loc="upper right", ncols=2)
        _bolden_legend(leg_top)
        if leg_top:
            for txt in leg_top.get_texts():
                txt.set_fontsize(9)
    ax_top.set_ylabel(r"$J_{\mathrm{app}}$ (cm$^{-3}$ s$^{-1}$)")
    ax_top2.set_ylabel(r"$S_{\mathrm{surv}}$ (-)", color="#2a2a2a")
    ax_top2.tick_params(axis="y", labelcolor="#2a2a2a")
    if not window_data.empty:
        max_hours = window_data["elapsed_hours"].max()
        ax_top.set_xlim(0, max_hours)
        ax_top.set_ylim(0, window_data["J_app"].quantile(0.99) * 1.1)
        ax_top2.set_ylim(0, window_data["S_surv"].quantile(0.99) * 1.1)
    ax_top.set_xlabel("Elapsed time (h)")
    ax_top.set_title("High-frequency aerosol formation and survival (10 s resolution)")
    ax_top.grid(alpha=0.3, linewidth=0.8)
    # Compress top panel horizontally to align with lower panels and leave room for colorbars.
    top_pos = ax_top.get_position()
    shrink_w = top_pos.width * 0.05
    ax_top.set_position(
        [top_pos.x0 + shrink_w / 2.0, top_pos.y0, top_pos.width - shrink_w, top_pos.height]
    )
    ax_top2.set_position(ax_top.get_position())

    ax_mid = fig.add_subplot(gs[1, 0])
    hb_mid = None
    if not ratio_df.empty:
        hb_mid = ax_mid.hexbin(
            np.log10(ratio_df["ratio"]),
            ratio_df["survival"],
            gridsize=50,
            cmap=COUNT_CMAP,
            mincnt=10,
            norm=LogNorm(),
        )
        x_curve = np.sort(ratio_df["ratio"].to_numpy())
        s_curve = 1.0 / (1.0 + best_eta * x_curve)
        ax_mid.plot(
            np.log10(x_curve),
            s_curve,
            color="#E64B35",
            linewidth=2.4,
            label=r"theory $S_{\mathrm{surv}}$",
        )
        ax_mid.set_xlabel(r"$\log_{10}(\mathrm{CS}^*/|G|)$")
        ax_mid.set_ylabel(r"$S_{\mathrm{surv}}$ (-)")
        ax_mid.set_title("Survival gate as a function of CS*/|G|")
        # Place legend beneath annotation block to avoid overlap.
        leg_mid = ax_mid.legend(loc="upper right", bbox_to_anchor=(0.98, 0.35), frameon=False)
        _bolden_legend(leg_mid)
        if leg_mid:
            for txt in leg_mid.get_texts():
                txt.set_fontsize(9)
        divider_mid = make_axes_locatable(ax_mid)
        cax_mid = divider_mid.append_axes("right", size="3%", pad=0.08)
        cb = fig.colorbar(hb_mid, cax=cax_mid, label="counts")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9)
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        cb.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
        gate_lines = [
            r"$S_{\mathrm{surv}} = \frac{1}{1 + \eta\,\mathrm{CS}^*/|G|}$",
            fr"$\eta={best_eta:.3f}$",
        ]
        if not np.isnan(gate_r2):
            gate_lines.append(f"$R^{2}={gate_r2:.2f}$")
        if not np.isnan(gate_rmse):
            gate_lines.append(f"RMSE={gate_rmse:.3f}")
        if not np.isnan(gate_spearman):
            gate_lines.append(fr"$\rho_s={gate_spearman:.2f}$")
        ax_mid.text(
            0.72,
            0.98,
            "\n".join(gate_lines),
            transform=ax_mid.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75),
        )
    else:
        ax_mid.text(0.5, 0.5, "CS* and G_abs unavailable", ha="center", va="center", fontsize=10)
        ax_mid.set_axis_off()

    ax_bot = fig.add_subplot(gs[1, 1])
    i_limit = max(valid_pairs["I_indicator"].quantile(0.995), 1.0)
    trimmed_pairs = valid_pairs[valid_pairs["I_indicator"] <= i_limit]
    r2_fit = np.nan
    if not trimmed_pairs.empty:
        i_corr = trimmed_pairs["I_indicator"].corr(trimmed_pairs["delta_mass"])
        q1 = trimmed_pairs["I_indicator"].quantile(0.25)
        q3 = trimmed_pairs["I_indicator"].quantile(0.75)
        low_med = trimmed_pairs.loc[trimmed_pairs["I_indicator"] <= q1, "delta_mass"].median()
        high_med = trimmed_pairs.loc[trimmed_pairs["I_indicator"] >= q3, "delta_mass"].median()
        delta_shift = high_med - low_med
        z = np.polyfit(trimmed_pairs["I_indicator"], trimmed_pairs["delta_mass"], 2)
        x_fit = np.linspace(0, trimmed_pairs["I_indicator"].max(), 200)
        y_fit = np.polyval(z, x_fit)
        y_pred_fit = np.polyval(z, trimmed_pairs["I_indicator"])
        ss_res = np.sum((trimmed_pairs["delta_mass"] - y_pred_fit) ** 2)
        ss_tot = np.sum((trimmed_pairs["delta_mass"] - trimmed_pairs["delta_mass"].mean()) ** 2)
        r2_fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        i_corr = np.nan
        delta_shift = np.nan
        x_fit = y_fit = np.array([])
    hb = ax_bot.hexbin(
        trimmed_pairs["I_indicator"],
        trimmed_pairs["delta_mass"],
        gridsize=60,
        cmap=COUNT_CMAP,
        mincnt=5,
        norm=LogNorm(),
    )
    ax_bot.set_xlim(0, trimmed_pairs["I_indicator"].max() * 1.05 if not trimmed_pairs.empty else 1)
    if x_fit.size:
        ax_bot.plot(x_fit, y_fit, color="#E64B35", linewidth=2.4, label="quadratic fit")
    ax_bot.set_xlabel(r"$I(t)=J_{\mathrm{app}}(t)S_{\mathrm{surv}}(t)$")
    ax_bot.set_ylabel(r"$M_{10\mathrm{s}}$ ($\mu$g m$^{-3}$)")
    ax_bot.set_title("Formation-survival driver versus 10 s mass increment")
    if x_fit.size:
        leg = ax_bot.legend(frameon=False, loc="upper left")
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(9)
        _bolden_legend(leg)
    if not np.isnan(i_corr):
        r2_text = f"$R^{2}={r2_fit:.2f}$" if not np.isnan(r2_fit) else "$R^{2}=\\mathrm{nan}$"
        q_text = f"$Q75-Q25={delta_shift:.3f}$"
        ann_text = "\n".join([f"$r={i_corr:.2f}$", r2_text, q_text])
        ax_bot.annotate(
            ann_text,
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75),
        )
    divider_bot = make_axes_locatable(ax_bot)
    cax_bot = divider_bot.append_axes("right", size="3%", pad=0.08)
    cb2 = fig.colorbar(hb, cax=cax_bot, label="counts")
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(labelsize=9)
    cb2.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    cb2.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))

    for ax_obj in [ax_top, ax_top2, ax_mid, ax_bot]:
        ax_obj.tick_params(direction="out", length=3.5, width=0.8)
        for spine in ["top", "right"]:
            spine_obj = ax_obj.spines.get(spine)
            if spine_obj is not None:
                spine_obj.set_visible(False)

    save_dual(fig, "formation_rate_vs_survival")
    plt.close(fig)

    print("Saved Fig05_Japp_Surv to paper\\figure.")


if __name__ == "__main__":
    main()
