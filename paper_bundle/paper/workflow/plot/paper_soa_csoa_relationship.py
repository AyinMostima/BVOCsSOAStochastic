from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR  # noqa: E402
from paper.workflow.lib.particle_size_distribution import (  # noqa: E402
    build_hourmin_means,
    load_grouped_soa_inputs,
)
from paper.workflow.lib.plot_style_helvetica import set_style_helvetica  # noqa: E402
from paper.workflow.lib.soa_csoa_relationship import fit_linear, format_slope_tex  # noqa: E402


TARGET_NAME = "SOA_CSOA_relationship.png"
PALETTE = {"CM": "#934B43", "JH": "#5F97D2"}
OUTPUT_SUBDIR = "extra"


def main() -> int:
    set_style_helvetica(repo_root=BUNDLE_ROOT.parent, savefig_dpi=500)

    df_jh, df_cm = load_grouped_soa_inputs(BUNDLE_ROOT)
    dataall = build_hourmin_means(df_jh, df_cm)

    fig, axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

    residuals_all = []
    for idx, (place, color) in enumerate(PALETTE.items()):
        subset = dataall[dataall["place"] == place]
        x = pd.to_numeric(subset["CSOA"], errors="coerce")
        y = pd.to_numeric(subset["SOA"], errors="coerce")
        valid = x.notna() & y.notna()
        x_valid = x[valid]
        y_valid = y[valid]

        fit, x_fit, y_pred = fit_linear(x_valid, y_valid)
        k_tex = format_slope_tex(fit.slope, decimals=3)

        axs[0].scatter(
            x_fit,
            y_valid.to_numpy(),
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5,
            color=color,
            label=rf"{place} $(k={k_tex})$",
            s=60,
        )
        if x_fit.size > 0:
            axs[0].plot(x_fit, y_pred, color=color, linestyle="--")
        axs[0].text(
            0.05,
            0.75 - 0.15 * idx,
            rf"{place} $R^2$ = {fit.r2:.3f}",
            transform=axs[0].transAxes,
            fontsize=16,
            color="black",
            fontweight="bold",
        )

        resid = y_valid.to_numpy() - y_pred
        residuals_all.append(resid)
        axs[1].scatter(
            x_fit,
            resid,
            alpha=0.03,
            edgecolors="w",
            linewidth=0.5,
            color=color,
            s=60,
        )

    axs[0].set_ylabel(r"SOA ($\mu$g m$^{-3}$)", fontsize=14, fontweight="bold", color="black")
    axs[0].legend(loc="lower right", fontsize=13, frameon=False, prop={"size": 13, "weight": "bold"})

    axs[1].axhline(0.0, color="gray", linestyle="--", lw=2)
    axs[1].set_xlabel("SOA Number (counts)", fontsize=14, fontweight="bold", color="black")
    axs[1].set_ylabel("Residuals", fontsize=14, fontweight="bold", color="black")

    if residuals_all:
        residuals_concat = np.concatenate([r for r in residuals_all if r.size > 0], axis=0)
        if residuals_concat.size > 0:
            r_mean = float(np.mean(residuals_concat))
            r_std = float(np.std(residuals_concat))
            axs[1].set_ylim(r_mean - 5.0 * r_std, r_mean + 5.0 * r_std)

    for ax in axs:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=12, color="black", width=1.5, length=6)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontweight("bold")
            label.set_color("black")

        formatter = ScalarFormatter(useOffset=True, useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.2f}"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.subplots_adjust(hspace=0)
    output_dir = FIGURE_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / TARGET_NAME
    fig.savefig(str(output_path), dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
