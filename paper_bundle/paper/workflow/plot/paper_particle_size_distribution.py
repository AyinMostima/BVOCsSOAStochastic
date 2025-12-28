from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR  # noqa: E402
from paper.workflow.lib.particle_size_distribution import (  # noqa: E402
    CONCENTRATION_COLUMNS,
    build_compressed_axis,
    build_hourmin_means,
    load_grouped_soa_inputs,
    to_long_concentration,
)
from paper.workflow.lib.plot_style_helvetica import set_style_helvetica  # noqa: E402


TARGET_NAME = "Particle_size_distribution.png"
PALETTE = {"CM": "#934B43", "JH": "#5F97D2"}
OUTPUT_SUBDIR = "extra"


def main() -> int:
    set_style_helvetica(repo_root=BUNDLE_ROOT.parent, savefig_dpi=500)

    df_jh, df_cm = load_grouped_soa_inputs(BUNDLE_ROOT)
    dataall = build_hourmin_means(df_jh, df_cm)
    data_long = to_long_concentration(dataall)

    size_um = [float(c[1:-2]) for c in CONCENTRATION_COLUMNS]
    axis = build_compressed_axis(size_um, power=0.25, tick_limit_um=0.65)
    data_long["compressed_size"] = data_long["size_um"].map(axis.mapping)

    mean_long = (
        data_long.groupby(["compressed_size", "place"], as_index=False)["concentration"]
        .mean()
        .sort_values(["place", "compressed_size"])
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.lineplot(
        data=data_long,
        x="compressed_size",
        y="concentration",
        hue="place",
        style="place",
        marker="o",
        palette=PALETTE,
        linewidth=3.5,
        errorbar=None,
        ax=ax,
    )

    for place, color in PALETTE.items():
        subset = mean_long[mean_long["place"] == place]
        if subset.empty:
            continue
        ax.fill_between(subset["compressed_size"], subset["concentration"], color=color, alpha=0.15)

    ax.set_xticks(axis.tick_positions)
    ax.set_xticklabels(axis.tick_labels, rotation=90, ha="right", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.text(
        axis.tick_positions[-1],
        -0.18,
        axis.tail_label,
        transform=ax.get_xaxis_transform(),
        ha="right",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlabel(r"Particle Size ($\mu$m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number Concentration", fontsize=14, fontweight="bold")
    ax.set_title("", fontsize=16, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(title="", loc="upper right", ncol=1, frameon=False, fontsize=20)

    fig.tight_layout()
    output_dir = FIGURE_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / TARGET_NAME
    fig.savefig(str(output_path), dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
