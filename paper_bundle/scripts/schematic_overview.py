from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
from pathlib import Path


def _box(ax, xy, text, color="#4e79a7"):
    x, y = xy
    box = FancyBboxPatch((x, y), 1.6, 0.6, boxstyle="round,pad=0.1", linewidth=1.2, edgecolor="black", facecolor=color, alpha=0.8)
    ax.add_patch(box)
    ax.text(x + 0.8, y + 0.3, text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")


def _arrow(ax, start, end, text=None):
    arrow = FancyArrowPatch(start, end, arrowstyle=ArrowStyle("Simple", head_length=6, head_width=6), linewidth=1.1, color="black")
    ax.add_patch(arrow)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.1, text, ha="center", va="bottom", fontsize=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # T -> BVOC SDE (H1)
    _box(ax, (0.5, 4.5), "T driver\nBVOC SDE\n(H1)")
    # M_theta (H2/H3 chem)
    _box(ax, (3.0, 4.5), "M_theta(X)\nSHAP/PDP\n(H2,H3)")
    # Nuc/NPF (H3)
    _box(ax, (5.5, 4.5), "J_app\nNucOX/H2SO4\n(H3)")
    # CS & survival (H4)
    _box(ax, (8.0, 4.5), "CS*, S_surv\n(H4)")
    # Mass pool (H5/H6)
    _box(ax, (5.5, 2.5), "Vapor pool\nK_env(CS)\n(H5)")
    # SOA mass (H7/H8)
    _box(ax, (8.0, 2.5), "SOA mass\nResiduals/Closure\n(H7,H8)")
    # Arrows
    _arrow(ax, (2.1, 4.8), (3.0, 4.8), "BVOC μ/σ")
    _arrow(ax, (4.6, 4.8), (5.5, 4.8), "Chem mod.")
    _arrow(ax, (7.1, 4.8), (8.0, 4.8), "CS gate")
    _arrow(ax, (6.3, 4.2), (6.3, 3.1), "Condensable\nvapors")
    _arrow(ax, (7.9, 4.2), (7.9, 3.1), "Survival")
    _arrow(ax, (6.3, 2.8), (8.0, 2.8), "K_env(CS)")
    _arrow(ax, (8.8, 4.8), (9.5, 4.8), "NPF → CS")
    _arrow(ax, (9.0, 3.0), (9.5, 3.0), "ΔM vs ∫J·S")
    _arrow(ax, (1.3, 4.2), (1.3, 1.2), "Counterfactual\nT→BVOC")
    # Panel labels
    ax.text(0.4, 5.1, "(a)", fontsize=11, fontweight="bold")
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out_dir / f"Fig_schematic_process_overview.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Schematic figure saved.")


if __name__ == "__main__":
    main()
