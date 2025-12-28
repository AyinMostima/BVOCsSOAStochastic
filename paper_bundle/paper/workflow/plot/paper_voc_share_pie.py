from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import ConnectionPatch

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR  # noqa: E402
from paper.workflow.lib.plot_style_helvetica import set_style_helvetica  # noqa: E402
from paper.workflow.lib.voc_share_pie import (  # noqa: E402
    compute_voc_shares_from_grouped,
    load_grouped_voc_inputs,
    top_n_with_other,
)


OUTPUT_SUBDIR = "extra"
HIGHLIGHT: Sequence[str] = ("Isoprene", "1,3-Butadiene", "Butene", "Pentene")


def _highlight_texts(texts, highlight: Sequence[str]) -> None:
    highlight_set = set(highlight)
    for text in texts:
        if text.get_text() in highlight_set:
            text.set_color("#B95756")
            text.set_fontsize(12)
            text.set_weight("bold")


def _plot_pie_only(values, labels, *, out_path: Path, top_n: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax2 = ax.twinx()
    ax2.axis("off")

    values_arr = np.asarray(values, dtype=float)
    startangle = -float(np.argmax(values_arr)) * 360.0 / float(len(values_arr))
    colors = sns.color_palette(palette="BrBG", n_colors=int(top_n) + 5)
    patches, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.2f%%",
        shadow=False,
        startangle=startangle,
        colors=colors[: len(values)],
        labeldistance=1.2,
        pctdistance=0.7,
        radius=0.4,
        wedgeprops=dict(width=0.24),
    )
    ax.axis("equal")

    _highlight_texts(texts, HIGHLIGHT)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=500, bbox_inches="tight")
    plt.close(fig)


def _plot_pie_with_other_bar(
    shares,
    *,
    out_path: Path,
    top_n: int,
) -> None:
    sorted_shares = shares.sort_values(ascending=False)
    labels = sorted_shares.index.tolist()
    sizes = (sorted_shares.values.astype(float)).tolist()

    labels_top = labels[:top_n]
    sizes_top = sizes[:top_n]
    if len(sizes) > top_n:
        labels_top = labels_top + ["Other VOCs"]
        sizes_top = sizes_top + [float(sum(sizes[top_n:]))]

    sizes_other = sizes[top_n:]
    labels_other = labels[top_n:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    fig.subplots_adjust(wspace=0.05, bottom=0.3)

    startangle = 30
    colors = sns.color_palette(palette="vlag", n_colors=min(top_n + 1, len(sizes_top)) + 4)
    patches, texts, autotexts = ax1.pie(
        sizes_top,
        labels=labels_top,
        autopct="%1.2f%%",
        shadow=False,
        startangle=startangle,
        colors=colors[: len(sizes_top)],
        labeldistance=1.2,
        pctdistance=0.9,
        radius=0.4,
        wedgeprops=dict(width=0.24),
    )
    ax1.axis("equal")

    _highlight_texts(texts, HIGHLIGHT)

    width = 0.2
    colors_other = sns.color_palette(palette="BrBG", n_colors=max(len(sizes_other), 1))
    for j, (height, label) in enumerate(zip(sizes_other, labels_other)):
        bc = ax2.bar(j, height, width, color=colors_other[j])
        ax2.bar_label(bc, labels=[f"{height:.2f}"], label_type="edge", fontsize=9, padding=3)

    ax2.set_title("")
    ax2.set_xticks(range(len(labels_other)))
    ax2.set_xticklabels(labels_other, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlim(-1, len(labels_other))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for tick in ax2.get_xticklabels():
        if tick.get_text() in set(HIGHLIGHT):
            tick.set_color("#B95756")
            tick.set_fontsize(12)
            tick.set_weight("bold")

    if sizes_other:
        theta1, theta2 = patches[-1].theta1, patches[-1].theta2
        center, r = patches[-1].center, patches[-1].r

        x = r * np.cos(np.pi / 180.0 * theta2) + center[0]
        y = r * np.sin(np.pi / 180.0 * theta2) + center[1]
        con = ConnectionPatch(
            xyA=(-width / 2.0 - 1.0, float(max(sizes_other))),
            coordsA=ax2.transData,
            xyB=(x, y),
            coordsB=ax1.transData,
        )
        con.set_linestyle("--")
        con.set_linewidth(2.5)
        con.set_edgecolor("#8D6A6E")
        ax2.add_artist(con)

        x = r * np.cos(np.pi / 180.0 * theta1) + center[0]
        y = r * np.sin(np.pi / 180.0 * theta1) + center[1]
        con2 = ConnectionPatch(
            xyA=(-width / 2.0 - 1.0, 0.0),
            coordsA=ax2.transData,
            xyB=(x, y),
            coordsB=ax1.transData,
        )
        con2.set_linestyle("--")
        con2.set_linewidth(2.5)
        con2.set_edgecolor("#8D6A6E")
        ax2.add_artist(con2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    set_style_helvetica(repo_root=BUNDLE_ROOT.parent, savefig_dpi=500)

    df_jh, df_cm = load_grouped_voc_inputs(BUNDLE_ROOT)
    result = compute_voc_shares_from_grouped(df_jh, df_cm)
    shares = result.voc_shares

    out_dir = FIGURE_DIR / OUTPUT_SUBDIR
    labels_1, values_1 = top_n_with_other(shares, top_n=15)
    _plot_pie_only(values_1, labels_1, out_path=out_dir / "VOCs_share.png", top_n=15)
    _plot_pie_with_other_bar(shares, out_path=out_dir / "VOCs_share_detail.png", top_n=14)

    print("[OK] Wrote paper\\\\figure\\\\extra\\\\VOCs_share.png")
    print("[OK] Wrote paper\\\\figure\\\\extra\\\\VOCs_share_detail.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
