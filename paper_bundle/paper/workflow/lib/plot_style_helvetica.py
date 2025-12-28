# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def register_helvetica_fonts(repo_root: Path) -> None:
    """
    Register Helvetica font files shipped with the repository.

    References
    ----------
    - Local font assets: HelveticaNeueLTPro-*.otf at repo root.

    Mathematical expression
    -----------------------
    Not applicable (font registration only).

    Parameter meanings
    ------------------
    - repo_root: repository root path (parent of paper_bundle).
    """
    candidates = [
        repo_root / "HelveticaNeueLTPro-Roman.otf",
        repo_root / "HelveticaNeueLTPro-Bd.otf",
        repo_root / "HelveticaNeueLTPro-It.otf",
        repo_root / "HelveticaNeueLTPro-BdIt.otf",
        repo_root / "Helvetica.ttf",
        repo_root / "helvetica.ttf",
        repo_root
        / "paper_bundle"
        / "paper"
        / "workflow"
        / "lib"
        / "fonts"
        / "texgyreheros"
        / "texgyreheros-regular.otf",
        repo_root
        / "paper_bundle"
        / "paper"
        / "workflow"
        / "lib"
        / "fonts"
        / "texgyreheros"
        / "texgyreheros-bold.otf",
        repo_root
        / "paper_bundle"
        / "paper"
        / "workflow"
        / "lib"
        / "fonts"
        / "texgyreheros"
        / "texgyreheros-italic.otf",
        repo_root
        / "paper_bundle"
        / "paper"
        / "workflow"
        / "lib"
        / "fonts"
        / "texgyreheros"
        / "texgyreheros-bolditalic.otf",
    ]
    for font_file in candidates:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))


def set_style_helvetica(*, repo_root: Path, savefig_dpi: int = 500) -> None:
    """
    Set Matplotlib style to a Helvetica-based, publication-ready configuration.

    References
    ----------
    - Matplotlib rcParams for deterministic typography.

    Mathematical expression
    -----------------------
    Not applicable (plot styling only).

    Parameter meanings
    ------------------
    - savefig_dpi: target DPI for saved figures.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    register_helvetica_fonts(repo_root)

    primary = "Helvetica Neue LT Pro"
    math_font = primary
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [primary, "Helvetica", "Arial", "DejaVu Sans"],
            "font.weight": "bold",
            "mathtext.fontset": "custom",
            "mathtext.rm": math_font,
            "mathtext.it": f"{math_font}:italic",
            "mathtext.bf": f"{math_font}:bold",
            "mathtext.default": "bf",
            "text.usetex": False,
            "axes.unicode_minus": True,
            "figure.dpi": 150,
            "savefig.dpi": int(savefig_dpi),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
