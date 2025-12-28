from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

SCRIPT_PATH = BUNDLE_ROOT / "scripts" / "BVOCs的拟合效果.py"
SVG_SOURCES = {
    "isoprene_saturation_fits.png": BUNDLE_ROOT / "isoprene_fitting_plots.svg",
    "isoprene_temperature_response.png": BUNDLE_ROOT / "isoprene温度响应.svg",
}
TARGETS = [
    FIGURE_DIR / "isoprene_temperature_response.png",
    FIGURE_DIR / "isoprene_saturation_fits.png",
]


def convert_svg_to_png(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    try:
        import cairosvg  # type: ignore
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "cairosvg"], check=True)
        import cairosvg  # type: ignore
    dest.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(src), write_to=str(dest), dpi=500)


def main() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    subprocess.run([sys.executable, str(SCRIPT_PATH)], cwd=BUNDLE_ROOT, check=True, env=env)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        if not target.exists():
            svg_src = SVG_SOURCES.get(target.name)
            if svg_src is not None:
                convert_svg_to_png(svg_src, target)
    missing = [p for p in TARGETS if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Isoprene figures missing: {missing}")
    print("Isoprene figures available as PNG.")


if __name__ == "__main__":
    sys.exit(main())
