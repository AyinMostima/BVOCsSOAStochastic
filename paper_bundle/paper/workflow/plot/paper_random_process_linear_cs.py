from __future__ import annotations

import os
import subprocess
import sys

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

SCRIPT_PATH = BUNDLE_ROOT / "scripts" / "随机过程分析_LINEAR_CS试验.py"
TARGET_NAMES = ["SOA_stochastic_linear_CS_fit.png", "SOA_stochastic_linear_CS.png"]


def main() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    subprocess.run([sys.executable, str(SCRIPT_PATH)], cwd=BUNDLE_ROOT, check=True, env=env)
    missing = [name for name in TARGET_NAMES if not (FIGURE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Expected LINEAR_CS figures missing: {missing}")
    print("Random process (LINEAR_CS) figures generated as PNG.")


if __name__ == "__main__":
    sys.exit(main())
