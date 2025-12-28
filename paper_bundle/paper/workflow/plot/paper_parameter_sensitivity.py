from __future__ import annotations

import os
import subprocess
import sys

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

SCRIPT_PATH = BUNDLE_ROOT / "scripts" / "参数敏感性.py"
TARGET_NAME = FIGURE_DIR / "SOA_stochastic_anthropogenic_impact.png"


def main() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    subprocess.run([sys.executable, str(SCRIPT_PATH)], cwd=BUNDLE_ROOT, check=True, env=env)
    if not TARGET_NAME.exists():
        raise FileNotFoundError(f"Expected parameter sensitivity figure missing: {TARGET_NAME}")
    print("Parameter sensitivity figure generated as PNG.")


if __name__ == "__main__":
    sys.exit(main())
