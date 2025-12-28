from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Ensure paper_bundle is in sys.path
BUNDLE_ROOT_GUESS = Path(__file__).resolve().parents[3]
if str(BUNDLE_ROOT_GUESS) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT_GUESS))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, CHECKPOINT_DIR, FIGURE_DIR, PAPER_ROOT

SCRIPT_PATH = BUNDLE_ROOT / "scripts" / "mass_closure.py"
TARGET_NAME = "SOA_mass_closure_deltaM_vs_I.png"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    
    # Ensure PYTHONPATH includes BUNDLE_ROOT so 'paper' module is found
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{str(BUNDLE_ROOT)}{os.pathsep}{existing_path}" if existing_path else str(BUNDLE_ROOT)
    )

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--figures-dir",
        str(FIGURE_DIR),
        "--tables-dir",
        str(CHECKPOINT_DIR),
    ]
    subprocess.run(cmd, cwd=BUNDLE_ROOT, check=True, env=env)
    expected = FIGURE_DIR / TARGET_NAME
    if not expected.exists():
        raise FileNotFoundError(f"Expected mass-closure output missing: {expected}")
    print("Mass closure figure regenerated as PNG.")


if __name__ == "__main__":
    sys.exit(main())
