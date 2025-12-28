from __future__ import annotations

import subprocess
from pathlib import Path


def run_cmd(args, cwd: Path):
    print(f"[run_all] Running: {' '.join(str(a) for a in args)}")
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(a) for a in args)}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        ["python", "soa_full_pipeline.py"],
        ["python", "scripts/analysis_Mtheta_SHAP.py"],
        ["python", "scripts/feature_importance_vs_linear.py"],
        ["python", "scripts/timescale_analysis.py"],
        ["python", "scripts/bootstrap_cs_params.py"],
        ["python", "scripts/mass_closure.py"],
        ["python", "scripts/h3_gate_analysis.py"],
        ["python", "scripts/residual_diagnostics.py"],
        ["python", "scripts/mtheta_pdp.py"],
        ["python", "scripts/schematic_overview.py"],
        ["python", "scripts/summary_key_numbers.py"],
    ]
    for cmd in scripts:
        run_cmd(cmd, cwd=root)
    print("[run_all] All workflows completed.")


if __name__ == "__main__":
    main()
