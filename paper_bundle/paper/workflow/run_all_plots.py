from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

"""
Bundle-friendly regeneration of paper figures using cached checkpoints only.
No directories are cleared and no whitelist-based deletion is performed.
"""

THIS_ROOT = Path(__file__).resolve().parents[2]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

TARGET_PNGS = {
    "formation_rate_vs_survival.png",
    "isoprene_saturation_fits.png",
    "isoprene_temperature_response.png",
    "SOA_delta_timeseries_combined.png",
    "SOA_extreme_exceedance.png",
    "SOA_linear_model_benchmark.png",
    "SOA_linear_model_cs_scaling.png",
    "SOA_mass_closure_deltaM_vs_I.png",
    "SOA_ML_vs_linear_scatter.png",
    "SOA_process_scale_case.png",
    "SOA_SHAP_CM_base.png",
    "SOA_SHAP_CM_with_CS.png",
    "SOA_SHAP_JH_base.png",
    "SOA_SHAP_JH_with_CS.png",
    "SOA_stochastic_anthropogenic_impact.png",
    "SOA_stochastic_linear_CS_fit.png",
    "SOA_stochastic_linear_CS.png",
    "SOA_stochastic_process.png",
    "SOA_timeseries_stationarity.png",
    "Fig3_mechanism_validation.png",
    "Fig2_temperature_amplifies_bvocs.png",
    "VOC_temperature_influence_mode.png",
    "VOC_covariance_clusters.png",
    "Fig5_warming_risks.png",
}


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{BUNDLE_ROOT}{os.pathsep}{existing_path}" if existing_path else str(BUNDLE_ROOT)
    )
    subprocess.run(cmd, cwd=BUNDLE_ROOT, check=True, env=env)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        [sys.executable, "paper\\workflow\\plot\\paper_growth_cs_plots.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_japp_surv_plots.py"],
        [sys.executable, "-c", "from paper.workflow.lib.plotting_paper import plot_from_cache; plot_from_cache()"],
        [sys.executable, "-c", "from paper.workflow.lib.plotting_paper import plot_gbdt_shap_from_cache; plot_gbdt_shap_from_cache()"],
        [sys.executable, "paper\\workflow\\plot\\paper_time_series_stationarity.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_mass_closure_plots.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_fig3_mechanism_validation.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_soa_random_process.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_random_process_linear_cs.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_parameter_sensitivity.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_covariance.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_isoprene_plots.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_fig2_temperature_amplifies_bvocs.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_temperature_influence_mode.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_threshold_analysis.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_new_fig5_risk_mechanism.py"],
        [sys.executable, "paper\\workflow\\plot\\paper_core_plots.py"],
    ]

    for cmd in steps:
        _run(cmd)

    existing_pngs = {p.name for p in FIGURE_DIR.glob("*.png")}
    missing = TARGET_PNGS - existing_pngs
    extras = {name for name in existing_pngs if name not in TARGET_PNGS}
    if missing:
        raise RuntimeError(f"Missing expected figures: {sorted(missing)}")
    if extras:
        raise RuntimeError(f"Unexpected extra PNG outputs: {sorted(extras)}")

    print("All paper figures regenerated from cached results (PNG only, bundle-safe).")


if __name__ == "__main__":
    main()
