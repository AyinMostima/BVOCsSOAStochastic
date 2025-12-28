from __future__ import annotations

import shutil

from paper.workflow.lib.paper_paths import FIGURE_DIR, FIGURES_DIR

TARGET_NAMES = {
    "formation_rate_vs_survival.png",
    "isoprene_saturation_fits.png",
    "isoprene_temperature_response.png",
    "SOA_delta_timeseries_combined.png",
    "SOA_extreme_exceedance.png",
    "SOA_linear_model_benchmark.png",
    "SOA_linear_model_cs_scaling.png",
    "SOA_mass_closure_deltaM_vs_I.png",
    "SOA_ML_vs_linear_scatter.png",
    "SOA_SHAP_CM_base.png",
    "SOA_SHAP_CM_with_CS.png",
    "SOA_SHAP_JH_base.png",
    "SOA_SHAP_JH_with_CS.png",
    "SOA_stochastic_anthropogenic_impact.png",
    "SOA_stochastic_linear_CS_fit.png",
    "SOA_stochastic_linear_CS.png",
    "SOA_stochastic_process.png",
    "SOA_timeseries_stationarity.png",
    "VOC_covariance_clusters.png",
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if FIGURES_DIR.exists():
        for name in TARGET_NAMES:
            src = FIGURES_DIR / name
            if src.exists():
                shutil.copy2(src, FIGURE_DIR / name)
    print("Core figure copy step completed (no deletions performed).")


if __name__ == "__main__":
    main()
