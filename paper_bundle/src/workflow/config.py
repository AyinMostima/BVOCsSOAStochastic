from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Column names stored in the raw data (UTF-8, shown here with unicode escapes for portability)
ISOPRENE_COL = "\u5f02\u620a\u4e8c\u70ef\u6d53\u5ea6"
BUTADIENE_13_COL = "1,3-\u4e01\u4e8c\u70ef\u6d53\u5ea6"
PENTENE_COL = "\u620a\u70ef\u6d53\u5ea6"
BUTENE_COL = "\u4e01\u70ef\u6d53\u5ea6"
TEMPERATURE_COL = "\u5730\u9762\u5c42\u6e29\u5ea6"
HUMIDITY_COL = "\u5730\u9762\u5c42\u6e7f\u5ea6"
RADIATION_COL = "\u5730\u9762\u5c42\u51c0\u8f90\u5c04"

BVOC_COMPONENTS: Tuple[str, str, str, str] = (
    ISOPRENE_COL,
    BUTADIENE_13_COL,
    PENTENE_COL,
    BUTENE_COL,
)


@dataclass(frozen=True)
class WorkflowConfig:
    """Central configuration shared across workflow steps."""

    project_root: Path = Path(".").resolve()
    chemistry_joblib: Path = Path("datanpcm.pkl")
    size_distribution_csv: Path = Path("groupedcmSOA.csv")
    # Multi-site support (site code -> file path). If use_multi_site=True, these will be used.
    use_multi_site: bool = True
    chemistry_joblib_sites: Dict[str, Path] = field(
        default_factory=lambda: {"CM": Path("datanpcm.pkl"), "JH": Path("datanpjh.pkl")}
    )
    size_distribution_csv_sites: Dict[str, Path] = field(
        default_factory=lambda: {"CM": Path("groupedcmSOA.csv"), "JH": Path("groupedjhSOA.csv")}
    )
    resample_rule: str = "1min"
    short_gap_limit: int = 2
    rad_scale_factor: float = 1000.0
    particle_density_g_cm3: float = 1.35
    growth_window_minutes: int = 15
    smoothing_window_minutes: int = 5
    temperature_bin_c: float = 1.0
    min_samples_per_temp_bin: int = 45
    cs_diffusivity_m2_s: float = 5.0e-5
    mean_free_path_nm: float = 65.0
    accommodation_coefficient: float = 1.0
    kappa_grid: Tuple[float, ...] = (
        1.0e-4,
        2.5e-4,
        5.0e-4,
        7.5e-4,
        1.0e-3,
    )
    theta_out_grid: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    eta_grid: Tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
    delta_d_nm: float = 19.0
    random_seed: int = 42
    max_lag_hours: int = 4
    counterfactual_shocks_c: Tuple[float, ...] = (1.5, 2.0, 3.0)
    delta_mass_minutes: float = 1.0 / 6.0  # 10-second horizon
    # Explainability / SHAP configuration
    shap_max_points: int = 2000
    shap_interaction_points: int = 800
    pdp_grid_resolution_1d: int = 40
    pdp_grid_resolution_2d: int = 30
    xgb_params: Dict[str, float | int] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        }
    )
    rf_params: Dict[str, float | int] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 2,
        }
    )

    @property
    def output_dirs(self) -> Tuple[Path, ...]:
        return (
            self.project_root / "figures",
            self.project_root / "tables",
            self.project_root / "intermediate",
            self.project_root / "logs",
            self.project_root / "environment",
            self.project_root / "scripts",
        )

    @property
    def bvoc_columns(self) -> Tuple[str, ...]:
        return BVOC_COMPONENTS

    @property
    def meteorology_columns(self) -> Dict[str, str]:
        return {
            "temperature_c": TEMPERATURE_COL,
            "rh_pct": HUMIDITY_COL,
            "rad_kw_m2": RADIATION_COL,
        }


def default_config() -> WorkflowConfig:
    return WorkflowConfig()
