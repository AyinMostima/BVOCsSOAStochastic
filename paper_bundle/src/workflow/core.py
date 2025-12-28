from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from .config import (
    BVOC_COMPONENTS,
    RADIATION_COL,
    TEMPERATURE_COL,
    WorkflowConfig,
)


def ensure_output_dirs(cfg: WorkflowConfig) -> None:
    """Create all required directories."""
    for path in cfg.output_dirs:
        path.mkdir(parents=True, exist_ok=True)


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "Time"
    return df


def load_chemistry_dataframe(path: Path) -> pd.DataFrame:
    obj = joblib.load(path)
    if isinstance(obj, dict):
        frames: List[pd.DataFrame] = []
        for segment in obj.values():
            if isinstance(segment, pd.DataFrame):
                frames.append(_to_datetime_index(segment))
            else:
                frames.append(_to_datetime_index(pd.DataFrame(segment)))
        data = pd.concat(frames, axis=0)
    elif isinstance(obj, pd.DataFrame):
        data = _to_datetime_index(obj)
    else:
        raise TypeError(f"Unsupported container type: {type(obj)!r}")
    return data


def load_size_distribution(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        engine="pyarrow",
        dtype_backend="pyarrow",
        parse_dates=["Datetime"],
    )
    time_cols = [c for c in df.columns if c == "Time"]
    if time_cols:
        df = df.drop(columns=time_cols)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={"Datetime": "Time"})
    df = df.set_index("Time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    keep_cols = [c for c in df.columns if c.endswith("um") or c.startswith("C")]
    return df[keep_cols]


def _rename_core_columns(df: pd.DataFrame, cfg: WorkflowConfig) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    rename_log: List[Dict[str, str]] = []
    rename_map = {
        TEMPERATURE_COL: "temperature_c",
        cfg.meteorology_columns["rh_pct"]: "rh_pct",
        cfg.meteorology_columns["rad_kw_m2"]: "rad_kw_m2",
    }
    out = df.copy()
    for old, new in rename_map.items():
        if old in out.columns:
            out[new] = out[old]
            rename_log.append(
                {
                    "variable": new,
                    "original_column": old,
                    "operation": "renamed",
                    "source_unit": "",
                    "target_unit": "",
                }
            )
    return out, rename_log


def _build_unit_log(entries: List[Dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(entries)


def _compute_bvocs(df: pd.DataFrame) -> pd.Series:
    missing = [col for col in BVOC_COMPONENTS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing BVOC components: {missing}")
    return df[list(BVOC_COMPONENTS)].sum(axis=1, min_count=1)


def _interpolate_short_gaps(series: pd.Series, limit: int) -> pd.Series:
    return series.interpolate(limit=limit, limit_direction="both")


def _resample(df: pd.DataFrame, rule: str, limit: int) -> pd.DataFrame:
    resampled = df.resample(rule).mean()
    for column in resampled.columns:
        resampled[column] = _interpolate_short_gaps(resampled[column], limit)
    return resampled


def _split_size_columns(columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    mass_cols = [c for c in columns if c.endswith("um") and not c.startswith("C")]
    number_cols = [c for c in columns if c.startswith("C") and c.endswith("um")]
    return mass_cols, number_cols


def _column_to_um(name: str) -> float:
    stripped = name.replace("C", "").replace("um", "")
    return float(stripped)


def aggregate_small_bins(
    df: pd.DataFrame,
    mass_cols: Sequence[str],
    number_cols: Sequence[str],
    max_um: float = 0.40,
) -> pd.DataFrame:
    mass_subset = [c for c in mass_cols if _column_to_um(c) <= max_um]
    number_subset = [c for c in number_cols if _column_to_um(c) <= max_um]
    out = df.copy()
    if mass_subset:
        out["M_1_20"] = out[mass_subset].sum(axis=1, min_count=1)
    if number_subset:
        out["N_1_20"] = out[number_subset].sum(axis=1, min_count=1)
    return out


def prepare_master_dataframe(cfg: WorkflowConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Single-site fallback
    if not getattr(cfg, "use_multi_site", False):
        chem_df = load_chemistry_dataframe(cfg.chemistry_joblib)
        chem_df, rename_entries = _rename_core_columns(chem_df, cfg)
        chem_df["rad_w_m2"] = chem_df["rad_kw_m2"] * cfg.rad_scale_factor
        rename_entries.append(
            {
                "variable": "rad_w_m2",
                "original_column": cfg.meteorology_columns["rad_kw_m2"],
                "operation": "scale_by_1000",
                "source_unit": "kW m^-2",
                "target_unit": "W m^-2",
            }
        )
        chem_df["bvocs"] = _compute_bvocs(chem_df)
        rename_entries.append(
            {
                "variable": "bvocs",
                "original_column": "+".join(BVOC_COMPONENTS),
                "operation": "sum",
                "source_unit": "μg m^-3",
                "target_unit": "μg m^-3",
            }
        )
        keep_cols = [
            "bvocs",
            "SO2",
            "NOx",
            "O3",
            "temperature_c",
            "rh_pct",
            "rad_w_m2",
        ]
        chem_df = chem_df[keep_cols]
        chem_df = _resample(chem_df, cfg.resample_rule, cfg.short_gap_limit)

        size_df = load_size_distribution(cfg.size_distribution_csv)
        size_df = _resample(size_df, cfg.resample_rule, cfg.short_gap_limit)

        mass_cols, number_cols = _split_size_columns(size_df.columns)
        merged = chem_df.join(size_df, how="inner")
        merged = aggregate_small_bins(merged, mass_cols, number_cols)
        merged = merged.dropna(subset=["M_1_20", "N_1_20"], how="any")

        unit_log = _build_unit_log(rename_entries)
        bin_metadata = pd.DataFrame(
            {
                "bin": mass_cols + number_cols,
                "type": ["mass"] * len(mass_cols) + ["number"] * len(number_cols),
                "diameter_um": [ _column_to_um(c) for c in mass_cols + number_cols ],
            }
        )
        return merged, unit_log, bin_metadata

    # Multi-site branch
    combined_rows: List[pd.DataFrame] = []
    unit_log_entries: List[Dict[str, str]] = []
    bin_meta_frames: List[pd.DataFrame] = []
    for place, chem_path in cfg.chemistry_joblib_sites.items():
        chem_df = load_chemistry_dataframe(chem_path)
        chem_df, rename_entries = _rename_core_columns(chem_df, cfg)
        chem_df["rad_w_m2"] = chem_df["rad_kw_m2"] * cfg.rad_scale_factor
        unit_log_entries.extend(
            [
                {
                    "variable": "rad_w_m2",
                    "original_column": cfg.meteorology_columns["rad_kw_m2"],
                    "operation": f"scale_by_1000 ({place})",
                    "source_unit": "kW m^-2",
                    "target_unit": "W m^-2",
                }
            ]
        )
        chem_df["bvocs"] = _compute_bvocs(chem_df)
        unit_log_entries.append(
            {
                "variable": f"bvocs ({place})",
                "original_column": "+".join(BVOC_COMPONENTS),
                "operation": "sum",
                "source_unit": "μg m^-3",
                "target_unit": "μg m^-3",
            }
        )
        keep_cols = [
            "bvocs",
            "SO2",
            "NOx",
            "O3",
            "temperature_c",
            "rh_pct",
            "rad_w_m2",
        ]
        chem_df = chem_df[keep_cols]
        chem_df = _resample(chem_df, cfg.resample_rule, cfg.short_gap_limit)

        size_path = cfg.size_distribution_csv_sites.get(place)
        if size_path is None:
            continue
        size_df = load_size_distribution(size_path)
        size_df = _resample(size_df, cfg.resample_rule, cfg.short_gap_limit)
        mass_cols, number_cols = _split_size_columns(size_df.columns)
        merged = chem_df.join(size_df, how="inner")
        merged = aggregate_small_bins(merged, mass_cols, number_cols)
        merged = merged.dropna(subset=["M_1_20", "N_1_20"], how="any")
        merged["place"] = place
        combined_rows.append(merged)

        bin_meta_frames.append(
            pd.DataFrame(
                {
                    "bin": mass_cols + number_cols,
                    "type": ["mass"] * len(mass_cols) + ["number"] * len(number_cols),
                    "diameter_um": [ _column_to_um(c) for c in mass_cols + number_cols ],
                    "place": place,
                }
            )
        )

    if not combined_rows:
        raise RuntimeError("No site data could be combined. Check config paths.")
    merged_all = pd.concat(combined_rows).sort_index()
    unit_log = _build_unit_log(unit_log_entries)
    bin_metadata = pd.concat(bin_meta_frames, ignore_index=True)
    return merged_all, unit_log, bin_metadata


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "bvocs": "Sum of isoprene + 1,3-butadiene + pentene + butene concentrations",
        "SO2": "Sulfur dioxide mixing ratio (proxy for sulfuric acid production)",
        "NOx": "Oxides of nitrogen mixing ratio",
        "O3": "Ozone mixing ratio",
        "temperature_c": "Near-surface air temperature in degrees Celsius",
        "rh_pct": "Relative humidity %",
        "rad_w_m2": "Net radiation converted from kW/m^2 to W/m^2",
        "M_1_20": "Mass concentration proxy for 1-20 nm particles (μg/m^3)",
        "N_1_20": "Number concentration proxy for 1-20 nm particles (cm^-3)",
    }
    units = {
        "bvocs": "μg m^-3",
        "SO2": "ppb",
        "NOx": "ppb",
        "O3": "ppb",
        "temperature_c": "°C",
        "rh_pct": "%",
        "rad_w_m2": "W m^-2",
        "M_1_20": "μg m^-3",
        "N_1_20": "cm^-3",
    }
    rows = []
    for column in descriptions:
        if column in df.columns:
            rows.append(
                {
                    "field": column,
                    "description": descriptions[column],
                    "unit": units.get(column, ""),
                }
            )
    return pd.DataFrame(rows)


# Reference: Kulmala et al. (2007) Atmos. Chem. Phys.; Equation: m_p = M/N, d_eff = ((6 m_p)/(pi rho_p))^(1/3); Parameters: M is mass concentration (μg m^-3), N is number concentration (cm^-3) converted to m^-3, rho_p is particle density (g cm^-3).
def compute_effective_diameter(
    mass: pd.Series,
    number: pd.Series,
    density_g_cm3: float,
) -> pd.Series:
    mass_g_m3 = mass * 1e-6
    number_m3 = number * 1e6
    mass_per_particle = mass_g_m3 / number_m3
    volume = (mass_per_particle) / (density_g_cm3 * 1e3)
    diameter_m = ((6 * volume) / np.pi) ** (1.0 / 3.0)
    return pd.Series(diameter_m * 1e9, index=mass.index, name="d_eff_nm")


# Reference: Kulmala et al. (2013) Nat. Prot.; Equation: G = d(d_eff)/dt; Parameters: d_eff is effective diameter (nm), t is time (s); uses centered finite differences.
def compute_growth_rate(diameter_nm: pd.Series, freq_minutes: int = 1) -> pd.Series:
    dt_seconds = freq_minutes * 60
    growth = diameter_nm.diff() / dt_seconds
    return growth.rolling(window=5, min_periods=1, center=True).median().rename("G_nm_s")


# Reference: Fuchs and Sutugin (1971); Equation: CS = sum_i 4 pi D_v r_i F(Kn_i) N_i; Parameters: D_v vapor diffusivity (m^2 s^-1), r_i particle radius (m), Kn_i = 2 lambda / d_i, N_i number concentration (cm^-3 converted to m^-3), lambda mean free path (m).
def compute_condensation_sink(
    number_df: pd.DataFrame,
    temperature_c: pd.Series,
    diffusivity_m2_s: float,
    mean_free_path_nm: float,
    accommodation: float = 1.0,
) -> pd.Series:
    diameters_m = np.array([_column_to_um(col) * 1e-6 for col in number_df.columns])
    radii_m = diameters_m / 2.0
    lambda_m = mean_free_path_nm * 1e-9
    kn = 2.0 * lambda_m / diameters_m
    correction = (1 + kn) / (1 + (4.0 / (3.0 * accommodation)) * kn + (4.0 * kn * kn) / (3.0 * accommodation))
    coeff = 4.0 * np.pi * diffusivity_m2_s * radii_m * correction
    number_m3 = number_df * 1e6
    cs = number_m3.mul(coeff, axis=1).sum(axis=1)
    return cs.rename("CS_star")


# Reference: Guenther et al. (1993) JGR; Equation: dC = μ(T) dt + σ(T) dW_t; Parameters: μ drift (μg m^-3 min^-1), σ diffusion amplitude, T temperature (°C), W_t Wiener process.
def estimate_temperature_response(
    series: pd.Series,
    temperature: pd.Series,
    bin_width: float,
    min_count: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"value": series, "temp": temperature}).dropna()
    df["temp_bin"] = (df["temp"] / bin_width).round() * bin_width
    grouped = df.groupby("temp_bin")
    records = []
    for temp, block in grouped:
        if block.shape[0] < min_count:
            continue
        mu = block["value"].mean()
        sigma = block["value"].std(ddof=1)
        standardized = (block["value"] - mu) / (sigma if sigma else 1.0)
        hist, edges = np.histogram(standardized, bins=20, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        normal_pdf = stats.norm.pdf(centers)
        kld = stats.entropy(hist + 1e-9, normal_pdf + 1e-9)
        records.append(
            {
                "temp_bin": temp,
                "mu": mu,
                "sigma": sigma,
                "count": block.shape[0],
                "kld": kld,
            }
        )
    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        return stats_df
    for target in ("mu", "sigma"):
        slope, intercept, r_value, p_value, _ = stats.linregress(stats_df["temp_bin"], stats_df[target])
        stats_df[f"{target}_slope"] = slope
        stats_df[f"{target}_pvalue"] = p_value
        stats_df[f"{target}_r2"] = r_value**2
    return stats_df


# Reference: Gardiner (1985) Handbook of Stochastic Methods; Equation: X_{t+Δt} = X_t + μ(T_t) Δt + σ(T_t) sqrt(Δt) ξ; Parameters: μ drift, σ diffusion, ξ ~ N(0,1), Δt time step in minutes.
def simulate_sde_paths(
    mu_series: pd.Series,
    sigma_series: pd.Series,
    steps: int,
    n_paths: int,
    delta_minutes: float,
    seed: int,
    initial_value: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = mu_series.to_numpy()
    sigma = sigma_series.to_numpy()
    dt = delta_minutes
    paths = np.zeros((n_paths, steps))
    paths[:, 0] = initial_value
    for i in range(1, steps):
        drift = mu[min(i - 1, len(mu) - 1)] * dt
        diffusion = sigma[min(i - 1, len(sigma) - 1)] * np.sqrt(dt) * rng.standard_normal(n_paths)
        proposal = paths[:, i - 1] + drift + diffusion
        paths[:, i] = np.clip(proposal, a_min=0, a_max=None)
    return paths
