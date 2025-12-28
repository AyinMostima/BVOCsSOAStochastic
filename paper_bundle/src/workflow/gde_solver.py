from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from . import core
from .config import WorkflowConfig, default_config
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class GDEConfig:
    """Configuration for the sectional 0D GDE solver."""

    hf_rule: str = "1min"
    
    # --- Physics / Grid ---
    # NOTE: Grid is now dynamic based on observation columns (M_*um).
    # This cap is used to separate 'Simulation Domain' from 'Background CS'.
    small_bin_max_um: float = 1.0 
    
    # --- Parameters for Optimization ---
    epsilon_cond: float = 0.1
    lambda_gas_s: float = 1.0e-3
    k_prod_bvoc_s: float = 1.0e-5
    loss_rate_s: float = 1.0e-5
    ventilation_factor_s: float = 0.0 
    
    # Nucleation
    d0_nucleation_nm: float = 1.5
    
    # --- Switches ---
    use_coagulation: bool = True
    fit_growth: bool = False
    use_internal_growth: bool = True
    growth_blend: float = 1.0
    
    # --- Constants ---
    temperature_k: float = 298.15
    air_viscosity_pa_s: float = 1.81e-5
    vapor_molar_mass_g_mol: float = 150.0
    
    # --- Advanced ---
    use_timescale_lambda: bool = False
    lambda_gas_cap_s: float = 0.1
    k_env_min: float = 0.05
    k_env_max: float = 3.0
    mtheta_min: float = 0.1
    mtheta_max: float = 10.0
    source_scale_by_place: Dict[str, float] = field(default_factory=lambda: {"CM": 1.0, "JH": 1.0})
    
    # --- Simulation Control ---
    max_steps: int = 1440
    dt_scale: float = 1.0
    spinup_cycles: int = 3 


def _parse_diameter_um(col: str) -> float:
    # Expected format: "M0.025um", "C0.30um", "0.25um" (sometimes mass cols are just number/um)
    # Need to be robust.
    # Remove common prefixes/suffixes
    s = col.replace("M", "").replace("C", "").replace("um", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _build_sectional_grid_from_columns(cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Build simulation grid based on available Mass columns in the data.
    """
    pairs = []
    for c in cols:
        d = _parse_diameter_um(c)
        if np.isfinite(d):
            pairs.append((d, c))
            
    if not pairs:
        raise ValueError("No valid diameter columns found.")
        
    pairs = sorted(pairs, key=lambda tup: tup[0])
    diam_um = np.array([p[0] for p in pairs], dtype=float)
    cols_sorted = [p[1] for p in pairs]
    diam_nm = diam_um * 1000.0
    return diam_nm, cols_sorted


def _resample_high_freq(df: pd.DataFrame, rule: str, cfg: WorkflowConfig) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).resample(rule).asfreq()
    base_dt = pd.to_timedelta(cfg.resample_rule)
    target_dt = pd.to_timedelta(rule)
    max_gap = cfg.short_gap_limit * base_dt
    limit_steps = max(1, int(np.ceil(max_gap / target_dt)))
    numeric = numeric.interpolate(method="time", limit=limit_steps, limit_direction="both")
    out = numeric.copy()
    for col in df.columns:
        if col in numeric.columns:
            continue
        out[col] = df[col].resample(rule).ffill()
    return out


def _compute_cs_from_numbers(
    numbers_conc: np.ndarray,
    diam_nm: np.ndarray,
    cfg: WorkflowConfig,
) -> float:
    if numbers_conc.size == 0: return 0.0
    diameters_m = np.asarray(diam_nm, dtype=float) * 1e-9
    radii_m = diameters_m / 2.0
    lambda_m = cfg.mean_free_path_nm * 1e-9
    kn = 2.0 * lambda_m / diameters_m
    correction = (1 + kn) / (1 + (4.0 / (3.0 * cfg.accommodation_coefficient)) * kn + (4.0 * kn * kn) / (3.0 * cfg.accommodation_coefficient))
    coeff = 4.0 * np.pi * cfg.cs_diffusivity_m2_s * radii_m * correction
    val = float(np.sum(numbers_conc * coeff))
    return val


def _fs_correction(diam_nm: float, cfg: WorkflowConfig) -> float:
    diam_m = diam_nm * 1e-9
    lambda_m = cfg.mean_free_path_nm * 1e-9
    kn = 2.0 * lambda_m / diam_m
    return (1 + kn) / (1 + (4.0 / (3.0 * cfg.accommodation_coefficient)) * kn + (4.0 * kn * kn) / (3.0 * cfg.accommodation_coefficient))


def _brownian_coag_kernel(
    d_i_nm: float,
    d_j_nm: float,
    temp_k: float,
    viscosity_pa_s: float,
    lambda_nm: float,
) -> float:
    k_b = 1.380649e-23
    d_i_m = d_i_nm * 1e-9
    d_j_m = d_j_nm * 1e-9
    d_ij = d_i_m + d_j_m
    lambda_m = lambda_nm * 1e-9
    kn_i = 2.0 * lambda_m / d_i_m
    kn_j = 2.0 * lambda_m / d_j_m
    c_i = 1.0 + kn_i * (1.257 + 0.4 * np.exp(-1.1 / kn_i))
    c_j = 1.0 + kn_j * (1.257 + 0.4 * np.exp(-1.1 / kn_j))
    prefactor = (2.0 * k_b * temp_k) / (3.0 * viscosity_pa_s)
    term = (c_i / d_i_m + c_j / d_j_m) * d_ij
    return prefactor * term


def _map_merge_bin(d_merge_nm: float, bin_centers_nm: np.ndarray) -> int:
    idx = int(np.argmin(np.abs(bin_centers_nm - d_merge_nm)))
    return idx


def _load_mtheta_effects(place: str) -> Dict[str, float]:
    table_path = Path("tables/Table_Mtheta_linear_effects.csv")
    if not table_path.exists():
        return {}
    df = pd.read_csv(table_path)
    if "Place" in df.columns:
        df = df[df["Place"] == place]
    if df.empty or "Feature" not in df.columns or "ScaledEffect" not in df.columns:
        return {}
    effects: Dict[str, float] = {}
    for _, row in df.iterrows():
        feature = row.get("Feature")
        effect = row.get("ScaledEffect")
        if pd.isna(feature) or pd.isna(effect):
            continue
        effects[str(feature)] = float(effect)
    return effects


def _build_env_features_mtheta(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    required = {"rh_pct", "NOx", "SO2", "O3"}
    if not required.issubset(set(frame.columns)):
        return None
    rh = pd.to_numeric(frame.get("rh_pct"), errors="coerce")
    nox = pd.to_numeric(frame.get("NOx"), errors="coerce")
    so2 = pd.to_numeric(frame.get("SO2"), errors="coerce")
    o3 = pd.to_numeric(frame.get("O3"), errors="coerce")
    if "rad_w_m2" in frame.columns:
        rad = pd.to_numeric(frame.get("rad_w_m2"), errors="coerce")
    elif "rad_kw_m2" in frame.columns:
        rad = pd.to_numeric(frame.get("rad_kw_m2"), errors="coerce") * 1000.0
    else:
        rad = None
    env = pd.DataFrame(index=frame.index)
    env["HNO3"] = rh * nox
    env["H2SO4"] = rh * so2
    env["H2SO4O3"] = rh * so2 * o3
    env["HNO3O3"] = rh * nox * o3
    env["O3hv"] = o3 * rad if rad is not None else np.nan
    env["K"] = 1.0
    env["hv"] = rad if rad is not None else np.nan
    return env


def _compute_mtheta_series(place: str, frame: pd.DataFrame) -> Optional[np.ndarray]:
    effects = _load_mtheta_effects(place)
    if not effects:
        return None
    env = _build_env_features_mtheta(frame)
    if env is None:
        return None
    missing = [f for f in effects if f not in env.columns]
    if missing:
        return None
    env_use = env[list(effects.keys())].astype(float)
    if env_use.empty:
        return None
    env_arr = env_use.to_numpy()
    med = np.nanmedian(env_arr, axis=0)
    iqr = np.nanpercentile(env_arr, 75, axis=0) - np.nanpercentile(env_arr, 25, axis=0)
    norm = (env_arr - med) / (iqr + 1e-6)
    coefs = np.array(list(effects.values()), dtype=float)
    score = np.nansum(norm * coefs, axis=1)
    score = np.clip(score, -10.0, 10.0)
    mtheta = np.exp(score)
    mtheta = np.where(np.isfinite(mtheta), mtheta, np.nan)
    return mtheta


def load_high_freq_state(
    cfg: WorkflowConfig, 
    gde_cfg: GDEConfig,
    override_inputs: Dict[str, pd.DataFrame] | None = None
) -> Dict[str, Dict[str, object]]:
    
    if override_inputs:
        df = override_inputs.get("base")
        growth_hf = override_inputs.get("growth")
        japp = override_inputs.get("japp")
    else:
        base_path = Path("intermediate/step01_clean.parquet")
        growth_hf_path = Path("intermediate/step04_growth_metrics_hf.parquet")
        japp_path = Path("intermediate/step05_japp_survival.parquet")
        
        if not base_path.exists(): raise FileNotFoundError("Missing step01_clean.parquet")
        df = pd.read_parquet(base_path)
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time")
        
        growth_hf = pd.read_parquet(growth_hf_path)
        japp = pd.read_parquet(japp_path)

    df = df.sort_index()
    growth_hf = growth_hf.sort_index()
    japp = japp.sort_index()

    hf_df = _resample_high_freq(df, gde_cfg.hf_rule, cfg)
    sites = sorted(hf_df.get("place", pd.Series()).dropna().unique().tolist())
    if not sites: sites = ["ALL"]

    site_payloads = {}
    for place in sites:
        if place == "ALL":
            hf_site = hf_df.copy()
            growth_site = growth_hf.copy()
            japp_site = japp.copy()
        else:
            hf_site = hf_df[hf_df["place"] == place].copy()
            growth_site = growth_hf[growth_hf.get("place", "") == place].copy()
            japp_site = japp[japp.get("place", "") == place].copy()
            
        if hf_site.empty: continue
        t_index = hf_site.index.intersection(growth_site.index).intersection(japp_site.index)
        if t_index.size < 5: continue
        
        hf_site = hf_site.loc[t_index]
        growth_site = growth_site.loc[t_index]
        japp_site = japp_site.loc[t_index]

        joined = hf_site.copy()
        for frame in (growth_site, japp_site):
            cols = [c for c in frame.columns if c not in joined.columns]
            joined[cols] = frame[cols]

        # --- CRITICAL FIX: Separating Domain vs Background ---
        # 1. Identify columns for Mass Observation (Model Target)
        # We look for columns like "0.025um", "0.30um" which contain mass data.
        # Note: We assume mass columns end with "um" and can be parsed.
        all_mass_cols = [c for c in joined.columns if c.endswith("um") and not c.startswith("C")]
        # Filter for domain range
        domain_mass_cols = []
        for c in all_mass_cols:
            d_um = _parse_diameter_um(c)
            if np.isfinite(d_um) and d_um <= gde_cfg.small_bin_max_um:
                domain_mass_cols.append(c)
        
        if not domain_mass_cols:
            # Fallback: if no explicit bins, use M_1_20 proxy logic but this is discouraged now
            # For robustness, let's error out or warn?
            # If we must fallback, we need to build a synthetic grid.
            # But the prompt says "Stop using M_1_20 ... use real PSD".
            # So we assume real PSD columns exist.
            print(f"[Warning] No specific mass bin columns found for {place} <= {gde_cfg.small_bin_max_um}um. Skipping.")
            continue

        # Build grid from data
        diam_nm, sorted_cols = _build_sectional_grid_from_columns(domain_mass_cols)
        mass_obs = joined[sorted_cols].to_numpy(dtype=float)
        # Clean NaNs
        mass_obs = np.nan_to_num(mass_obs, nan=0.0)

        # 2. Identify columns for Background CS (Out of Domain)
        all_number_cols = [c for c in joined.columns if c.startswith("C") and c.endswith("um")]
        bg_number_cols = []
        for c in all_number_cols:
            d_um = _parse_diameter_um(c)
            if np.isfinite(d_um) and d_um > gde_cfg.small_bin_max_um:
                bg_number_cols.append(c)
        
        cs_bg = None
        if bg_number_cols and "temperature_c" in joined.columns:
            cs_bg = core.compute_condensation_sink(
                joined[bg_number_cols],
                joined["temperature_c"],
                cfg.cs_diffusivity_m2_s,
                cfg.mean_free_path_nm,
                cfg.accommodation_coefficient,
            )
        
        # Load forcing
        g_nm_s = pd.to_numeric(joined.get("G_nm_s"), errors="coerce").to_numpy(dtype=float)
        j_app = pd.to_numeric(joined.get("J_app"), errors="coerce").to_numpy(dtype=float)
        s_surv = pd.to_numeric(joined.get("S_surv"), errors="coerce").to_numpy(dtype=float)
        bvocs = pd.to_numeric(joined.get("bvocs"), errors="coerce").to_numpy(dtype=float)
        temp_c = pd.to_numeric(joined.get("temperature_c"), errors="coerce").to_numpy(dtype=float)
        rh_pct = pd.to_numeric(joined.get("rh_pct"), errors="coerce").to_numpy(dtype=float)
        
        rad_w_m2 = pd.to_numeric(joined.get("rad_w_m2"), errors="coerce").to_numpy(dtype=float)
        if np.isnan(rad_w_m2).all() and "rad_kw_m2" in joined.columns:
             rad_w_m2 = pd.to_numeric(joined.get("rad_kw_m2"), errors="coerce").to_numpy(dtype=float) * 1000.0
        
        m_theta_series = _compute_mtheta_series(place, joined)

        site_payloads[place] = {
            "time": t_index,
            "diam_nm": diam_nm,
            "mass_obs": mass_obs,
            "number_obs": np.zeros_like(mass_obs), # Not used for mass balance
            "g_nm_s": g_nm_s,
            "j_app": j_app,
            "s_surv": s_surv,
            "bvocs": bvocs,
            "temperature_c": temp_c,
            "rh_pct": rh_pct,
            "rad_w_m2": rad_w_m2,
            "m_theta": m_theta_series,
            "cs_background": cs_bg.to_numpy(dtype=float) if cs_bg is not None else np.zeros(len(t_index))
        }
    return site_payloads


def condensation_advection_vanleer(
    field_prev: np.ndarray,
    diam_nm: np.ndarray,
    growth_nm_s_vec: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """
    Van Leer advection.
    FIXED: Allow negative growth (evaporation).
    """
    q = np.asarray(field_prev, dtype=float)
    u = np.asarray(growth_nm_s_vec, dtype=float) * dt_s  # displacement
    n = q.size
    if n == 0: return q
    dx = np.diff(diam_nm)
    dx = np.concatenate([dx, dx[-1:]])
    
    flux = np.zeros(n + 1)

    def phi(r):
        return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-12)

    for i_face in range(1, n):
        u_face = 0.5 * (u[i_face - 1] + u[i_face])
        
        # Upwind scheme with Limiter
        if u_face >= 0:
            im1 = max(i_face - 2, 0)
            i0 = i_face - 1
            ip1 = i_face
            
            dq_up = q[i0] - q[im1]
            dq_dn = q[ip1] - q[i0]
            r = dq_up / (dq_dn + 1e-12)
            q_recon = q[i0] + 0.5 * phi(r) * dq_dn
            flux[i_face] = u_face / max(dx[i0], 1e-6) * q_recon
            
        else: # Negative velocity (Evaporation)
            # Simple First Order Upwind for Evaporation
            flux[i_face] = u_face / max(dx[i_face], 1e-6) * q[i_face] # Flow is from i_face to i_face-1

    # Boundaries
    flux[0] = 0.0
    flux[-1] = 0.0
    
    # Update: M_new = M_old - (Flux_out - Flux_in)
    # Flux[i] is flow across left face of cell i. Flux[i+1] is flow across right face.
    out = q - (flux[1:] - flux[:-1])
    out = np.maximum(out, 0.0)
    return out


def _compute_nucleation_mass_increment(
    j_app_t: float,
    s_surv_t: float,
    dt_s: float,
    d0_nm: float,
    density_g_cm3: float,
) -> float:
    volume_m3 = (np.pi / 6.0) * (d0_nm * 1e-9) ** 3
    m_per_particle_ug = density_g_cm3 * 1e3 * volume_m3 * 1e9 
    j_eff = max(j_app_t * s_surv_t, 0.0) 
    delta_n_m3 = j_eff * dt_s * 1e6 # cm-3 to m-3
    return float(delta_n_m3 * m_per_particle_ug)


def _compute_psd_skill(mass_obs: np.ndarray, mass_sim: np.ndarray) -> Tuple[float, np.ndarray]:
    obs_flat = mass_obs.ravel()
    sim_flat = mass_sim.ravel()
    mask = np.isfinite(obs_flat) & np.isfinite(sim_flat)
    if mask.sum() < 3: return np.nan, np.array([])
    
    denom = np.sum((obs_flat[mask] - obs_flat[mask].mean()) ** 2)
    if denom <= 0: return np.nan, np.array([])
    r2 = 1.0 - np.sum((obs_flat[mask] - sim_flat[mask]) ** 2) / denom
    return r2, np.array([])


def run_gde_simulation_full(
    cfg: WorkflowConfig | None = None,
    gde_cfg: GDEConfig | None = None,
    preloaded_inputs: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, Dict[str, object]]:
    
    cfg = cfg or default_config()
    gde_cfg = gde_cfg or GDEConfig()

    if preloaded_inputs:
        site_inputs = preloaded_inputs
    else:
        site_inputs = load_high_freq_state(cfg, gde_cfg)
        
    results = {}

    for place, payload in site_inputs.items():
        # Extract arrays
        time_idx = payload["time"]
        diam_nm = payload["diam_nm"]
        mass_obs = payload["mass_obs"]
        bvocs = payload["bvocs"]
        temp_c = payload["temperature_c"]
        m_theta = payload["m_theta"]
        j_app = payload["j_app"]
        s_surv = payload["s_surv"]
        cs_bg = payload["cs_background"]
        rad = payload.get("rad_w_m2")
        
        n_steps = len(time_idx)
        n_bins = len(diam_nm)
        
        cycles = gde_cfg.spinup_cycles
        total_steps = n_steps * cycles
        
        bvocs_tile = np.tile(bvocs, cycles)
        m_theta_tile = np.tile(m_theta, cycles) if m_theta is not None else np.ones(total_steps)
        j_app_tile = np.tile(j_app, cycles)
        s_surv_tile = np.tile(s_surv, cycles)
        cs_bg_tile = np.tile(cs_bg, cycles)
        rad_tile = np.tile(rad, cycles) if rad is not None else np.zeros(total_steps)
        
        max_rad = np.nanmax(rad) if rad is not None else 1.0
        if max_rad < 1.0: max_rad = 1.0
        
        mass_sim = np.zeros((total_steps, n_bins))
        # Init with first obs or small seed
        mass_sim[0] = mass_obs[0] if len(mass_obs) > 0 else 1e-3
        
        C_gas = np.zeros(total_steps)
        C_gas[0] = 0.1
        
        dt_seconds = 60.0 # Fixed 1min for diurnal
        if gde_cfg.hf_rule == "10s": dt_seconds = 10.0
        dt_seconds *= gde_cfg.dt_scale
        
        lambda_base = gde_cfg.loss_rate_s
        
        # Precalc geometry
        beta_vec = np.array([_fs_correction(d, cfg) for d in diam_nm])
        diam_m = diam_nm * 1e-9
        rho_kg_m3 = cfg.particle_density_g_cm3 * 1000.0
        
        # FIX: Pre-calculate single particle mass in ug correctly
        vol_m3 = (np.pi/6.0) * diam_m**3
        m_particle_ug_const = rho_kg_m3 * vol_m3 * 1e9
        
        for k in range(1, total_steps):
            # Dynamic Ventilation Loss
            curr_rad = rad_tile[k-1] if rad is not None else 0.0
            vent_loss = gde_cfg.ventilation_factor_s * (curr_rad / max_rad)
            total_lambda_loss = lambda_base + vent_loss
            
            # 1. Gas Balance
            # Avoid div by zero
            n_sim_m3 = mass_sim[k-1] / (m_particle_ug_const + 1e-30)
            
            # Compute CS (internal) - Vectorized
            coeff_vec = 4.0 * np.pi * cfg.cs_diffusivity_m2_s * (diam_m / 2.0) * beta_vec
            cs_vec = n_sim_m3 * coeff_vec
            cs_int = np.sum(cs_vec)
            cs_total = cs_int + cs_bg_tile[k-1]
            
            # Source P_bc
            source_scale = gde_cfg.source_scale_by_place.get(place, 1.0)
            P_bc = gde_cfg.k_prod_bvoc_s * source_scale * bvocs_tile[k-1] * m_theta_tile[k-1]
            P_bc = max(P_bc, 0.0)
            
            # Update C_gas
            # dC/dt = P - (lambda + eps*CS)*C
            sink_rate = gde_cfg.lambda_gas_s + gde_cfg.epsilon_cond * cs_total
            C_prev = C_gas[k-1]
            
            # Analytical step
            if sink_rate > 1e-8:
                C_eq = P_bc / sink_rate
                C_new = C_eq + (C_prev - C_eq) * np.exp(-sink_rate * dt_seconds)
                C_avg = C_eq + (C_prev - C_eq) * (1 - np.exp(-sink_rate * dt_seconds)) / (sink_rate * dt_seconds)
            else:
                C_new = C_prev + P_bc * dt_seconds
                C_avg = C_prev + 0.5 * P_bc * dt_seconds
            
            C_gas[k] = max(C_new, 0.0)
            
            # 2. Condensation Mass Flux
            total_flux = gde_cfg.epsilon_cond * cs_total * C_avg 
            dM_total = total_flux * dt_seconds
            
            if cs_total > 0:
                fraction_internal = cs_int / cs_total
                dM_internal = dM_total * fraction_internal
                if cs_int > 0:
                    dist_frac = cs_vec / cs_int
                    mass_gain_bins = dM_internal * dist_frac
                else:
                    mass_gain_bins = np.zeros(n_bins)
            else:
                mass_gain_bins = np.zeros(n_bins)
                
            # 3. Growth Rates for Advection
            C_kg = C_avg * 1e-9
            g_m_s = (4.0 * cfg.cs_diffusivity_m2_s * beta_vec * C_kg) / (rho_kg_m3 * diam_m)
            g_nm_s = g_m_s * 1e9
            
            # 4. Advection
            m_adv = mass_sim[k-1] + mass_gain_bins # Add condensation mass first (operator splitting)
            m_adv = condensation_advection_vanleer(m_adv, diam_nm, g_nm_s, dt_seconds)
            
            # 5. Nucleation (New Particles)
            dM_nuc = _compute_nucleation_mass_increment(
                j_app_tile[k-1], s_surv_tile[k-1], dt_seconds, 
                gde_cfg.d0_nucleation_nm, cfg.particle_density_g_cm3
            )
            m_adv[0] += dM_nuc
            
            # 6. Loss
            m_final = m_adv * np.exp(-total_lambda_loss * dt_seconds)
            mass_sim[k] = m_final

        # Extract Last Cycle
        start_idx = (cycles - 1) * n_steps
        mass_final = mass_sim[start_idx:]
        c_gas_final = C_gas[start_idx:]
        cs_tot_final = cs_bg_tile[start_idx:] # Returning background part for diag
        
        r2, _ = _compute_psd_skill(mass_obs, mass_final)
        
        results[place] = {
            "time": time_idx,
            "diam_nm": diam_nm,
            "mass_obs": mass_obs,
            "mass_sim": mass_final,
            "r2_overall": r2,
            "C_gas": c_gas_final,
            "cs_tot": cs_tot_final
        }
        
    return results

__all__ = [
    "GDEConfig",
    "run_gde_simulation_full",
    "load_high_freq_state"
]
