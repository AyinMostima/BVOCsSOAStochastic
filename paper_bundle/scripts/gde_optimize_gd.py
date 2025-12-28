from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow.config import default_config  # noqa: E402
from src.workflow.gde_solver import GDEConfig, run_gde_simulation_full, load_high_freq_state  # noqa: E402

CHECKPOINT_PATH = Path("intermediate/gde_opt_checkpoint.json")

@dataclass
class OptimizationState:
    stage: int
    iteration: int
    params: List[float]
    best_loss: float
    history_loss: List[float]
    history_params: List[List[float]]
    
    @classmethod
    def from_dict(cls, data: Dict) -> OptimizationState:
        return cls(
            stage=data["stage"],
            iteration=data["iteration"],
            params=data["params"],
            best_loss=data["best_loss"],
            history_loss=data["history_loss"],
            history_params=data["history_params"]
        )

def aggregate_by_hour_min(df: pd.DataFrame) -> pd.DataFrame:
    if "place" not in df.columns:
        return df
    work = df.copy()
    if "Time" in work.columns:
        work["Time"] = pd.to_datetime(work["Time"])
        work = work.set_index("Time")
    
    if not isinstance(work.index, pd.DatetimeIndex):
        return df 

    work["hour_min"] = work.index.strftime("%H:%M")
    numeric_cols = work.select_dtypes(include=[np.number]).columns
    grouped = work.groupby(["place", "hour_min"])[numeric_cols].mean().reset_index()
    
    base_date = pd.to_datetime("2000-01-01")
    grouped["minutes_since_midnight"] = grouped["hour_min"].apply(
        lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
    )
    grouped = grouped.sort_values(["place", "minutes_since_midnight"])
    place_offsets = {p: i for i, p in enumerate(grouped["place"].unique())}
    grouped["Time"] = grouped.apply(
        lambda row: base_date
        + pd.Timedelta(days=place_offsets.get(row["place"], 0))
        + pd.to_timedelta(row["minutes_since_midnight"], unit="m"),
        axis=1,
    )
    grouped = grouped.drop(columns=["minutes_since_midnight"])
    grouped = grouped.set_index("Time")
    return grouped

def _compute_loss(results: Dict[str, Dict[str, object]]) -> Tuple[float, Dict[str, float]]:
    losses: List[float] = []
    r2_sites: Dict[str, float] = {}
    debug_printed = False
    
    for place, payload in results.items():
        m_obs = payload["mass_obs"]
        m_sim = payload["mass_sim"]
        tot_obs = np.nansum(m_obs, axis=1)
        tot_sim = np.nansum(m_sim, axis=1)
        
        mask = np.isfinite(tot_obs) & np.isfinite(tot_sim)
        if mask.sum() < 10: continue
            
        obs = tot_obs[mask]
        sim = tot_sim[mask]
        
        if not debug_printed:
            ratio = sim.mean() / obs.mean() if obs.mean() > 0 else 0
            print(f"    [Diag {place}] Mean Obs: {obs.mean():.3f} | Mean Sim: {sim.mean():.3f} | Ratio: {ratio:.3f}")
            debug_printed = True
        
        # MSLE
        log_obs = np.log1p(np.maximum(obs, 0.0))
        log_sim = np.log1p(np.maximum(sim, 0.0))
        msle = np.mean((log_obs - log_sim) ** 2)
        losses.append(msle)
        
        denom = np.sum((obs - obs.mean()) ** 2)
        r2 = np.nan
        if denom > 0:
            r2 = 1.0 - np.sum((obs - sim) ** 2) / denom
        r2_sites[place] = r2
        
    if not losses: return 1e9, r2_sites
    return float(np.mean(losses)), r2_sites

def _build_cfg_from_params(vector: np.ndarray, base_cfg: GDEConfig, max_steps: int) -> GDEConfig:
    return GDEConfig(
        hf_rule="1min",
        small_bin_max_um=base_cfg.small_bin_max_um,
        n_small_bins=base_cfg.n_small_bins,
        
        # Optimized Parameters
        epsilon_cond=vector[0],
        lambda_gas_s=vector[1],
        k_prod_bvoc_s=vector[2],
        loss_rate_s=vector[3],
        ventilation_factor_s=vector[4], # NEW
        
        d0_nucleation_nm=base_cfg.d0_nucleation_nm,
        use_coagulation=base_cfg.use_coagulation,
        temperature_k=base_cfg.temperature_k,
        air_viscosity_pa_s=base_cfg.air_viscosity_pa_s,
        fit_growth=base_cfg.fit_growth,
        max_steps=max_steps,
        dt_scale=base_cfg.dt_scale,
        use_internal_growth=base_cfg.use_internal_growth,
        growth_blend=base_cfg.growth_blend,
        use_timescale_lambda=base_cfg.use_timescale_lambda,
        lambda_gas_cap_s=base_cfg.lambda_gas_cap_s,
        k_env_min=base_cfg.k_env_min,
        k_env_max=base_cfg.k_env_max,
        mtheta_min=base_cfg.mtheta_min,
        mtheta_max=base_cfg.mtheta_max,
        source_scale_by_place=base_cfg.source_scale_by_place,
        spinup_cycles=base_cfg.spinup_cycles
    )

def _run_simulation_task(params, base_cfg, cfg_project, max_steps, preloaded_data):
    try:
        cfg_p = _build_cfg_from_params(params, base_cfg, max_steps)
        res_p = run_gde_simulation_full(cfg_project, cfg_p, preloaded_inputs=preloaded_data)
        loss_p, _ = _compute_loss(res_p)
        return loss_p
    except Exception as e:
        return 1e9

def _finite_diff_grad(params, steps, base_cfg, cfg_project, max_steps, preloaded_data):
    grads = np.zeros_like(params)
    cfg_base = _build_cfg_from_params(params, base_cfg, max_steps)
    res_base = run_gde_simulation_full(cfg_project, cfg_base, preloaded_inputs=preloaded_data)
    loss_base, r2_sites = _compute_loss(res_base)

    perturbed_list = []
    for i, h in enumerate(steps):
        p_new = params.copy()
        p_new[i] = max(p_new[i] + h, 1e-9)
        perturbed_list.append(p_new)

    losses_perturbed = Parallel(n_jobs=-1, backend="loky")(
        delayed(_run_simulation_task)(p, base_cfg, cfg_project, max_steps, preloaded_data) 
        for p in perturbed_list
    )

    for i, loss_p in enumerate(losses_perturbed):
        grads[i] = (loss_p - loss_base) / steps[i]

    return grads, loss_base, r2_sites

def _clip_params(p: np.ndarray) -> np.ndarray:
    bounds = np.array([
        [1e-3, 2.0],      # epsilon
        [1e-4, 0.1],      # lambda_gas
        [1e-6, 1e-2],     # k_prod
        [1e-7, 1e-3],     # loss_rate
        [0.0,  1e-2],     # ventilation (NEW)
    ])
    return np.clip(p, bounds[:, 0], bounds[:, 1])

def _save_checkpoint(state: OptimizationState):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(asdict(state), f, indent=2)

def _load_checkpoint() -> Optional[OptimizationState]:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                data = json.load(f)
            return OptimizationState.from_dict(data)
        except Exception:
            return None
    return None

def _plot_history(state: OptimizationState, param_names: List[str]):
    if len(state.history_loss) < 2: return
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(state.history_loss, 'o-', color='#d95f02', label='MSLE Loss')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.set_title(f'Optimization (Best: {state.best_loss:.4f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    p0 = state.history_params[0]
    norm_params = np.array(state.history_params) / (np.array(p0) + 1e-12)
    markers = ['s', '^', 'd', 'v', 'x'] # Added marker for 5th param
    for i, name in enumerate(param_names):
        ax2.plot(norm_params[:, i], marker=markers[i % 5], label=name, alpha=0.8)
    ax2.set_ylabel('Relative Change (P/P0)')
    ax2.set_xlabel('Iteration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/Fig_GDE_optimization_history.png")
    plt.close()

def prepare_aggregated_data() -> Dict[str, pd.DataFrame]:
    print("[GD] Loading and aggregating data...")
    base_path = Path("intermediate/step01_clean.parquet")
    growth_path = Path("intermediate/step04_growth_metrics_hf.parquet")
    japp_path = Path("intermediate/step05_japp_survival.parquet")
    df_base = pd.read_parquet(base_path)
    df_growth = pd.read_parquet(growth_path)
    df_japp = pd.read_parquet(japp_path)
    agg_base = aggregate_by_hour_min(df_base)
    if "place" not in df_growth.columns and "Place" in df_growth.columns:
        df_growth = df_growth.rename(columns={"Place": "place"})
    if "place" not in df_japp.columns and "Place" in df_japp.columns:
        df_japp = df_japp.rename(columns={"Place": "place"})
    agg_growth = aggregate_by_hour_min(df_growth)
    agg_japp = aggregate_by_hour_min(df_japp)
    return {"base": agg_base, "growth": agg_growth, "japp": agg_japp}

def run_multi_stage_optimization():
    cfg_project = default_config()
    base_config = GDEConfig(hf_rule="1min", spinup_cycles=3)
    
    diurnal_data = prepare_aggregated_data()
    preloaded_data = load_high_freq_state(cfg_project, base_config, override_inputs=diurnal_data)
    
    if CHECKPOINT_PATH.exists():
        print("[GD] Found old checkpoint, but physics changed (added Ventilation). Resetting...")
        CHECKPOINT_PATH.unlink()
        
    STAGES = [
        {"id": 1, "max_steps": 1440, "iters": 20, "lr": 0.25, "desc": "Ventilation Tuning"},
        {"id": 2, "max_steps": 1440, "iters": 20, "lr": 0.15, "desc": "Fine Tuning"},
    ]
    
    # Added 'ventilation_factor_s'
    param_names = ["epsilon_cond", "lambda_gas_s", "k_prod_bvoc_s", "loss_rate_s", "ventilation_factor_s"]
    
    initial_params = np.array([
        0.1,        # epsilon
        1e-3,       # lambda_gas
        1e-5,       # k_prod
        1e-5,       # loss_rate
        5e-5        # ventilation (init guess: moderate dilution)
    ], dtype=float)
    
    state = OptimizationState(
        stage=1, iteration=0, params=initial_params.tolist(),
        best_loss=1e9, history_loss=[], history_params=[initial_params.tolist()]
    )
    
    current_params = np.array(state.params)
    print(f"[GD] Starting Fresh. Initial: {dict(zip(param_names, initial_params))}")

    for stage_conf in STAGES:
        stage_id = stage_conf["id"]
        lr = stage_conf["lr"]
        max_steps = stage_conf["max_steps"]
        target_iters = stage_conf["iters"]
        print(f"\n=== Stage {stage_id}: {stage_conf['desc']} ===")
        patience = 0
        
        for it in range(target_iters):
            t0 = time.time()
            steps = 0.05 * current_params
            grads, loss, r2_sites = _finite_diff_grad(
                current_params, steps, base_config, cfg_project, max_steps, preloaded_data
            )
            dt = time.time() - t0
            state.history_loss.append(loss)
            state.history_params.append(current_params.tolist())
            if loss < state.best_loss:
                state.best_loss = loss
                patience = 0
            else:
                patience += 1
            print(f"[Stage {stage_id}] Iter {it+1}/{target_iters} | MSLE={loss:.4f} | Time={dt:.2f}s")
            print(f"    R2: {r2_sites}")
            grad_rel = grads * current_params
            max_grad = np.max(np.abs(grad_rel))
            step_rel = np.zeros_like(current_params)
            if max_grad > 1e-12:
                step_rel = -lr * (grad_rel / max_grad)
            new_params = current_params * (1.0 + step_rel)
            new_params = _clip_params(new_params)
            print(f"    Updated: {dict(zip(param_names, new_params))}")
            current_params = new_params
            state.params = current_params.tolist()
            state.iteration = it + 1
            state.stage = stage_id
            _save_checkpoint(state)
            _plot_history(state, param_names)
            if patience >= 5:
                print("Early stopping.")
                break

    print("\n[GD] Optimization Completed.")
    print(f"Final Params: {dict(zip(param_names, current_params))}")
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "gde_gd_best_params.txt").write_text(
        "\n".join([f"{n}={v:.6e}" for n, v in zip(param_names, current_params)] + [f"loss={state.best_loss:.6e}"])
    )

if __name__ == "__main__":
    run_multi_stage_optimization()
