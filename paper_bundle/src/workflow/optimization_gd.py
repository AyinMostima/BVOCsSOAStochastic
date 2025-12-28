
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import json
from pathlib import Path

from src.workflow import modeling_framework as mf
from src.workflow.config import default_config, WorkflowConfig

class GlobalSOAModel(nn.Module):
    def __init__(self, num_features: int, init_params: Dict[str, float] = None):
        super().__init__()
        
        # Initialize parameters with reasonable guesses or provided values
        init_params = init_params or {}
        
        # SDE Parameters for BVOC mean: mu(T) = 0.5 * a * T^2 + v0 * T
        # We use a raw parameter and transform it if constraints are needed.
        # For now, initializing close to expected values.
        self.a = nn.Parameter(torch.tensor(init_params.get('a', 0.01)))
        self.v0 = nn.Parameter(torch.tensor(init_params.get('v0', 0.1)))
        
        # CS Scaling Parameters: k_env = beta_max * CS / (CS + CS0)
        # Enforce positivity using softplus during forward pass
        self.raw_beta_max = nn.Parameter(torch.tensor(init_params.get('beta_max', 1.0)))
        self.raw_cs0 = nn.Parameter(torch.tensor(init_params.get('cs0', 0.01)))
        
        # Linear Chemical Modulation Parameters
        self.linear_weights = nn.Parameter(torch.zeros(num_features))
        self.linear_bias = nn.Parameter(torch.tensor(0.0))

    def get_bvoc_mean(self, temp: torch.Tensor) -> torch.Tensor:
        # mu(T) = 0.5 * a * T^2 + v0 * T
        return 0.5 * self.a * temp**2 + self.v0 * temp

    def get_k_env(self, cs: torch.Tensor) -> torch.Tensor:
        # k_env = beta_max * CS / (CS + CS0)
        # Softplus to ensure positive parameters
        beta_max = nn.functional.softplus(self.raw_beta_max)
        cs0 = nn.functional.softplus(self.raw_cs0)
        # Avoid division by zero
        return beta_max * cs / (cs + cs0 + 1e-8)

    def forward(self, temp: torch.Tensor, cs: torch.Tensor, env_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bvoc_pred: Predicted BVOC mean concentration
            soa_pred: Predicted SOA concentration
        """
        # 1. Predict BVOC
        bvoc_pred = self.get_bvoc_mean(temp)
        
        # 2. Calculate CS scaling factor
        k_env = self.get_k_env(cs)
        
        # 3. Calculate Chemical Modulation
        # chem_mod = w * features + b
        chem_mod = torch.matmul(env_features, self.linear_weights) + self.linear_bias
        
        # 4. Predict SOA
        # SOA = k_env * BVOC * ChemMod
        soa_pred = k_env * bvoc_pred * chem_mod
        
        return bvoc_pred, soa_pred

def prepare_data(df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], int, List[str]]:
    """
    Prepares data for PyTorch training.
    Returns:
        tensors: Dict of tensors (temp, cs, env, bvoc_obs, soa_obs)
        num_features: Number of environmental features
        feature_names: List of feature names
    """
    # Generate features using existing framework logic
    env_df = mf._build_env_features(df)
    feature_names = list(env_df.columns)
    
    # Extract relevant columns
    temp = df["temperature_c"].values.astype(np.float32)
    
    # CS calculation requires some overhead, assuming it's already in df or we compute it
    # If 'cs' is not in df, we need to compute it.
    # mf.compute_cs requires specific columns.
    if "cs" in df.columns:
        cs = df["cs"].values.astype(np.float32)
    else:
        # Try to compute
        try:
            cfg = default_config()
            cs_series = mf.compute_cs(df, cfg)
            cs = cs_series.values.astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not compute CS, using placeholder ones (will fail if CS is critical): {e}")
            cs = np.ones_like(temp)

    bvoc_obs = df["bvocs"].values.astype(np.float32)
    soa_obs = df["SOA"].values.astype(np.float32)
    env_features = env_df.values.astype(np.float32)
    
    # Handle NaNs by creating a mask
    mask = np.isfinite(temp) & np.isfinite(cs) & np.isfinite(bvoc_obs) & np.isfinite(soa_obs) & np.all(np.isfinite(env_features), axis=1)
    
    # Apply mask first
    temp = temp[mask]
    cs = cs[mask]
    env_features = env_features[mask]
    bvoc_obs = bvoc_obs[mask]
    soa_obs = soa_obs[mask]

    # Normalize features
    env_mean = np.mean(env_features, axis=0)
    env_std = np.std(env_features, axis=0)
    env_std[env_std == 0] = 1.0 # Avoid division by zero
    env_features = (env_features - env_mean) / env_std
    
    print(f"Data Stats: Temp Mean={temp.mean():.2f}, BVOC Mean={bvoc_obs.mean():.2f}, SOA Mean={soa_obs.mean():.2f}, SOA Var={np.var(soa_obs):.2f}")
    
    tensors = {
        "temp": torch.from_numpy(temp),
        "cs": torch.from_numpy(cs),
        "env": torch.from_numpy(env_features),
        "bvoc_obs": torch.from_numpy(bvoc_obs),
        "soa_obs": torch.from_numpy(soa_obs),
        "env_mean": torch.from_numpy(env_mean), # Save for restoration if needed
        "env_std": torch.from_numpy(env_std)
    }
    
    return tensors, len(feature_names), feature_names

def train_model(
    df: pd.DataFrame, 
    epochs: int = 5000, 
    lr: float = 0.01,
    bvoc_loss_weight: float = 1.0,
    soa_loss_weight: float = 1.0,
    device: str = "cpu"
) -> Tuple[GlobalSOAModel, Dict[str, List[float]]]:
    
    tensors, num_features, feature_names = prepare_data(df)
    
    # Move to device
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.to(device)
            
    model = GlobalSOAModel(num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = {"total": [], "bvoc": [], "soa": []}
    
    print(f"Starting training for {epochs} epochs on {len(tensors['temp'])} samples...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        bvoc_pred, soa_pred = model(tensors["temp"], tensors["cs"], tensors["env"])
        
        # Calculate variances for loss normalization
        var_bvoc = torch.var(tensors["bvoc_obs"])
        var_soa = torch.var(tensors["soa_obs"])
        
        # Avoid division by zero
        if var_bvoc == 0: var_bvoc = 1.0
        if var_soa == 0: var_soa = 1.0
        
        loss_bvoc = nn.MSELoss()(bvoc_pred, tensors["bvoc_obs"]) / var_bvoc
        loss_soa = nn.MSELoss()(soa_pred, tensors["soa_obs"]) / var_soa
        
        total_loss = bvoc_loss_weight * loss_bvoc + soa_loss_weight * loss_soa
        
        total_loss.backward()
        optimizer.step()
        
        loss_history["total"].append(total_loss.item())
        loss_history["bvoc"].append(loss_bvoc.item())
        loss_history["soa"].append(loss_soa.item())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} (BVOC: {loss_bvoc.item():.4f}, SOA: {loss_soa.item():.4f})")
            
    return model, loss_history

def run_optimization_pipeline():
    cfg = default_config()
    mf.set_plot_style()
    
    print("Loading data...")
    df = mf.load_base_data(cfg)
    df = mf.aggregate_by_hour_min(df)
    df = mf._clean_outliers(df, ["SOA", "bvocs", "NOx", "O3", "SO2", "rh_pct", "temperature_c"])
    
    # Compute CS and add to df
    cs_series = mf.compute_cs(df, cfg)
    df["cs"] = cs_series
    
    results = {}
    
    for place, sub_df in df.groupby("place"):
        print(f"\nOptimizing for Place: {place}")
        model, history = train_model(sub_df, epochs=5000, lr=0.005)
        
        # Extract parameters
        params = {
            "a": model.a.item(),
            "v0": model.v0.item(),
            "beta_max": nn.functional.softplus(model.raw_beta_max).item(),
            "cs0": nn.functional.softplus(model.raw_cs0).item(),
            "linear_bias": model.linear_bias.item(),
            "weights": model.linear_weights.detach().cpu().numpy().tolist()
        }
        
        results[place] = {
            "params": params,
            "final_loss": history["total"][-1]
        }
        print(f"Optimization finished for {place}.")
        print(f"Parameters: {json.dumps(params, indent=2)}")

    # Save results
    output_path = Path("intermediate/gd_optimization_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_optimization_pipeline()
