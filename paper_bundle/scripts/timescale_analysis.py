from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.workflow.config import default_config
from src.workflow.modeling_framework import load_cached_results, save_table, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate tau_C and tau_SOA from cached CS and scaling parameters.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"), help="Directory to save tables.")
    parser.add_argument("--eps-values", type=float, nargs="+", default=[0.3, 0.5, 1.0], help="Epsilon scenarios.")
    parser.add_argument("--cs-quantiles", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="CS quantiles to report.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on cached rows for quick runs.")
    return parser.parse_args()


def _compute_tau_rows(
    cs_series: pd.Series,
    cs0: float,
    eps_list: List[float],
    quantiles: List[float],
    kappa_loss: float,
    place: str,
    obs_dt_seconds: float,
) -> List[Dict[str, float | str]]:
    """
    References: condensation sink steady state dC/dt = Y_C - (lambda + eps*CS)*C.
    Equation: tau_C = 1 / (lambda + eps*CS); lambda approximated by eps*CS0.
    Parameters: CS condensation sink (s^-1), CS0 half-saturation scale, eps scaling factor, lambda background loss (s^-1).
    """
    rows: List[Dict[str, float | str]] = []
    cs_vals = cs_series.quantile(quantiles).to_dict()
    for eps in eps_list:
        lambda_est = eps * cs0
        for q, cs_val in cs_vals.items():
            tau_c = 1.0 / max(lambda_est + eps * cs_val, 1e-9)
            rows.append(
                {
                    "Place": place,
                    "CS_quantile": q,
                    "CS_value": cs_val,
                    "epsilon": eps,
                    "lambda_est": lambda_est,
                    "tau_C_s": tau_c,
                    "kappa_loss": kappa_loss,
                    "tau_SOA_s": 1.0 / max(kappa_loss, 1e-12),
                    "delta_t_obs_s": obs_dt_seconds,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cfg = default_config()
    set_plot_style()
    df_sde, cat1, cat2, ml_outputs, labels_cfg = load_cached_results()
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)

    cs_series = cat2.get("cs")
    cs_params = cat2.get("params", pd.DataFrame())
    if cs_series is None or cs_series.empty or cs_params.empty:
        raise RuntimeError("CS series or CS parameters not available in cache; run Category II first.")

    eps_list = args.eps_values
    quantiles = args.cs_quantiles
    # References: boundary-layer mixing loss; assume kappa_loss ~ 1e-4 s^-1 for typical daytime mixing ~ 3 h.
    kappa_loss_default = 1.0e-4
    figures_dir = args.figures_dir
    tables_dir = args.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows_tau: List[Dict[str, float | str]] = []
    rows_cs_params: List[Dict[str, float | str]] = []

    # Compute median observation delta t from df_sde index spacing.
    if isinstance(df_sde.index, pd.DatetimeIndex) and len(df_sde.index) > 1:
        obs_dt_seconds = float(df_sde.index.to_series().diff().median().total_seconds())
    else:
        obs_dt_seconds = math.nan

    for place, sub in df_sde.groupby("place"):
        cs_place = cs_series.loc[sub.index] if isinstance(cs_series, pd.Series) else pd.Series(dtype=float)
        if cs_place.empty:
            continue
        params_place = cs_params[cs_params["Place"] == place]
        beta_row = params_place[params_place["Parameter"] == "beta_max"]
        cs0_row = params_place[params_place["Parameter"] == "CS0"]
        if beta_row.empty or cs0_row.empty:
            continue
        beta_max = float(beta_row["Estimate"].iloc[0])
        cs0 = float(cs0_row["Estimate"].iloc[0])
        # plateau of k_env(CS) = beta_max * CS / (CS + CS0); consistent with Category II definition.
        k_env_max = beta_max
        rows_cs_params.append(
            {
                "Place": place,
                "beta_max": beta_max,
                "CS0": cs0,
                "K_env_max": k_env_max,
            }
        )
        rows_tau.extend(
            _compute_tau_rows(
                cs_place,
                cs0=cs0,
                eps_list=eps_list,
                quantiles=quantiles,
                kappa_loss=kappa_loss_default,
                place=place,
                obs_dt_seconds=obs_dt_seconds,
            )
        )

    tau_df = pd.DataFrame(rows_tau)
    cs_param_df = pd.DataFrame(rows_cs_params)
    if not tau_df.empty:
        tau_df_sorted = tau_df.sort_values(["Place", "epsilon", "CS_quantile"])
        save_table(tau_df_sorted, "Table_timescales", "Timescales")
        tau_df_sorted.to_csv(tables_dir / "Table_timescales.csv", index=False)
    if not cs_param_df.empty:
        save_table(cs_param_df, "Table_CS_params", "CSParams")
        cs_param_df.to_csv(tables_dir / "Table_CS_params.csv", index=False)

    if tau_df.empty:
        print("No tau data computed; skipping figure.")
        return

    plt.figure(figsize=(7.5, 5.2))
    ax = plt.gca()
    palette = {"JH": "#1b9e77", "CM": "#d95f02"}
    for (place, eps), sub in tau_df.groupby(["Place", "epsilon"]):
        ax.scatter(
            sub["tau_C_s"],
            sub["tau_SOA_s"],
            label=f"{place} eps={eps}",
            color=palette.get(place, "#4e79a7"),
            alpha=0.75,
            s=40,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tau_C (s)")
    ax.set_ylabel("tau_SOA (s)")
    ax.set_title("Timescale comparison: tau_C vs tau_SOA")
    ax.grid(alpha=0.3, linewidth=0.8, which="both")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out_stem = figures_dir / "Fig_timescales_summary"
    plt.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_stem}.svg", bbox_inches="tight")
    plt.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Timescale analysis complete.")


if __name__ == "__main__":
    main()
