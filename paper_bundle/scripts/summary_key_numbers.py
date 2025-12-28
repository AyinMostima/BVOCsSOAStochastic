from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def main() -> None:
    root = Path(".")
    tables = root / "tables"
    rows = []

    # CS params bootstrap
    cs_boot = tables / "Table_CS_params_bootstrap.csv"
    if cs_boot.exists():
        df = pd.read_csv(cs_boot)
        rows.extend(
            {
                "Category": "CS",
                "Place": r["Place"],
                "Key": k,
                "Value": v,
            }
            for _, r in df.iterrows()
            for k, v in {
                "CS0_mean": r.get("CS0_mean"),
                "CS0_ci_low": r.get("CS0_ci_low"),
                "CS0_ci_high": r.get("CS0_ci_high"),
                "beta_max_mean": r.get("beta_max_mean"),
                "beta_max_ci_low": r.get("beta_max_ci_low"),
                "beta_max_ci_high": r.get("beta_max_ci_high"),
            }.items()
        )

    # Timescales
    timescales = tables / "Table_timescales.csv"
    if timescales.exists():
        df = pd.read_csv(timescales)
        # store median CS quantile entries
        med = df[df["CS_quantile"] == 0.5]
        for _, r in med.iterrows():
            rows.append(
                {
                    "Category": "Timescale",
                    "Place": r["Place"],
                    "Key": f"tau_C_s (eps={r['epsilon']},Q50)",
                    "Value": r["tau_C_s"],
                }
            )
            rows.append(
                {
                    "Category": "Timescale",
                    "Place": r["Place"],
                    "Key": "tau_SOA_s",
                    "Value": r["tau_SOA_s"],
                }
            )

    # Mass closure
    mass_table = tables / "Table_MassClosure_alpha.csv"
    if mass_table.exists():
        df = pd.read_csv(mass_table)
        for _, r in df.iterrows():
            rows.append(
                {
                    "Category": "MassClosure",
                    "Place": r["Place"],
                    "Key": f"alpha_hat_Delta{r['Delta_min']}_Lag{r.get('Lag_min', 0)}",
                    "Value": r["alpha_hat"],
                }
            )
            rows.append(
                {
                    "Category": "MassClosure",
                    "Place": r["Place"],
                    "Key": f"alpha_theory_Delta{r['Delta_min']}",
                    "Value": r["alpha_theory"],
                }
            )
            if "r" in r:
                rows.append(
                    {
                        "Category": "MassClosure",
                        "Place": r["Place"],
                        "Key": f"r_all_Delta{r['Delta_min']}_Lag{r.get('Lag_min', 0)}",
                        "Value": r.get("r"),
                    }
                )
            if "r_highI" in r:
                rows.append(
                    {
                        "Category": "MassClosure",
                        "Place": r["Place"],
                        "Key": f"r_highI_Delta{r['Delta_min']}_Lag{r.get('Lag_min', 0)}",
                        "Value": r.get("r_highI"),
                    }
                )

    # H3 power law
    h3_table = tables / "Table_H3_powerlaw.csv"
    if h3_table.exists():
        df = pd.read_csv(h3_table)
        for _, r in df.iterrows():
            rows.append(
                {
                    "Category": "H3",
                    "Place": r["Place"],
                    "Key": "a_H2SO4",
                    "Value": r.get("a_H2SO4"),
                }
            )
            rows.append(
                {
                    "Category": "H3",
                    "Place": r["Place"],
                    "Key": "b_NucOX",
                    "Value": r.get("b_NucOX"),
                }
            )
            rows.append(
                {
                    "Category": "H3",
                    "Place": r["Place"],
                    "Key": "c_Base",
                    "Value": r.get("c_Base"),
                }
            )

    # Residual stats
    resid_table = tables / "Table_residuals_stats.csv"
    if resid_table.exists():
        df = pd.read_csv(resid_table)
        for _, r in df.iterrows():
            rows.append(
                {
                    "Category": "Residual",
                    "Place": r["Place"],
                    "Key": f"{r['Model']}_Residual_rmse",
                    "Value": r.get("Residual_rmse"),
                }
            )

    out_path = tables / "Table_key_numbers_for_paper.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Key numbers saved to {out_path}")


if __name__ == "__main__":
    main()
