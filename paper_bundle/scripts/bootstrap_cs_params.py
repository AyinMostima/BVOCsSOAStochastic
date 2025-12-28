from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.workflow.modeling_framework import load_cached_results, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CS scaling parameters (CS0, beta_max) by day.")
    parser.add_argument("--n-boot", type=int, default=500, help="Bootstrap iterations.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory to save figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"), help="Directory to save tables.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick runs.")
    return parser.parse_args()


def _fit_cs_scaling(cs: pd.Series, ratio: pd.Series) -> Tuple[float, float]:
    """
    References: saturation response K_env = beta_max * CS / (CS + CS0).
    Parameters: CS condensation sink (s^-1), ratio = SOA / SOA_lin.
    """
    cs_clean = pd.to_numeric(cs, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ratio_clean = pd.to_numeric(ratio, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = cs_clean.notna() & ratio_clean.notna() & (cs_clean > 0) & (ratio_clean > 0)
    x = cs_clean[mask].to_numpy()
    y = ratio_clean[mask].to_numpy()
    if x.size < 5:
        return np.nan, np.nan
    p0 = [np.nanpercentile(y, 90), np.nanmedian(x)]
    popt, _ = curve_fit(lambda c, bmax, cs0: bmax * c / (c + cs0), x, y, p0=p0, bounds=(0, np.inf), maxfev=20000)
    beta_max, cs0 = popt
    return float(beta_max), float(cs0)


def main() -> None:
    args = parse_args()
    set_plot_style()
    df_sde, cat1, cat2, ml_outputs, labels_cfg = load_cached_results()
    if args.max_rows:
        df_sde = df_sde.head(args.max_rows)
    diag = cat2.get("diag", {})
    tables_dir = args.tables_dir
    figures_dir = args.figures_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows_summary: List[Dict[str, float | str]] = []

    for place in sorted(df_sde["place"].dropna().unique()):
        diag_entry = diag.get((place, "2_cs"))
        if diag_entry is None:
            continue
        cs_series = pd.Series(diag_entry.get("cs")).replace([np.inf, -np.inf], np.nan)
        ratio_series = pd.Series(diag_entry.get("ratio")).replace([np.inf, -np.inf], np.nan)
        idx = cs_series.index
        days = pd.to_datetime(idx).normalize()
        unique_days = days.unique()
        beta_samples = []
        cs0_samples = []
        for _ in range(args.n_boot):
            if unique_days.size == 0:
                continue
            sampled_days = rng.choice(unique_days, size=unique_days.size, replace=True)
            mask = days.isin(sampled_days)
            beta_hat, cs0_hat = _fit_cs_scaling(cs_series[mask], ratio_series[mask])
            beta_samples.append(beta_hat)
            cs0_samples.append(cs0_hat)
        beta_arr = np.array(beta_samples)
        cs0_arr = np.array(cs0_samples)
        beta_ci = (np.nanpercentile(beta_arr, 2.5), np.nanpercentile(beta_arr, 97.5))
        cs0_ci = (np.nanpercentile(cs0_arr, 2.5), np.nanpercentile(cs0_arr, 97.5))
        rows_summary.append(
            {
                "Place": place,
                "beta_max_mean": float(np.nanmean(beta_arr)),
                "beta_max_ci_low": float(beta_ci[0]),
                "beta_max_ci_high": float(beta_ci[1]),
                "CS0_mean": float(np.nanmean(cs0_arr)),
                "CS0_ci_low": float(cs0_ci[0]),
                "CS0_ci_high": float(cs0_ci[1]),
                "n_boot": args.n_boot,
            }
        )
        samples_df = pd.DataFrame({"beta_max": beta_arr, "CS0": cs0_arr})
        samples_df.to_csv(tables_dir / f"Table_CS_params_bootstrap_samples_{place}.csv", index=False)

    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(tables_dir / "Table_CS_params_bootstrap.csv", index=False)
    print("CS bootstrap complete.")


if __name__ == "__main__":
    main()
