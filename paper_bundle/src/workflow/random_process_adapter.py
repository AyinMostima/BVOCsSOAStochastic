from __future__ import annotations

import pandas as pd

from .modeling_framework import (
    WorkflowConfig,
    default_config,
    load_base_data,
    fit_sde_by_site,
    run_category_i,
    run_category_ii,
    run_category_iii,
)


def load_full_timeseries(cfg: WorkflowConfig | None = None) -> pd.DataFrame:
    """Load full time series data using the standard workflow.

    This is a thin wrapper so that standalone random-process scripts can
    share the same data loading logic as the main workflow.
    """

    cfg = cfg or default_config()
    df = load_base_data(cfg)
    return df


def run_full_linear_and_cs(
    df: pd.DataFrame, cfg: WorkflowConfig | None = None
) -> tuple[dict, dict]:
    """Run Category I and II to obtain linear and CS-scaled predictions.

    Returns
    -------
    cat1_outputs : dict
        Outputs from run_category_i, including linear predictions.
    cat2_outputs : dict
        Outputs from run_category_ii, including CS parameters, CS series,
        and SOA predictions with CS scaling.
    """

    cfg = cfg or default_config()
    _, df_sde = fit_sde_by_site(df, cfg)
    cat1_outputs = run_category_i(df_sde)
    cat2_outputs = run_category_ii(df_sde, cat1_outputs, cfg)
    return cat1_outputs, cat2_outputs


def run_full_ml_pipeline(
    df: pd.DataFrame, cfg: WorkflowConfig | None = None
) -> tuple[dict, dict, dict]:
    """Run Category I, II, and III to obtain ML predictions.

    This helper mirrors the main workflow when random-process or
    sensitivity scripts need access to the full ML pipeline outputs.
    """

    cfg = cfg or default_config()
    _, df_sde = fit_sde_by_site(df, cfg)
    cat1_outputs = run_category_i(df_sde)
    cat2_outputs = run_category_ii(df_sde, cat1_outputs, cfg)
    ml_outputs = run_category_iii(df_sde, cat1_outputs, cat2_outputs)
    return cat1_outputs, cat2_outputs, ml_outputs
