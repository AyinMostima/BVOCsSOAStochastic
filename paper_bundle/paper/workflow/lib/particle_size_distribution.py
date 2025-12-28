# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


CONCENTRATION_COLUMNS: Tuple[str, ...] = (
    "C0.25um",
    "C0.28um",
    "C0.30um",
    "C0.35um",
    "C0.40um",
    "C0.45um",
    "C0.50um",
    "C0.58um",
    "C0.65um",
    "C0.70um",
    "C0.80um",
    "C1.00um",
    "C1.30um",
    "C1.60um",
    "C2.00um",
    "C2.50um",
    "C3.00um",
    "C3.50um",
    "C4.00um",
    "C5.00um",
    "C6.50um",
    "C7.50um",
    "C8.50um",
    "C10.00um",
    "C12.50um",
    "C15.00um",
    "C17.50um",
    "C20.00um",
    "C25.00um",
    "C30.00um",
    "C32.00um",
)


@dataclass(frozen=True)
class CompressedAxis:
    size_um: List[float]
    positions: List[float]
    mapping: Dict[float, float]
    tick_positions: List[float]
    tick_labels: List[str]
    tail_label: str


def _sum_columns_slice_exclusive(df: pd.DataFrame, start_col: str, end_col: str) -> pd.Series:
    """
    References
    ----------
    - Legacy script behavior: pandas column slicing with an exclusive end index.

    Mathematical expression
    -----------------------
    s = sum_{i in cols[start:end)} x_i.

    Parameter meanings
    ------------------
    - start_col: first included column label.
    - end_col: first excluded column label (exclusive).
    """
    cols = list(df.columns)
    i0 = cols.index(start_col)
    i1 = cols.index(end_col)
    subset = cols[i0:i1]
    return df.loc[:, subset].sum(axis=1, min_count=1)


def load_grouped_soa_inputs(bundle_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load grouped SOA size-distribution CSVs for both sites.

    References
    ----------
    - Project assets: paper_bundle\\groupedjhSOA.csv and paper_bundle\\groupedcmSOA.csv.

    Mathematical expression
    -----------------------
    Not applicable (I/O only).

    Parameter meanings
    ------------------
    - bundle_root: paper_bundle root directory (BUNDLE_ROOT).
    """
    jh_path = bundle_root / "groupedjhSOA.csv"
    cm_path = bundle_root / "groupedcmSOA.csv"
    if not jh_path.exists() or not cm_path.exists():
        raise FileNotFoundError("Grouped SOA CSV files not found under bundle root.")
    return pd.read_csv(jh_path), pd.read_csv(cm_path)


def add_soa_csoa_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SOA/CSOA columns using the legacy slice logic from the provided script.

    References
    ----------
    - Internal dataset convention: SOA is sum over size bins (0.25um, 0.28um, 0.30um).
    - Legacy snippet uses an exclusive end slice: [loc(0.25um) : loc(0.30um)).

    Mathematical expression
    -----------------------
    SOA = sum_{d in [0.25, 0.30)} N_d, CSOA = sum_{d in [0.25, 0.30)} C_d.

    Parameter meanings
    ------------------
    - N_d: number concentration in bin d (um) from columns like "0.25um".
    - C_d: number concentration in bin d (um) from columns like "C0.25um".
    """
    out = df.copy()
    out["SOA"] = _sum_columns_slice_exclusive(out, "0.25um", "0.30um")
    out["CSOA"] = _sum_columns_slice_exclusive(out, "C0.25um", "C0.30um")
    return out


def build_hourmin_means(df_jh: pd.DataFrame, df_cm: pd.DataFrame) -> pd.DataFrame:
    """
    Build mean-by-(Hour_Min, place) table consistent with the legacy snippet.

    References
    ----------
    - Legacy snippet: dataall = data.groupby(["Hour_Min", "place"]).mean(numeric_only=True).reset_index().

    Mathematical expression
    -----------------------
    mean_x(h, p) = (1/n) * sum_{t in group(h,p)} x_t.

    Parameter meanings
    ------------------
    - h: minute-of-day label "HH:MM".
    - p: site label ("JH" or "CM").
    - x_t: any numeric column at time t.
    """
    df_jh = add_soa_csoa_legacy_columns(df_jh)
    df_cm = add_soa_csoa_legacy_columns(df_cm)
    df_jh = df_jh.copy()
    df_cm = df_cm.copy()
    df_jh["place"] = "JH"
    df_cm["place"] = "CM"
    data = pd.concat([df_jh, df_cm], axis=0, ignore_index=True)
    if "Hour_Min" not in data.columns:
        raise KeyError("Expected column 'Hour_Min' missing in grouped SOA CSVs.")
    return data.groupby(["Hour_Min", "place"]).mean(numeric_only=True).reset_index()


def to_long_concentration(dataall: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide concentration table to long format for line plotting.

    References
    ----------
    - Seaborn lineplot input convention (long-form data).

    Mathematical expression
    -----------------------
    Not applicable (reshape only).

    Parameter meanings
    ------------------
    - dataall: mean-by-(Hour_Min, place) table containing concentration columns.
    """
    missing = [c for c in CONCENTRATION_COLUMNS if c not in dataall.columns]
    if missing:
        raise KeyError(f"Missing concentration columns: {missing[:5]}")
    data_long = dataall.loc[:, ["place", *CONCENTRATION_COLUMNS]].melt(
        id_vars="place",
        value_vars=list(CONCENTRATION_COLUMNS),
        var_name="size_col",
        value_name="concentration",
    )
    data_long["size_um"] = data_long["size_col"].apply(lambda s: float(str(s)[1:-2]))
    return data_long.drop(columns=["size_col"])


def build_compressed_axis(
    size_um: Sequence[float],
    *,
    power: float = 0.25,
    tick_limit_um: float = 0.65,
) -> CompressedAxis:
    """
    Build the non-linear (compressed) x-axis mapping used in the legacy plot.

    References
    ----------
    - Legacy snippet: positions = linspace(0,1,n)**0.25 * max(size).

    Mathematical expression
    -----------------------
    x_i = (i/(n-1))^p * max(size), i=0..n-1.

    Parameter meanings
    ------------------
    - power p: compression strength (smaller p -> stronger compression).
    - tick_limit_um: show explicit ticks up to this size, then collapse to a tail label.
    """
    size_list = list(size_um)
    if not size_list:
        raise ValueError("size_um is empty.")
    n = len(size_list)
    positions = (np.linspace(0.0, 1.0, n) ** float(power) * float(max(size_list))).tolist()
    mapping = {float(s): float(p) for s, p in zip(size_list, positions)}

    filtered_positions = [pos for pos, s in zip(positions, size_list) if float(s) <= float(tick_limit_um)]
    filtered_labels = [f"{float(s):.2f}" for s in size_list if float(s) <= float(tick_limit_um)]
    if positions:
        filtered_positions.append(float(positions[-1]))
        filtered_labels.append("")

    return CompressedAxis(
        size_um=[float(s) for s in size_list],
        positions=[float(p) for p in positions],
        mapping=mapping,
        tick_positions=[float(p) for p in filtered_positions],
        tick_labels=[str(s) for s in filtered_labels],
        tail_label=f"{float(tick_limit_um):.2f}-32.00",
    )

