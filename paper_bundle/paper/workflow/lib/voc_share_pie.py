# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


VOCS_CN: Tuple[str, ...] = (
    "甲硫醇浓度",
    "1,3-丁二烯浓度",
    "丁烯浓度",
    "丙酮、丁烷浓度",
    "正丙醇浓度",
    "甲硫醚、乙硫醇浓度",
    "氯乙烷浓度",
    "异戊二烯浓度",
    "戊烯浓度",
    "戊烷、异戊烷浓度",
    "二甲基甲酰胺浓度",
    "甲酸乙酯浓度",
    "二硫化碳、丙硫醇浓度",
    "苯浓度",
    "环己烯浓度",
    "己烯、甲基环戊烷浓度",
    "正己烷、二甲基丁烷浓度",
    "乙硫醚、丁硫醇浓度",
    "甲苯浓度",
    "苯胺浓度",
    "二甲基二硫醚浓度",
    "1,1-二氯乙烯浓度",
    "甲基环己烷浓度",
    "正庚烷浓度",
    "三乙胺浓度",
    "乙酸正丙酯浓度",
    "二亚乙基三胺浓度",
    "苯乙烯浓度",
    "二甲苯、乙苯浓度",
    "1,3-二氯丙烯浓度",
    "正辛烷浓度",
    "乙酸正丁酯浓度",
    "己硫醇浓度",
    "二甲苯酚浓度",
    "三氯乙烯浓度",
    "二乙基苯浓度",
    "苯甲酸甲酯浓度",
    "磷酸三甲酯浓度",
    "正癸醇浓度",
    "二氯苯浓度",
    "二乙基苯胺浓度",
    "十一烷浓度",
    "四氯乙烯浓度",
    "正十二烷浓度",
    "二溴甲烷浓度",
    "1,2,4-三氯苯浓度",
    "正十三烷浓度",
    "1,2-二溴乙烷浓度",
)

VOCS_EN: Tuple[str, ...] = (
    "Methyl Mercaptan",
    "1,3-Butadiene",
    "Butene",
    "Acetone and Butane",
    "n-Propanol",
    "Methyl Sulfide and Ethyl Mercaptan",
    "Chloroethane",
    "Isoprene",
    "Pentene",
    "Pentane and Isopentane",
    "Dimethylformamide",
    "Ethyl Formate",
    "Carbon Disulfide and Propyl Mercaptan",
    "Benzene",
    "Cyclohexene",
    "Hexene and Methylcyclopentane",
    "n-Hexane and Dimethylbutane",
    "Ethyl Sulfide and Butyl Mercaptan",
    "Toluene",
    "Aniline",
    "Dimethyl Disulfide",
    "1,1-Dichloroethylene",
    "Methylcyclohexane",
    "n-Heptane",
    "Triethylamine",
    "Propyl Acetate",
    "Diethylene Triamine",
    "Styrene",
    "Xylene and Ethylbenzene",
    "1,3-Dichloropropene",
    "n-Octane",
    "Butyl Acetate",
    "Hexyl Mercaptan",
    "Xylenol",
    "Trichloroethylene",
    "Diethylbenzene",
    "Methyl Benzoate",
    "Trimethyl Phosphate",
    "n-Decanol",
    "Dichlorobenzene",
    "Diethyl Aniline",
    "Undecane",
    "Tetrachloroethylene",
    "n-Dodecane",
    "Dibromomethane",
    "1,2,4-Trichlorobenzene",
    "n-Tridecane",
    "1,2-Dibromoethane",
)

VOC_NAME_MAP: Dict[str, str] = dict(zip(VOCS_CN, VOCS_EN))


@dataclass(frozen=True)
class VocShareResult:
    voc_means: pd.Series
    voc_shares: pd.Series


def load_grouped_voc_inputs(bundle_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load grouped VOC CSVs for both sites from the paper bundle.

    References
    ----------
    - Project assets: paper_bundle\\groupedjhS.csv and paper_bundle\\groupedcmS.csv.

    Mathematical expression
    -----------------------
    Not applicable (I/O only).

    Parameter meanings
    ------------------
    - bundle_root: paper_bundle root directory (BUNDLE_ROOT).
    """
    jh_path = bundle_root / "groupedjhS.csv"
    cm_path = bundle_root / "groupedcmS.csv"
    if not jh_path.exists() or not cm_path.exists():
        raise FileNotFoundError("Grouped VOC CSV files not found under bundle root.")
    return pd.read_csv(jh_path), pd.read_csv(cm_path)


def compute_voc_shares_from_grouped(
    df_jh: pd.DataFrame,
    df_cm: pd.DataFrame,
    voc_cols_cn: Sequence[str] = VOCS_CN,
    *,
    hour_start: int = 6,
    hour_end: int = 18,
) -> VocShareResult:
    """
    Compute VOC mean composition for daytime hours, consistent with the legacy approach.

    References
    ----------
    - Legacy snippet: concatenate sites -> filter Hour in [6,18] -> groupby Hour mean -> mean over Hour.

    Mathematical expression
    -----------------------
    1) mu_i(h) = mean_t x_i(t) for each hour h (after filtering).
    2) mu_i = mean_h mu_i(h).
    3) share_i = mu_i / sum_j mu_j * 100.

    Parameter meanings
    ------------------
    - x_i(t): VOC i concentration at time t.
    - h: hour-of-day (integer).
    - hour_start/hour_end: inclusive daytime window.
    """
    df_jh = df_jh.copy()
    df_cm = df_cm.copy()
    df_jh["place"] = "JH"
    df_cm["place"] = "CM"
    data = pd.concat([df_jh, df_cm], axis=0, ignore_index=True)
    if "Hour" not in data.columns:
        raise KeyError("Expected column 'Hour' missing in grouped VOC CSVs.")

    missing = [c for c in voc_cols_cn if c not in data.columns]
    if missing:
        raise KeyError(f"Missing VOC columns: {missing[:5]}")

    data = data[(data["Hour"] >= hour_start) & (data["Hour"] <= hour_end)]
    if data.empty:
        raise ValueError("No data left after daytime filtering.")
    hourly = data.groupby("Hour").mean(numeric_only=True)
    voc_means_cn = hourly.loc[:, list(voc_cols_cn)].mean(axis=0)
    voc_means = voc_means_cn.copy()
    voc_means.index = [VOC_NAME_MAP.get(c, c) for c in voc_means.index]

    total = float(voc_means.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("VOC mean sum is non-positive; cannot compute shares.")
    voc_shares = voc_means / total * 100.0
    return VocShareResult(voc_means=voc_means, voc_shares=voc_shares)


def top_n_with_other(values: pd.Series, top_n: int) -> Tuple[List[str], List[float]]:
    """
    Prepare top-N + "Other VOCs" lists for pie plots.

    References
    ----------
    - Legacy snippet: keep top_n and aggregate the remainder as "Other VOCs".

    Mathematical expression
    -----------------------
    other = sum_{i>n} v_i, after sorting v in descending order.

    Parameter meanings
    ------------------
    - values: per-species weights (any positive scale).
    - top_n: number of explicit labels.
    """
    sorted_vals = values.sort_values(ascending=False)
    labels = sorted_vals.index.tolist()
    vals = sorted_vals.values.astype(float).tolist()
    if len(vals) <= top_n:
        return labels, vals
    labels_top = labels[:top_n] + ["Other VOCs"]
    vals_top = vals[:top_n] + [float(np.sum(vals[top_n:]))]
    return labels_top, vals_top

