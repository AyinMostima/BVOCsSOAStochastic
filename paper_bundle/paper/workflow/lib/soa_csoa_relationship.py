# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r2: float


def fit_linear(x: pd.Series, y: pd.Series) -> tuple[LinearFit, np.ndarray, np.ndarray]:
    """
    Fit a simple linear regression y = k x + b, consistent with the provided snippet.

    References
    ----------
    - Standard least squares regression (scikit-learn LinearRegression).

    Mathematical expression
    -----------------------
    y_hat = k x + b, where (k, b) minimize sum_i (y_i - (k x_i + b))^2.
    R^2 = 1 - SSE / SST.

    Parameter meanings
    ------------------
    - x: CSOA (number counts) predictor.
    - y: SOA (mass) response.
    - k: fitted slope.
    - b: fitted intercept.
    """
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()
    x_arr = x_num[valid].to_numpy().reshape(-1, 1)
    y_arr = y_num[valid].to_numpy()
    if x_arr.size == 0:
        fit = LinearFit(np.nan, np.nan, np.nan)
        return fit, np.array([]), np.array([])
    model = LinearRegression()
    model.fit(x_arr, y_arr)
    y_pred = model.predict(x_arr)
    r2 = float(r2_score(y_arr, y_pred)) if y_arr.size > 1 else float("nan")
    fit = LinearFit(float(model.coef_[0]), float(model.intercept_), r2)
    return fit, x_arr.ravel(), y_pred


def format_slope_tex(slope: float, decimals: int = 3) -> str:
    """
    Format slope as a TeX math string using scientific notation.

    References
    ----------
    - Matplotlib mathtext scientific formatting.

    Mathematical expression
    -----------------------
    k = m * 10^e, with m in [1, 10).

    Parameter meanings
    ------------------
    - slope: fitted k value.
    - decimals: number of decimals for mantissa.
    """
    if not np.isfinite(slope):
        return r"\mathrm{nan}"
    mantissa_str, exp_str = f"{float(slope):.{int(decimals)}e}".split("e")
    exp = int(exp_str)
    mantissa = float(mantissa_str)
    return rf"{mantissa:.{int(decimals)}f}\times 10^{{{exp:d}}}"

