from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

HF_RULE = "10s"
HF_SECONDS = int(pd.to_timedelta(HF_RULE).total_seconds())

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.2,
    }
)

def resample_high_freq(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).resample(rule).interpolate(method="time")
    out = numeric
    for col in df.columns:
        if col not in numeric.columns:
            out[col] = df[col].resample(rule).ffill()
    return out

def save_dual(fig: plt.Figure, stem: str) -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"figures/{stem}.svg", bbox_inches="tight")
    fig.savefig(f"figures/{stem}.png", dpi=300, bbox_inches="tight")


def load_data() -> pd.DataFrame:
    base = pd.read_parquet("intermediate/step01_clean.parquet")
    base["Time"] = pd.to_datetime(base["Time"])
    base = base.set_index("Time")
    base = resample_high_freq(base, HF_RULE)

    growth = pd.read_parquet("intermediate/step04_growth_metrics_hf.parquet")
    growth = base.join(growth.drop(columns=["place"], errors="ignore"), how="inner")

    japp = pd.read_parquet("intermediate/step05_japp_survival.parquet").drop(columns=["place"], errors="ignore")
    merged = growth.join(japp, how="inner")
    # Replace non-finite values and attempt a short-range fill to retain samples
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.ffill(limit=6).bfill(limit=6)
    required = [
        "M_1_20",
        "I_indicator",
        "bvocs",
        "temperature_c",
        "NOx",
        "O3",
        "SO2",
        "rh_pct",
        "rad_w_m2",
        "CS_relative",
    ]
    existing = [c for c in required if c in merged.columns]
    clean = merged.dropna(subset=existing)
    return clean


def load_sde_params() -> Dict[str, float]:
    params = pd.read_csv("tables/Table03_SDE_Params.csv")
    lookup = {row["target"]: row for _, row in params.iterrows()}
    return {
        "mu_intercept": lookup["mu"]["intercept"],
        "mu_slope": lookup["mu"]["slope"],
        "sigma_intercept": lookup["sigma"]["intercept"],
        "sigma_slope": lookup["sigma"]["slope"],
    }


def predict_sigma(temp: pd.Series, sde_params: Dict[str, float]) -> pd.Series:
    sigma = sde_params["sigma_intercept"] + sde_params["sigma_slope"] * temp
    return sigma.clip(lower=0.5)


def build_env_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "o3_rad": df["O3"] * df["rad_w_m2"],
            "nox_rh": df["NOx"] * df["rh_pct"],
            "so2_rh": df["SO2"] * df["rh_pct"],
            "o3_rh": df["O3"] * df["rh_pct"],
            "rad": df["rad_w_m2"],
            "cs_rel": df["CS_relative"],
        },
        index=df.index,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> Dict[str, float]:
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - rss / tss if tss else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    n = len(y_true)
    aic = n * np.log(rss / n) + 2 * n_params
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return {"r2": r2, "rmse": rmse, "aic": aic, "bic": bic}


def normalize_ct(ct_raw: np.ndarray) -> tuple[np.ndarray, float]:
    scale = np.nanmedian(ct_raw)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    ct_norm = np.clip(ct_raw / scale, 1e-3, None)
    return ct_norm, scale


def main() -> None:
    df = load_data()
    sde_params = load_sde_params()
    sigma_pred = predict_sigma(df["temperature_c"], sde_params)
    weights = 1.0 / (sigma_pred**2 + 1e-6)

    groups = df.index.floor("D")
    gkf = GroupKFold(n_splits=5)

    feature_cols = ["bvocs", "NOx", "O3", "SO2", "rh_pct", "rad_w_m2", "CS_relative", "I_indicator"]
    model_names = ["model1", "model2", "model3", "model_formula"]
    oof_preds = {name: np.zeros_like(df["M_1_20"].to_numpy()) for name in model_names}
    cv_records: Dict[str, List[float]] = {"model": [], "r2": [], "rmse": []}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, df["M_1_20"], groups=groups)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        w_train = weights.iloc[train_idx]

        # Model 1: direct linear regression
        lr = LinearRegression()
        lr.fit(train[feature_cols], train["M_1_20"])
        pred1 = lr.predict(test[feature_cols])
        oof_preds["model1"][test_idx] = pred1
        cv_records["model"].append("model1")
        cv_records["r2"].append(compute_metrics(test["M_1_20"].to_numpy(), pred1, len(feature_cols))["r2"])
        cv_records["rmse"].append(np.sqrt(np.mean((test["M_1_20"].to_numpy() - pred1) ** 2)))

        # Model 2: two-stage + g(I)
        exog_stage1 = sm.add_constant(train["temperature_c"])
        stage1 = sm.WLS(train["bvocs"], exog_stage1, weights=w_train).fit()
        bvocs_hat_train = stage1.predict(sm.add_constant(train["temperature_c"]))
        bvocs_hat_test = stage1.predict(sm.add_constant(test["temperature_c"]))

        stage2_features = pd.DataFrame(
            {
                "bvocs_hat": bvocs_hat_train,
                "NOx": train["NOx"],
                "O3": train["O3"],
                "SO2": train["SO2"],
                "rh_pct": train["rh_pct"],
                "rad_w_m2": train["rad_w_m2"],
                "CS_relative": train["CS_relative"],
            }
        ).astype(float)
        stage2 = sm.WLS(train["M_1_20"], sm.add_constant(stage2_features), weights=w_train).fit()
        stage2_test = pd.DataFrame(
            {
                "bvocs_hat": bvocs_hat_test,
                "NOx": test["NOx"],
                "O3": test["O3"],
                "SO2": test["SO2"],
                "rh_pct": test["rh_pct"],
                "rad_w_m2": test["rad_w_m2"],
                "CS_relative": test["CS_relative"],
            }
        ).astype(float)
        linear_pred_test = stage2.predict(sm.add_constant(stage2_test))
        resid_train = (train["M_1_20"] - stage2.predict(sm.add_constant(stage2_features))).clip(lower=0)

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(train["I_indicator"], resid_train)
        g_val = iso.transform(test["I_indicator"])
        pred2 = linear_pred_test + g_val
        oof_preds["model2"][test_idx] = pred2
        cv_records["model"].append("model2")
        cv_records["r2"].append(compute_metrics(test["M_1_20"].to_numpy(), pred2, stage2.df_model + 1)["r2"])
        cv_records["rmse"].append(np.sqrt(np.mean((test["M_1_20"].to_numpy() - pred2) ** 2)))

        # Model 3: temperature only
        exog_stage3 = sm.add_constant(train["temperature_c"])
        stage3 = sm.WLS(train["M_1_20"], exog_stage3, weights=w_train).fit()
        pred3 = stage3.predict(sm.add_constant(test["temperature_c"]))
        oof_preds["model3"][test_idx] = pred3
        cv_records["model"].append("model3")
        cv_records["r2"].append(compute_metrics(test["M_1_20"].to_numpy(), pred3, 2)["r2"])
        cv_records["rmse"].append(np.sqrt(np.mean((test["M_1_20"].to_numpy() - pred3) ** 2)))

        # Model formula: SOA = k_env * C_T * M_theta
        exog_ct = sm.add_constant(train["temperature_c"])
        stage1_formula = sm.WLS(train["bvocs"], exog_ct, weights=w_train).fit()
        ct_train_raw = stage1_formula.predict(sm.add_constant(train["temperature_c"]))
        ct_train, scale = normalize_ct(ct_train_raw)
        ct_test_raw = stage1_formula.predict(sm.add_constant(test["temperature_c"]))
        ct_test = np.clip(ct_test_raw / scale, 1e-3, None)

        env_train = build_env_features(train)
        env_test = build_env_features(test)
        env_train_np = env_train.to_numpy(dtype=float)
        env_test_np = env_test.to_numpy(dtype=float)
        target_formula = np.asarray(train["M_1_20"] / ct_train, dtype=float)
        stage2_formula = sm.WLS(target_formula, sm.add_constant(env_train_np), weights=w_train).fit()
        k_env_train = np.clip(stage2_formula.predict(sm.add_constant(env_train_np)), 1e-3, None)
        k_env_test = np.clip(stage2_formula.predict(sm.add_constant(env_test_np)), 1e-3, None)
        ratio_train = np.clip(target_formula / k_env_train, 1e-3, None)
        iso_formula = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso_formula.fit(train["I_indicator"], ratio_train)
        m_theta_test = np.clip(iso_formula.transform(test["I_indicator"]), 1e-3, None)
        pred_formula = k_env_test * ct_test * m_theta_test
        oof_preds["model_formula"][test_idx] = pred_formula
        cv_records["model"].append("model_formula")
        cv_records["r2"].append(compute_metrics(test["M_1_20"].to_numpy(), pred_formula, stage2_formula.df_model + 1)["r2"])
        cv_records["rmse"].append(np.sqrt(np.mean((test["M_1_20"].to_numpy() - pred_formula) ** 2)))

    cv_df = pd.DataFrame(cv_records)
    cv_summary = cv_df.groupby("model").agg({"r2": "mean", "rmse": "mean"}).reset_index()

    # Full-data fits for metrics and diagnostics
    lr_full = LinearRegression().fit(df[feature_cols], df["M_1_20"])
    pred1_full = lr_full.predict(df[feature_cols])

    exog_stage1 = sm.add_constant(df["temperature_c"])
    stage1_full = sm.WLS(df["bvocs"], exog_stage1, weights=weights).fit()
    bvocs_hat_full = stage1_full.predict(sm.add_constant(df["temperature_c"]))
    stage2_full_features = pd.DataFrame(
        {
            "bvocs_hat": bvocs_hat_full,
            "NOx": df["NOx"],
            "O3": df["O3"],
            "SO2": df["SO2"],
            "rh_pct": df["rh_pct"],
            "rad_w_m2": df["rad_w_m2"],
            "CS_relative": df["CS_relative"],
        }
    ).astype(float)
    stage2_full = sm.WLS(df["M_1_20"], sm.add_constant(stage2_full_features), weights=weights).fit()
    resid_full = (df["M_1_20"] - stage2_full.predict(sm.add_constant(stage2_full_features))).clip(lower=0)
    iso_full = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_full.fit(df["I_indicator"], resid_full)
    pred2_full = stage2_full.predict(sm.add_constant(stage2_full_features)) + iso_full.transform(df["I_indicator"])

    stage3_full = sm.WLS(df["M_1_20"], sm.add_constant(df["temperature_c"]), weights=weights).fit()
    pred3_full = stage3_full.predict(sm.add_constant(df["temperature_c"]))

    exog_formula = sm.add_constant(df["temperature_c"])
    stage1_formula_full = sm.WLS(df["bvocs"], exog_formula, weights=weights).fit()
    ct_full_raw = stage1_formula_full.predict(sm.add_constant(df["temperature_c"]))
    ct_full, scale_full = normalize_ct(ct_full_raw)
    env_full = build_env_features(df)
    env_full_np = env_full.to_numpy(dtype=float)
    target_full = np.asarray(df["M_1_20"] / ct_full, dtype=float)
    stage2_formula_full = sm.WLS(target_full, sm.add_constant(env_full_np), weights=weights).fit()
    k_env_full = np.clip(stage2_formula_full.predict(sm.add_constant(env_full_np)), 1e-3, None)
    ratio_full = np.clip(target_full / k_env_full, 1e-3, None)
    iso_formula_full = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_formula_full.fit(df["I_indicator"], ratio_full)
    m_theta_full = np.clip(iso_formula_full.transform(df["I_indicator"]), 1e-3, None)
    pred_formula_full = k_env_full * ct_full * m_theta_full

    full_metrics = pd.DataFrame(
        [
            {"model": "model1", **compute_metrics(df["M_1_20"].to_numpy(), pred1_full, len(feature_cols))},
            {"model": "model2", **compute_metrics(df["M_1_20"].to_numpy(), pred2_full, stage2_full.df_model + 1)},
            {"model": "model3", **compute_metrics(df["M_1_20"].to_numpy(), pred3_full, 2)},
            {"model": "model_formula", **compute_metrics(df["M_1_20"].to_numpy(), pred_formula_full, stage2_formula_full.df_model + 1)},
        ]
    )

    metrics_table = cv_summary.merge(full_metrics, on="model", suffixes=("_cv", "_full"))
    metrics_table.to_csv("tables/Table06_Metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    scatter_models = [
        ("(a) Model 1 direct", df["M_1_20"], pred1_full),
        ("(b) Model 2 two-stage + g(I)", df["M_1_20"], pred2_full),
        ("(c) Functional form $k_{env}C_TM_\\theta$", df["M_1_20"], pred_formula_full),
    ]
    for idx, (ax, (title, y_true, y_pred)) in enumerate(zip(axes.flat[:3], scatter_models)):
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#1b9e77")
        lims = [0, max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, color="#d95f02", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        if idx >= 2:
            ax.set_xlabel("Observed (ug m$^{-3}$)")
        else:
            ax.set_xlabel("")
        ax.set_ylabel("Predicted (ug m$^{-3}$)")
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        rmse = np.sqrt(np.mean((y_true.to_numpy() - y_pred) ** 2))
        ax.annotate(
            f"R^2 = {r2:.2f}\nRMSE = {rmse:.2f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax_metrics = axes.flat[3]
    metric_plot = metrics_table.set_index("model").loc[["model1", "model2", "model_formula", "model3"]]
    metric_plot.index = ["Model 1", "Model 2", "Functional", "Model 3"]
    x = np.arange(len(metric_plot))
    width = 0.3
    ax_metrics.bar(x - width / 2, metric_plot["r2_full"], width=width, color="#1b9e77", label="Full R^2")
    ax_metrics.bar(x + width / 2, metric_plot["r2_cv"], width=width, color="#d95f02", label="CV R^2")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_plot.index)
    ax_metrics.set_ylabel("R^2")
    ax_metrics.set_title("(d) Skill summary")
    ax_metrics.grid(alpha=0.3, linewidth=0.8, axis="y")
    ax_metrics.legend(frameon=False, loc="upper left")
    ax_rmse = ax_metrics.twinx()
    ax_rmse.plot(x, metric_plot["rmse_full"], color="#7570b3", marker="o", label="RMSE (full)")
    ax_rmse.set_ylabel("RMSE (ug m$^{-3}$)", color="#7570b3")
    ax_rmse.tick_params(axis="y", labelcolor="#7570b3")
    ax_rmse.legend(loc="upper right", frameon=False)

    fig.suptitle("Model skill comparison (formation-survival hypotheses)")
    save_dual(fig, "Fig06_Model_Fits")
    plt.close(fig)

    print("Model metrics:")
    print(metrics_table)


if __name__ == "__main__":
    main()



