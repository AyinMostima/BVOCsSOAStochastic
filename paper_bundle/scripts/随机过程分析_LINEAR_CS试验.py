import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import joblib
import statsmodels.api as sm
from paper.workflow.lib.modeling_framework_paper import set_plot_style
set_plot_style()

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.patheffects as path_effects
from pathlib import Path

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_JH = BUNDLE_ROOT / "groupedjhS.csv"
DATA_CM = BUNDLE_ROOT / "groupedcmS.csv"
DATA_JH_SOA = BUNDLE_ROOT / "groupedjhSOA.csv"
DATA_CM_SOA = BUNDLE_ROOT / "groupedcmSOA.csv"
from src.workflow.modeling_framework import (
    compute_cs,
    default_config,
    load_cached_results,
    load_base_data,
)


#----End Of Cell----

np.random.seed(20231125)
random.seed(20231125)

datajh = pd.read_csv(DATA_JH)

datacm = pd.read_csv(DATA_CM)

datajhsoa = pd.read_csv(DATA_JH_SOA)

datacmsoa = pd.read_csv(DATA_CM_SOA)



#----End Of Cell----



#娓╁害褰卞搷

VOCs = ['Methyl Mercaptan', '1,3-Butadiene', 'Butene', 'Acetone/Butane', 'n-Propanol',

        'Dimethyl Sulfide/Ethyl Mercaptan', 'Chloroethane', 'Isoprene', 'Pentene', 'Pentane/Isopentane',

        'Dimethylformamide', 'Ethyl Formate', 'Carbon Disulfide/Propyl Mercaptan', 'Benzene', 'Cyclohexene',

        'Hexene/Methylcyclopentane', 'n-Hexane/Dimethylbutane', 'Ethyl Sulfide/Butyl Mercaptan', 'Toluene', 'Aniline',

        'Dimethyl Disulfide', '1,1-Dichloroethylene', 'Methylcyclohexane', 'n-Heptane', 'Triethylamine',

        'n-Propyl Acetate', 'Diethylene Triamine', 'Styrene', 'Xylene/Ethylbenzene', '1,3-Dichloropropene', 'n-Octane',

        'n-Butyl Acetate', 'Hexyl Mercaptan', 'Xylenol', 'Trichloroethylene', 'Diethylbenzene', 'Methyl Benzoate',

        'Trimethyl Phosphate', 'n-Decanol', 'Dichlorobenzene', 'Diethyl Aniline', 'Undecane', 'Tetrachloroethylene',

        'n-Dodecane', 'Dibromomethane', '1,2,4-Trichlorobenzene', 'n-Tridecane', '1,2-Dibromoethane']



datajhsoa["SOA"] = 0

datajh["SOA"] = 0

for i in datajhsoa.columns[(datajhsoa.columns.get_loc("0.25um")):(datajhsoa.columns.get_loc("0.30um"))]:

    datajhsoa["SOA"] = datajhsoa["SOA"] + datajhsoa[i]

    datajh["SOA"] = datajh["SOA"] + datajh[i]

datajhsoa["CSOA"] = 0

for i in datajhsoa.columns[(datajhsoa.columns.get_loc("C0.25um")):(datajhsoa.columns.get_loc("C0.30um"))]:

    datajhsoa["CSOA"] = datajhsoa["CSOA"] + datajhsoa[i]

datacmsoa["SOA"] = 0

datacm["SOA"] = 0

for i in datacmsoa.columns[(datacmsoa.columns.get_loc("0.25um")):(datacmsoa.columns.get_loc("0.30um"))]:

    datacmsoa["SOA"] = datacmsoa["SOA"] + datacmsoa[i]

    datacm["SOA"] = datacm["SOA"] + datacm[i]

datacmsoa["CSOA"] = 0

for i in datacmsoa.columns[(datacmsoa.columns.get_loc("C0.25um")):(datacmsoa.columns.get_loc("C0.30um"))]:

    datacmsoa["CSOA"] = datacmsoa["CSOA"] + datacmsoa[i]

datajh['place'] = 'JH'

datacm['place'] = 'CM'

dataall = pd.concat([datajh, datacm], axis=0)

dataall.columns = ['Time', 'TVOCs', 'Methyl Mercaptan', '1,3-Butadiene', 'Butene', 'Acetone/Butane', 'n-Propanol',

                   'Dimethyl Sulfide/Ethyl Mercaptan', 'Chloroethane', 'Isoprene', 'Pentene', 'Pentane/Isopentane',

                   'Dimethylformamide', 'Ethyl Formate', 'Carbon Disulfide/Propyl Mercaptan', 'Benzene', 'Cyclohexene',

                   'Hexene/Methylcyclopentane', 'n-Hexane/Dimethylbutane', 'Ethyl Sulfide/Butyl Mercaptan', 'Toluene',

                   'Aniline', 'Dimethyl Disulfide', '1,1-Dichloroethylene', 'Methylcyclohexane', 'n-Heptane',

                   'Triethylamine', 'n-Propyl Acetate', 'Diethylene Triamine', 'Styrene', 'Xylene/Ethylbenzene',

                   '1,3-Dichloropropene', 'n-Octane', 'n-Butyl Acetate', 'Hexyl Mercaptan', 'Xylenol',

                   'Trichloroethylene', 'Diethylbenzene', 'Methyl Benzoate', 'Trimethyl Phosphate', 'n-Decanol',

                   'Dichlorobenzene', 'Diethyl Aniline', 'Undecane', 'Tetrachloroethylene', 'n-Dodecane',

                   'Dibromomethane', '1,2,4-Trichlorobenzene', 'n-Tridecane', '1,2-Dibromoethane', '0.25um', '0.28um',

                   '0.30um', '0.35um', '0.40um', '0.45um', '0.50um', '0.58um', '0.65um', '0.70um', '0.80um', '1.00um',

                   '1.30um', '1.60um', '2.00um', '2.50um', '3.00um', '3.50um', '4.00um', '5.00um', '6.50um', '7.50um',

                   '8.50um', '10.00um', '12.50um', '15.00um', '17.50um', '20.00um', '25.00um', '30.00um', '32.00um',

                   'PM10', 'PM2.5', 'PM1', 'SO2', 'NOx', 'NO', 'NO2', 'CO', 'O3', 'NO2.1', 'NegativeOxygenIons',

                   'Radiation', 'Temperature', 'Humidity', 'WindSpeed', 'Hour_Min_Sec', 'Hour_Min', 'Hour', 'Month',

                   'Day', 'Datetime', 'seconds', 'SOA', 'place']



#----End Of Cell----



from scipy.optimize import curve_fit


# Use cached full-pipeline results so that this script shares the same
# condensation sink and ML outputs as the main workflow. The stochastic
# calculations themselves remain on the full time series from
# load_base_data to preserve variance.
cfg_main = default_config()
df_base = load_base_data(cfg_main)
df_sde, cat1_outputs, cat2_outputs, ml_outputs, labels_cfg = load_cached_results()

df_long = df_base.reset_index().rename(columns={"Time": "Datetime"})
df_long["Time"] = df_long["Datetime"]
df_long["Temperature"] = df_long["temperature_c"]
df_long["Radiation"] = df_long["rad_w_m2"]
df_long["Humidity"] = df_long["rh_pct"]
df_long["Isoprene"] = df_long["bvocs"]
df_long["Hour_Min"] = df_long["Datetime"].dt.strftime("%H:%M")
df_long["Hour"] = df_long["Time"].dt.hour
df_long["Month"] = df_long["Time"].dt.month
df_long["Day"] = df_long["Time"].dt.day
df_long["seconds"] = (df_long["Time"] - df_long["Time"].dt.normalize()).dt.total_seconds()

dataall = df_long[
    [
        "Time",
        "Isoprene",
        "SOA",
        "SO2",
        "NOx",
        "O3",
        "Radiation",
        "Temperature",
        "Humidity",
        "Hour_Min",
        "Hour",
        "Month",
        "Day",
        "Datetime",
        "seconds",
        "place",
    ]
].copy()

dataall["TVOCs"] = np.nan
dataall["PM10"] = np.nan
dataall["PM2.5"] = np.nan
dataall["PM1"] = np.nan

data_jh = dataall[dataall["place"] == "JH"].copy()
data_cm = dataall[dataall["place"] == "CM"].copy()

# References: Fuchs and Sutugin (1971) condensation sink theory; CS = sum(4*pi*D_v*r*F(Kn)*N).
# Equations: CS(Time) already computed in the main workflow and cached as cat2_outputs['cs'].
# Parameters: condensation sink series CS(t) derived from particle number size distribution and temperature.
cs_series_cached = cat2_outputs.get("cs")
cs_hour_min_mean = None
if isinstance(cs_series_cached, pd.Series) and not cs_series_cached.empty:
    cs_frame = pd.DataFrame({"Time": cs_series_cached.index, "CS": cs_series_cached.values})
    cs_frame["Hour_Min"] = cs_frame["Time"].dt.strftime("%H:%M")
    cs_hour_min_mean = cs_frame.groupby("Hour_Min")["CS"].mean(numeric_only=True).reset_index()

# Fallback CS aggregation from base data if cache is unavailable.
if cs_hour_min_mean is None:
    cfg_cs = default_config()
    number_cols = [c for c in df_base.columns if c.startswith("C") and c.endswith("um")]
    if not number_cols:
        raise ValueError("No C*um columns found for CS calculation.")
    df_hour = df_base.reset_index()
    df_hour["Hour_Min"] = df_hour["Time"].dt.strftime("%H:%M")
    grouped_num = (
        df_hour.groupby(["place", "Hour_Min"])[number_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped_temp = (
        df_hour.groupby(["place", "Hour_Min"])["temperature_c"]
        .mean(numeric_only=True)
        .reset_index()
    )
    cs_input = grouped_num.merge(grouped_temp, on=["place", "Hour_Min"], how="inner")
    cs_input["temperature_c"] = cs_input["temperature_c"]
    cs_values = compute_cs(cs_input[["temperature_c"] + number_cols], cfg_cs)
    cs_input["CS"] = cs_values
    cs_hour_min_mean = cs_input.groupby("Hour_Min")["CS"].mean(numeric_only=True).reset_index()

from scipy.stats import zscore

from matplotlib.ticker import FuncFormatter, MaxNLocator

import matplotlib.dates as mdates

from scipy.stats import t





# 瀹氫箟涓庡潎鍊肩殑鍏崇郴鍑芥暟锛堜簩娆″嚱鏁帮級

def mean_relation(T, Q0, a, v0):

    return Q0 + (a * T**2) / 2 + T * v0



# 瀹氫箟涓庢柟宸殑鍏崇郴鍑芥暟锛堜笁娆″嚱鏁帮級

def std_dev_relation(T, k, sigma0):

    return (k**2 * T**3) / 3 + k * T**2 * sigma0 + T * sigma0**2





# 瀹氫箟鍘婚櫎绂荤兢鍊肩殑鍑芥暟锛堜娇鐢?Z-score 鏂规硶锛?
def remove_outliers(data):

    z_scores = zscore(data)

    return data[(np.abs(z_scores) < 3)]  # 閫氬父浣跨敤 3 浣滀负 Z-score 鐨勯槇鍊?
    

from scipy.spatial import ConvexHull

from shapely.geometry import Polygon

from shapely.ops import unary_union

from scipy.stats import norm



def compute_area(points):

    # 璁＄畻鐐圭殑鍑稿寘鍖哄煙

    hull = ConvexHull(points)

    polygon = Polygon(points[hull.vertices])

    return polygon.area



def monte_carlo_r_squared_area(T, mean_params, std_dev_params, real_data, num_simulations=1000):

    original_points = np.column_stack((T, real_data))

    original_area = compute_area(original_points)



    simulation_areas = []

    overlap_areas = []



    for _ in range(num_simulations):

        simulated_path = norm.rvs(

            loc=mean_relation(T, *mean_params),

            scale=np.sqrt(std_dev_relation(T, *std_dev_params)),

            size=len(T)

        )

        simulated_points = np.column_stack((T, simulated_path))

        sim_area = compute_area(simulated_points)

        simulation_areas.append(sim_area)



        original_polygon = Polygon(original_points[ConvexHull(original_points).vertices])

        simulated_polygon = Polygon(simulated_points[ConvexHull(simulated_points).vertices])

        intersection_area = original_polygon.intersection(simulated_polygon).area

        overlap_areas.append(intersection_area)

    

    mean_overlap_area = np.mean(overlap_areas)

    mean_simulation_area = np.mean(simulation_areas)



    # Calculate R2 based on area overlap

    r_squared_area = mean_overlap_area / original_area

    return min(max(r_squared_area, 0), 1)  # Ensure R2 is in [0, 1]


def compute_r2_metrics(y_true, y_pred):
    """Compute uncentered R2 (only).

    R2 = 1 - sum((y - yhat)^2) / sum(y^2).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot_unc = np.sum(y_true ** 2)
    r2_unc = 1.0 - ss_res / ss_tot_unc if ss_tot_unc > 0 else np.nan
    return r2_unc



# 瀹氫箟寮傚父鍊兼竻鐞嗗嚱鏁?
def clean_data(df, columns, threshold=3):

    for col in columns:

        df = df[np.abs(zscore(df[col])) < threshold]

    return df.reset_index(drop=True)



def hour_min_to_float(hour_min_str):

    hour, minute = map(int, hour_min_str.split(":"))

    return hour + minute / 60.0



import re

def scientific_notation_with_superscript(value, precision=3):
    # If value is zero, return plain fixed-point string.
    if value == 0:
        return f"{value:.{precision}f}"

    # Format as scientific notation and normalize exponent string.
    formatted_value = f"{value:.{precision}e}"
    formatted_value = formatted_value.replace("e+", "x10^").replace("e-", "x10^-")
    formatted_value = re.sub(r"x10\^([-+]?)0*(\d+)", r"x10^\1\2", formatted_value)

    # Keep simple ASCII format without Unicode superscripts. Example: 3.21x10^4.
    return formatted_value



from scipy.stats import norm, expon, gamma, lognorm, beta, kstest, shapiro

def normal_distribution_fit_and_test(grouped_by_hour):

    normality_results = {}

    for hour, group in grouped_by_hour:

        # Fit normal distribution and get parameters

        params = norm.fit(group)

        # Perform Shapiro-Wilk test

        _, p_value = shapiro(group)

        # Check if data is normally distributed based on p-value

        is_normal = True if p_value > 0.05 else False

        normality_results[hour] = (

        params[0], params[1], p_value, is_normal)  # params[0]: mean, params[1]: standard deviation



    normality_df = pd.DataFrame(normality_results).T

    normality_df.columns = ["Mean", "Standard Deviation", "P-Value", "Is Normal"]

    normality_df.reset_index(inplace=True)

    normality_df.rename(columns={'index': 'Hour'}, inplace=True)



    return normality_df



#----End Of Cell----



# 浠?dataall 涓瓫閫夊嚭 JH 鍜?CM 鍦扮偣鐨勬暟鎹?
data_jh = dataall[dataall['place'] == 'JH'].copy()

data_cm = dataall[dataall['place'] == 'CM'].copy()

data=dataall.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()



#----End Of Cell----





def create_models_for_location(data_location):

    # 璁＄畻姣忓皬鏃剁殑鍧囧€?
    datare = data_location.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()

    dataj = pd.DataFrame({

        "Time": pd.to_datetime(datare["Hour_Min"]),

        "T": datare["Temperature"],

        "hv": datare["Radiation"],

        "RH": datare["Humidity"],

        "O3": datare["O3"],

        "NOx": datare["NOx"],

        "SO2": datare["SO2"],

        "SOA": datare["SOA"],

        "K": 1,

        'Isoprene': datare["Isoprene"]

    })



    # 璁＄畻浜や簰鍙橀噺

    dataj["HNO3"] = dataj["RH"] * dataj["NOx"]

    dataj["H2SO4"] = dataj["RH"] * dataj["SO2"]

    dataj["H2SO403"] = dataj["RH"] * dataj["SO2"] * dataj["O3"]

    dataj["HNO3O3"] = dataj["RH"] * dataj["NOx"] * dataj["O3"]

    dataj["O3hv"] = dataj["O3"] * dataj["hv"]



    variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]



    # 妯″瀷 1锛氱洿鎺ヤ娇鐢ㄨ娴嬪埌鐨?Isoprene 娴撳害

    dataj["BVOCs"] = datare["Isoprene"]

    for var in variables_to_regress:

        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene"]

    



    # 妯″瀷 2锛氫娇鐢ㄦ俯搴︽嫙鍚?Isoprene 鐨勫潎鍊煎拰鏂瑰樊锛屽啀浠ユ柟宸负鏉冮噸鎷熷悎 SOA

    data_grouped = data_location.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()

    data_grouped["Concentration"] = data_grouped["Isoprene"]

    grouped_by_hour = data_grouped.groupby('Hour')

    normality_df = normal_distribution_fit_and_test(grouped_by_hour["Isoprene"])

    normality_df['T'] = data_grouped.groupby('Hour').mean(numeric_only=True)["Temperature"]



    # 鎻愬彇鏁版嵁

    T = normality_df['T'].values

    mean_values = normality_df['Mean'].astype("float").values

    std_dev_values = normality_df['Standard Deviation'].astype("float").values ** 2



    # 鍘婚櫎绂荤兢鍊?
    mean_values_filtered = remove_outliers(mean_values)

    std_dev_values_filtered = remove_outliers(std_dev_values)

    T_filtered_mean = T[np.isin(mean_values, mean_values_filtered)]

    T_filtered_std_dev = T[np.isin(std_dev_values, std_dev_values_filtered)]



    # 鎷熷悎妯″瀷

    params_mean, _ = curve_fit(mean_relation, T_filtered_mean, mean_values_filtered, method='trf', maxfev=10000)

    params_std_dev, _ = curve_fit(std_dev_relation, T_filtered_std_dev, std_dev_values_filtered, method='trf')



    # 璁＄畻 fitted Isoprene 鍜屾潈閲?
    mean_isoprene = mean_relation(dataj["T"], *params_mean)

    std_dev_isoprene = std_dev_relation(dataj["T"], *params_std_dev)

    dataj["Isoprene_fitted"] = mean_isoprene

    weights = 1 / std_dev_isoprene  # 浣跨敤鏂瑰樊鐨勫€掓暟浣滀负鏉冮噸



    for var in variables_to_regress:

        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]



    X2 = dataj[[var + "_BVOCs" for var in variables_to_regress]]

    Y2 = dataj["SOA"]

    model2 = sm.WLS(Y2, X2, weights=weights).fit(cov_type='HC3')





    # 杈撳嚭妯″瀷

    return model2,params_mean,params_std_dev



#----End Of Cell----



dataall['SOA'].mean()/dataall['PM2.5'].mean()



#----End Of Cell----



# 鏋勫缓 JH 鍜?CM 鍦扮偣鐨勬ā鍨?
modeljh,params_meanjh,params_std_devjh= create_models_for_location(data_jh)

modelcm,params_meancm,params_std_devcm=create_models_for_location(data_cm)



#----End Of Cell----



modelall,params_mean,params_std_dev=create_models_for_location(dataall)



#----End Of Cell----



# data_location=data_jh.copy()

# params_mean=params_meanjh

# params_std_dev=params_std_devjh

# model=modeljh



# data_location=data_cm.copy()

# params_mean=params_meancm

# params_std_dev=params_std_devcm

# model=modelcm



data_location=dataall.copy()

params_mean=params_mean

params_std_dev=params_std_dev

model=modelall



# 璁＄畻姣忓皬鏃剁殑鍧囧€?
datare = data_location.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()

dataj = pd.DataFrame({

    "Time": datare["Hour_Min"],

    "T": datare["Temperature"],

    "hv": datare["Radiation"],

    "RH": datare["Humidity"],

    "O3": datare["O3"],

    "NOx": datare["NOx"],

    "SO2": datare["SO2"],

    "SOA": datare["SOA"],

    "K": 1,

    'Isoprene': datare["Isoprene"]

})



# 璁＄畻浜や簰鍙橀噺

dataj["HNO3"] = dataj["RH"] * dataj["NOx"]

dataj["H2SO4"] = dataj["RH"] * dataj["SO2"]

dataj["H2SO403"] = dataj["RH"] * dataj["SO2"] * dataj["O3"]

dataj["HNO3O3"] = dataj["RH"] * dataj["NOx"] * dataj["O3"]

dataj["O3hv"] = dataj["O3"] * dataj["hv"]



variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]

# attach CS climatology for two-step k_env(CS) scaling
cs_sim = pd.merge(
    datare[["Hour_Min"]],
    cs_hour_min_mean,
    on="Hour_Min",
    how="left"
)
dataj["CS"] = cs_sim["CS"].values


#----End Of Cell----


dataj["Isoprene_fitted"] = mean_relation(dataj["T"], *params_mean)

for var in variables_to_regress:

    dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]


# Baseline linear kernel without CS (first step): reuse original modelall.
X_base = dataj[[var + "_BVOCs" for var in variables_to_regress]]
Y_obs = dataj["SOA"]
Y_base = modelall.predict(X_base)


# Two-step CS scaling: fit k_env(CS) = beta_max * CS / (CS + CS0) to ratio.
def fit_cs_scaling(cs_series, ratio_series):
    cs_clean = cs_series.replace([np.inf, -np.inf], np.nan)
    ratio_clean = ratio_series.replace([np.inf, -np.inf], np.nan)
    mask = cs_clean.notna() & ratio_clean.notna() & (cs_clean > 0) & (ratio_clean > 0)
    x = cs_clean[mask].to_numpy()
    y = ratio_clean[mask].to_numpy()
    if x.size < 5:
        return np.nan, np.nan
    p0 = [np.nanpercentile(y, 90), np.nanmedian(x)]
    popt, _ = curve_fit(
        lambda c, bmax, cs0: bmax * c / (c + cs0),
        x,
        y,
        p0=p0,
        bounds=(0, np.inf),
        maxfev=20000,
    )
    return popt[0], popt[1]


ratio = (Y_obs / Y_base.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
beta_max_cs_linear, cs0_cs_linear = fit_cs_scaling(dataj["CS"], ratio)

if not np.isfinite(beta_max_cs_linear) or not np.isfinite(cs0_cs_linear):
    k_env_series_linear = pd.Series(1.0, index=dataj.index)
else:
    k_env_series_linear = beta_max_cs_linear * dataj["CS"] / (dataj["CS"] + cs0_cs_linear)

# R2 metric for two-step linear+CS kernel (uncentered only).
Y_linear_cs = Y_base * k_env_series_linear.values
r2_linear_unc = compute_r2_metrics(Y_obs, Y_linear_cs)


#----End Of Cell----


# 线性+CS 核的观测值 vs 预测值拟合图（用于诊断 R2 合理性）
fig_fit, ax_fit = plt.subplots(figsize=(5, 5))
ax_fit.scatter(Y_obs, Y_linear_cs, s=15, alpha=0.4, color="#1b9e77", edgecolors="none")

lim = max(np.nanmax(Y_obs), np.nanmax(Y_linear_cs))
ax_fit.plot([0, lim], [0, lim], linestyle="--", color="#d95f02", linewidth=1.2)

ax_fit.set_xlabel(r"Observed SOA ($\mu g/m^3$)")
ax_fit.set_ylabel(r"Predicted SOA ($\mu g/m^3$)")
ax_fit.set_title(f"Linear+CS kernel fit\nR_uc = {r2_linear_unc:.3f}", fontsize=9, weight="bold")
ax_fit.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax_fit.tick_params(axis="both", which="major", labelsize=10)

fig_fit.tight_layout()
plt.savefig(FIGURE_DIR / "SOA_stochastic_linear_CS_fit.png", dpi=500, bbox_inches="tight")
plt.close(fig_fit)


#----End Of Cell----



# 鐢熸垚50鏉￠殢鏈虹殑娓╁害搴忓垪

num_simulations = 1000

dataset=[]



# 鐢熸垚鍘熷鎯呭喌鐨凷OA

mean_isoprene = mean_relation(dataj["T"], *params_mean)

std_dev_isoprene = std_dev_relation(dataj["T"], *params_std_dev)



for i in range(num_simulations):

    random_BVOCs = np.random.normal(loc=mean_isoprene, scale=std_dev_isoprene, size=dataj.shape[0])

    dataj["Isoprene_fitted"] = random_BVOCs

    for var in variables_to_regress:

        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]

    

    X = dataj[[var + "_BVOCs" for var in variables_to_regress]]
    Y_base_sim = modelall.predict(X)
    Y = Y_base_sim * k_env_series_linear.values

    

    temp_data = pd.DataFrame({

        "Time": dataj["Time"],

        "Temperature": dataj["T"],

        "BVOCs": dataj["Isoprene_fitted"],

        "SOA": Y

    })

    dataset.append(temp_data)



# 鐢熸垚1.5degC娓╁崌鎯呮櫙鐨凷OA

mean_isoprene = mean_relation(dataj["T"] + 1.5, *params_mean)

std_dev_isoprene = std_dev_relation(dataj["T"] + 1.5, *params_std_dev)



for i in range(num_simulations):

    random_BVOCs = np.random.normal(loc=mean_isoprene, scale=std_dev_isoprene, size=dataj.shape[0])

    dataj["Isoprene_fitted"] = random_BVOCs

    for var in variables_to_regress:

        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]

    

    X = dataj[[var + "_BVOCs" for var in variables_to_regress]]
    Y_base_sim = modelall.predict(X)
    Y = Y_base_sim * k_env_series_linear.values

    

    temp_data = pd.DataFrame({

        "Time": dataj["Time"],

        "Temperature": dataj["T"] + 1.5,

        "BVOCs": dataj["Isoprene_fitted"],

        "SOA": Y

    })

    dataset.append(temp_data)



# 鐢熸垚2degC娓╁崌鎯呮櫙鐨凷OA

mean_isoprene = mean_relation(dataj["T"] + 2, *params_mean)

std_dev_isoprene = std_dev_relation(dataj["T"] + 2, *params_std_dev)



for i in range(num_simulations):

    random_BVOCs = np.random.normal(loc=mean_isoprene, scale=std_dev_isoprene, size=dataj.shape[0])

    dataj["Isoprene_fitted"] = random_BVOCs

    for var in variables_to_regress:

        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]

    

    X = dataj[[var + "_BVOCs" for var in variables_to_regress]]
    Y_base_sim = modelall.predict(X)
    Y = Y_base_sim * k_env_series_linear.values

    

    temp_data = pd.DataFrame({

        "Time": dataj["Time"],

        "Temperature": dataj["T"] + 2,

        "BVOCs": dataj["Isoprene_fitted"],

        "SOA": Y

    })

    dataset.append(temp_data)



# 鐢熸垚3degC娓╁崌鎯呮櫙鐨凷OA

mean_isoprene = mean_relation(dataj["T"] + 3, *params_mean)

std_dev_isoprene = std_dev_relation(dataj["T"] + 3, *params_std_dev)



for i in range(num_simulations):

    random_BVOCs = np.random.normal(loc=mean_isoprene, scale=std_dev_isoprene, size=dataj.shape[0])

    dataj["Isoprene_fitted"] = random_BVOCs

    for var in variables_to_regress:

        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]

    

    X = dataj[[var + "_BVOCs" for var in variables_to_regress]]
    Y_base_sim = modelall.predict(X)
    Y = Y_base_sim * k_env_series_linear.values

    

    temp_data = pd.DataFrame({

        "Time": dataj["Time"],

        "Temperature": dataj["T"] + 3,

        "BVOCs": dataj["Isoprene_fitted"],

        "SOA": Y

    })

    dataset.append(temp_data)



#----End Of Cell----



# # 鐢ㄤ簬瀛樺偍姣忎釜 model 鐨勫潎鍊艰秴鏍囧拰鎬讳綋瓒呮爣姣斾緥

# exceedance_dict = {}

# 

# # 鎬绘椂闂寸偣鏁伴噺

# total_time_points = dataset[0].shape[0]

# 

# # 閬嶅巻姣忎釜妯℃嫙鎯呮櫙鏁版嵁闆嗗苟璁＄畻瓒呮爣姣斾緥

# for data in dataset:

#     # 鏍规嵁娓╁害鍊煎垽鏂綋鍓嶆暟鎹殑妯″紡

#     if data["Temperature"].iloc[0] == dataj["T"].iloc[0]:

#         model_label = "Baseline"

#     elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 1.5:

#         model_label = "+1.5degC Warming"

#     elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 2:

#         model_label = "+2.0degC Warming"

#     elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 3:

#         model_label = "+3.0degC Warming"

#     else:

#         continue  # 璺宠繃涓嶅尮閰嶄换浣曟儏鏅殑鏁版嵁

# 

#     # 鍒濆鍖?model 鐨勮褰?
#     if model_label not in exceedance_dict:

#         exceedance_dict[model_label] = {"mean_exceedance": 0, "total_exceedance": 0, "count": 0, "exceed_times": set()}

# 

#     # 璁＄畻 mean_exceedance锛氭寜 Time 鍒嗙粍璁＄畻 SOA 鐨勫潎鍊煎苟妫€鏌ユ槸鍚﹁秴杩?5

#     mean_exceedance = (data.groupby("Time")["SOA"].mean() > 5).mean() * 100

#     

#     # 鎵惧嚭璇ヨ矾寰勪腑鎵€鏈夎秴鏍囩殑鏃堕棿鐐?
#     exceed_times = set(data.loc[data["SOA"] > 5, "Time"])

# 

#     # 鏇存柊瓒呮爣鏃堕棿鐐归泦鍚堬紙闆嗗悎鑷姩鍘婚噸锛?
#     exceedance_dict[model_label]["exceed_times"].update(exceed_times)

# 

#     # 绱姞 mean_exceedance

#     exceedance_dict[model_label]["mean_exceedance"] += mean_exceedance

#     exceedance_dict[model_label]["count"] += 1

# 

# # 璁＄畻姣忎釜 model 鐨勫钩鍧囪秴鏍囨瘮渚嬪拰鎬讳綋瓒呮爣姣斾緥

# for model in exceedance_dict:

#     # 璁＄畻鍧囧€艰秴鏍囨瘮渚?
#     exceedance_dict[model]["mean_exceedance"] /= exceedance_dict[model]["count"]

#     # 璁＄畻鎬昏秴鏍囨瘮渚?
#     exceedance_dict[model]["total_exceedance"] = len(exceedance_dict[model]["exceed_times"]) / total_time_points * 100







#----End Of Cell----



combined_data = pd.DataFrame()



# 鍒濆鍖栧瓧鍏革紝鐢ㄤ簬璺熻釜姣忎釜 model 鐨勯噰鏍锋鏁?
sample_counts = {"Baseline": 0, "+1.5degC Warming": 0, "+2.0degC Warming": 0, "+3.0degC Warming": 0}



# 閬嶅巻姣忎釜妯℃嫙鎯呮櫙骞舵坊鍔犲埌 combined_data 涓紝鎺у埗姣忎釜 model 鏈€澶?10 娆?
for data in dataset:

    # 鏍规嵁娓╁害鍊煎垽鏂綋鍓嶆暟鎹殑妯″紡骞舵坊鍔犲埌鏂板垪 'model'

    if data["Temperature"].iloc[0] == dataj["T"].iloc[0]:

        model_label = "Baseline"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 1.5:

        model_label = "+1.5degC Warming"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 2:

        model_label = "+2.0degC Warming"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 3:

        model_label = "+3.0degC Warming"

    else:

        continue  # 濡傛灉涓嶅尮閰嶄换浣曟儏鏅紝璺宠繃褰撳墠鏁版嵁



    # 濡傛灉褰撳墠 model 宸茬粡鏈?10 娆￠噰鏍凤紝鍒欒烦杩?
    if sample_counts[model_label] >= 10:

        continue



    # 璁剧疆 model 鍒楋紝骞舵坊鍔犲埌 combined_data 涓?
    data["model"] = model_label

    combined_data = pd.concat([combined_data, data], ignore_index=True)

    

    # 鏇存柊閲囨牱娆℃暟

    sample_counts[model_label] += 1



#----End Of Cell----



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import patches, lines

from adjustText import adjust_text

import matplotlib.patheffects as path_effects

from statsmodels.nonparametric.smoothers_lowess import lowess



colors = ['#464AA6', '#F2CB05', '#F28A2E', '#BF2633']

# 璁剧疆鍥惧舰灏哄

fig = plt.figure(figsize=(8, 8))

gs = fig.add_gridspec(2,2)

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])

ax3 = fig.add_subplot(gs[1, :])









# 绗竴寮犲浘锛歋OA瀵嗗害鍥?(subplot 1)

sns.kdeplot(data=combined_data, x='SOA', hue="model", fill=True, palette=colors,ax=ax1)

ax1.set_title("")

ax1.set_xlabel(r"SOA ($\mu g/m^3$)", fontsize=14, fontweight="bold")

ax1.set_ylabel("Density", fontsize=14, fontweight="bold")

ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 浼樺寲鍒诲害鏍囩鐨勬牱寮?
ax1.tick_params(axis='both', which='major', labelsize=12, width=1.3)

ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Y杞存牸寮忓寲鏄剧ず涓や綅灏忔暟

ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Y杞存牸寮忓寲鏄剧ず涓や綅灏忔暟

ax1.legend_.remove()  







# Subplot 2: SOA versus temperature with model annotation (ASCII-safe math text).
# The expression follows the mechanism reconstruction note:
# SOA = k_env(CS) * C_T * M_theta(X) = beta_max * CS/(CS + CS_0) * C_T * M_theta(X).
model_label_tex = (
    r"$SOA = k_{env}(CS)\,C_T\,\mathcal{M}_{\theta}(\mathbf{X})$" + "\n"
    r"$k_{env}(CS) = \beta_{\max}\dfrac{CS}{CS + CS_0}$"
)



# 閬嶅巻姣忕鎯呮櫙锛岀粯鍒禠OESS骞虫粦鏇茬嚎鍜岄€忔槑鏁ｇ偣

for i,model in enumerate(combined_data["model"].unique()):

    model_data = combined_data[combined_data["model"] == model]

    # 浣跨敤 LOESS 杩涜骞虫粦

    loess_result = lowess(model_data["SOA"], model_data["Temperature"], frac=0.3)

    # 缁樺埗骞虫粦鏇茬嚎

    ax2.plot(

        loess_result[:, 0], loess_result[:, 1],

        label=f"{model} LOESS fit",

        linewidth=2.5,zorder=10,color=colors[i])





# 鐢ㄤ簬璁板綍姣忎釜 model 宸茬粯鍒剁殑璺緞鏁伴噺

plot_count = {
    "Baseline": 0,
    "+1.5$^{\\circ}$C Warming": 0,
    "+2.0$^{\\circ}$C Warming": 0,
    "+3.0$^{\\circ}$C Warming": 0,
}

# 閬嶅巻姣忎釜璺緞鏁版嵁闆嗗苟缁樺埗锛岀‘淇濇瘡涓?model 浠呯粯鍒?10 鏉¤矾寰?
for data in dataset:

    # 纭畾褰撳墠璺緞鐨?model 绫诲瀷

    if data["Temperature"].iloc[0] == dataj["T"].iloc[0]:

        model_label = "Baseline"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 1.5:

        model_label = "+1.5$^{\\circ}$C Warming"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 2:

        model_label = "+2.0$^{\\circ}$C Warming"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 3:

        model_label = "+3.0$^{\\circ}$C Warming"

    else:

        continue  # 璺宠繃涓嶅尮閰嶄换浣曟儏鏅殑鏁版嵁



    # 濡傛灉褰撳墠 model 宸茬粯鍒朵簡 10 鏉¤矾寰勶紝鍒欒烦杩?
    if plot_count[model_label] >= 5:

        continue

    # 缁樺埗璺緞

    sns.lineplot(data=data, x="Temperature", y="SOA", ax=ax2, color="grey", alpha=0.1, linewidth=0.7, linestyle='-')

    # 鏇存柊缁樺埗璺緞璁℃暟

    plot_count[model_label] += 1

    



# 璁剧疆鏍囬鍜屽潗鏍囪酱鏍囩鐨勬牸寮?
ax2.set_title("", pad=15)

ax2.set_xlabel(r"Temperature ($^{\circ}$C)", fontsize=14, fontweight="bold")

ax2.set_ylabel(r"Average SOA ($\mu g/m^3$)", fontsize=14, fontweight="bold")

# 璁剧疆缃戞牸鏍峰紡

ax2.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)

# 浼樺寲鍒诲害鏍囩鐨勬牱寮?
ax2.tick_params(axis='both', which='major', labelsize=12, width=1.3)

ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Y杞存牸寮忓寲鏄剧ず涓や綅灏忔暟

ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Y杞存牸寮忓寲鏄剧ず涓や綅灏忔暟
for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
    lbl.set_fontweight("bold")
    lbl.set_fontsize(12)

# 璁＄畻 SOA 鐨勫潎鍊煎拰鏍囧噯宸?
mean_soa = dataall["SOA"].mean()

std_soa = dataall["SOA"].std()

# 缁樺埗鍧囧€肩嚎

ax2.axhline(y=mean_soa, color='#63E398', linestyle='--', linewidth=2, label="Mean SOA")

# 缁樺埗姝ｈ礋 2 涓爣鍑嗗樊

ax2.axhline(y=mean_soa + 1 * std_soa, color='#B1CE46', linestyle='--', linewidth=2, label="Mean + 1 Std Dev")

ax2.axhline(y=mean_soa + 2 * std_soa, color='#F1D77E', linestyle='--', linewidth=2, label="Mean + 2 Std Dev")

ax2.text(
    0.14,
    0.93,
    f"Model R_uc = {r2_linear_unc:.3f}\n" + model_label_tex,
    transform=ax2.transAxes,
    fontsize=14,
    color="black",
    fontweight="bold",
    verticalalignment="top",
)

ax2.legend_.remove()  







#绗笁寮犲浘

# 鐢ㄤ簬璁板綍姣忎釜 model 宸茬粯鍒剁殑璺緞鏁伴噺

plot_count = {"Baseline": 0, "+1.5degC Warming": 0, "+2.0degC Warming": 0, "+3.0degC Warming": 0}

# 閬嶅巻姣忎釜璺緞鏁版嵁闆嗗苟缁樺埗锛岀‘淇濇瘡涓?model 浠呯粯鍒?10 鏉¤矾寰?
for data in dataset:

    # 纭畾褰撳墠璺緞鐨?model 绫诲瀷

    if data["Temperature"].iloc[0] == dataj["T"].iloc[0]:

        model_label = "Baseline"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 1.5:

        model_label = "+1.5degC Warming"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 2:

        model_label = "+2.0degC Warming"

    elif data["Temperature"].iloc[0] == dataj["T"].iloc[0] + 3:

        model_label = "+3.0degC Warming"

    else:

        continue  # 璺宠繃涓嶅尮閰嶄换浣曟儏鏅殑鏁版嵁



    # 濡傛灉褰撳墠 model 宸茬粯鍒朵簡 10 鏉¤矾寰勶紝鍒欒烦杩?
    if plot_count[model_label] >= 10:

        continue



    # 缁樺埗璺緞

    sns.lineplot(data=data, x="Time", y="SOA", ax=ax3, color="grey", alpha=0.3, linewidth=0.7, linestyle='-')



    # 鏇存柊缁樺埗璺緞璁℃暟

    plot_count[model_label] += 1

# # 缁樺埗姣忎釜鎯呮櫙鐨勫潎鍊兼洸绾?
sns.lineplot(data=combined_data, x="Time", y="SOA", hue="model", ax=ax3, linewidth=2, palette=colors, marker=None)

# 鐢ㄤ簬瀛樺偍娉ㄩ噴鐨勬枃鏈璞″垪琛?
texts = []

# 璁＄畻骞舵爣娉ㄦ瘡涓儏鏅殑鍧囧€艰秴鏍囧拰鎬讳綋瓒呮爣姣斾緥

for i,model in enumerate(combined_data["model"].unique()):

    model_data = combined_data[combined_data["model"] == model]

    # 鑾峰彇姣忎釜鎯呮櫙鐨勪腑蹇冧綅缃?
    center_time = model_data["Time"].iloc[dataset[0].shape[0] // 2]

    center_soa = model_data["SOA"].mean()

    mean=model_data["SOA"].mean()

    std=model_data["SOA"].std()

    text = ax3.text(

    center_time, center_soa,

    f"{model}\nMean={mean:.2f}\nStd={std:.2f}",

    ha='center', fontsize=14, fontweight="bold", color='black'

)

    texts.append(text)

    # 娣诲姞鐧借壊鎻忚竟鏁堟灉

    for text in texts:

        text.set_path_effects([

            path_effects.Stroke(linewidth=1.5, foreground="white"),

            path_effects.Normal()

        ])

# 浣跨敤 adjust_text 閬垮厤鏂囧瓧閲嶅彔

adjust_text(

    texts,

    ax=ax3,

    arrowprops=dict(arrowstyle="-|>", color='black', zorder=10),  # 璁剧疆绠ご鐨?zorder

)

# 璁＄畻 SOA 鐨勫潎鍊煎拰鏍囧噯宸?
mean_soa = dataall["SOA"].mean()

std_soa = dataall["SOA"].std()

# 缁樺埗鍧囧€肩嚎

ax3.axhline(y=mean_soa, color='#63E398', linestyle='--', linewidth=2, label="Mean SOA")

# 缁樺埗姝ｈ礋 2 涓爣鍑嗗樊

ax3.axhline(y=mean_soa + 1 * std_soa, color='#B1CE46', linestyle='--', linewidth=2, label="Mean + 1 Std Dev")

ax3.axhline(y=mean_soa + 2 * std_soa, color='#F1D77E', linestyle='--', linewidth=2, label="Mean + 2 Std Dev")

# 璁剧疆鏍囬鍜屾爣绛?
ax3.set_title("", )

ax3.set_xlabel("", )

ax3.set_ylabel(r"SOA ($\mu g/m^3$)", fontsize=14, fontweight="bold")

ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 璁剧疆 x 杞寸殑鏄剧ず闂撮殧涓烘瘡 4 灏忔椂

ax3.set_xticks(ax3.get_xticks()[::90])  

# 浼樺寲鏍囩

ax3.tick_params(axis='both', which='major', labelsize=12, width=1.3)

ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Y杞存牸寮忓寲鏄剧ず涓や綅灏忔暟

ax3.tick_params(axis='x', labelrotation=45, labelsize=10)

ax3.legend_.remove()  

# 鑷畾涔夊浘渚?
from matplotlib.lines import Line2D

import matplotlib.patches as mpatches

# 鑷畾涔夊浘渚嬮」

legend_elements = [

    mpatches.Patch(color='#464AA6', label="Baseline"),

    mpatches.Patch(color='#F2CB05', label="+1.5$^{\\circ}$C Warming"),

    mpatches.Patch(color='#F28A2E', label="+2.0$^{\\circ}$C Warming"),

    mpatches.Patch(color='#BF2633', label="+3.0$^{\\circ}$C Warming"),

    Line2D([0], [0], color='#63E398', linestyle='--', linewidth=2, label="Mean SOA"),

    Line2D([0], [0], color='#B1CE46', linestyle='--', linewidth=2, label="Mean + 1 Std Dev"),

    Line2D([0], [0], color='#F1D77E', linestyle='--', linewidth=2, label="Mean + 2 Std Dev"),

    Line2D([0], [0], color='grey', linestyle='-', linewidth=2, alpha=1, label="Monte Carlo\nSimulation")

]

# 鍦ㄥぇ鍥惧彸渚ф坊鍔犲浘渚?
fig.legend(
    handles=legend_elements,
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    prop={'size': 12, 'weight': 'bold'},
    frameon=False
)




# 璋冩暣甯冨眬锛岀‘淇濈編瑙?
fig.tight_layout()
fig.canvas.draw()  # ensure ticks are created before styling

for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.3)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="bold")
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="bold")
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")

plt.savefig(FIGURE_DIR / 'SOA_stochastic_linear_CS.png', dpi=500, bbox_inches='tight')



#----End Of Cell----





#----End Of Cell----







