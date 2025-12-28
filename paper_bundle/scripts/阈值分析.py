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
plt.rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "legend.fontsize": 14,
    }
)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.metrics import mean_squared_error
from pathlib import Path

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib import patheffects
from adjustText import adjust_text
from src.workflow.modeling_framework import (
    compute_cs,
    default_config,
    load_cached_results,
    load_base_data,
)

#----End Of Cell----

cfg = default_config()
df_base = load_base_data(cfg)
df_sde, cat1_outputs, cat2_outputs, ml_outputs, labels_cfg = load_cached_results()

# VOCs list
VOCs = ['Methyl Mercaptan', '1,3-Butadiene', 'Butene', 'Acetone/Butane', 'n-Propanol',
        'Dimethyl Sulfide/Ethyl Mercaptan', 'Chloroethane', 'Isoprene', 'Pentene', 'Pentane/Isopentane',
        'Dimethylformamide', 'Ethyl Formate', 'Carbon Disulfide/Propyl Mercaptan', 'Benzene', 'Cyclohexene',
        'Hexene/Methylcyclopentane', 'n-Hexane/Dimethylbutane', 'Ethyl Sulfide/Butyl Mercaptan', 'Toluene', 'Aniline',
        'Dimethyl Disulfide', '1,1-Dichloroethylene', 'Methylcyclohexane', 'n-Heptane', 'Triethylamine',
        'n-Propyl Acetate', 'Diethylene Triamine', 'Styrene', 'Xylene/Ethylbenzene', '1,3-Dichloropropene', 'n-Octane',
        'n-Butyl Acetate', 'Hexyl Mercaptan', 'Xylenol', 'Trichloroethylene', 'Diethylbenzene', 'Methyl Benzoate',
        'Trimethyl Phosphate', 'n-Decanol', 'Dichlorobenzene', 'Diethyl Aniline', 'Undecane', 'Tetrachloroethylene',
        'n-Dodecane', 'Dibromomethane', '1,2,4-Trichlorobenzene', 'n-Tridecane', '1,2-Dibromoethane']

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

# References: Fuchs and Sutugin (1971) condensation sink theory; CS = sum(4*pi*D_v*r*F(Kn)*N).
# Equations: CS(Time) provided by Category II outputs; aggregated to CS(Hour_Min) for threshold analysis kernels.
cs_series_cached = cat2_outputs.get("cs")
cs_hour_min_mean = None
if isinstance(cs_series_cached, pd.Series) and not cs_series_cached.empty:
    cs_frame = pd.DataFrame({"Time": cs_series_cached.index, "CS": cs_series_cached.values})
    cs_frame["Hour_Min"] = cs_frame["Time"].dt.strftime("%H:%M")
    cs_hour_min_mean = cs_frame.groupby("Hour_Min")["CS"].mean(numeric_only=True).reset_index()

# Fallback to compute CS from base data if cache is unavailable.
if cs_hour_min_mean is None:
    cfg_cs = default_config()
    number_cols = [c for c in df_base.columns if c.startswith("C") and c.endswith("um")]
    if not number_cols:
        raise ValueError("No C*um columns found for CS calculation in threshold analysis.")
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
    cs_input["CS"] = compute_cs(cs_input[["temperature_c"] + number_cols], cfg_cs)
    cs_hour_min_mean = cs_input.groupby("Hour_Min")["CS"].mean(numeric_only=True).reset_index()


#----End Of Cell----

from scipy.optimize import curve_fit
from scipy.stats import zscore
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates
from scipy.stats import t


# 定义与均值的关系函数（二次函数）
def mean_relation(T, Q0, a, v0):
    return Q0 + (a * T**2) / 2 + T * v0

# 定义与方差的关系函数（三次函数）
def std_dev_relation(T, k, sigma0):
    return (k**2 * T**3) / 3 + k * T**2 * sigma0 + T * sigma0**2


# 定义去除离群值的函数（使用 Z-score 方法）
def remove_outliers(data):
    z_scores = zscore(data)
    return data[(np.abs(z_scores) < 3)]  # 通常使用 3 作为 Z-score 的阈值
    
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.stats import norm

def compute_area(points):
    # 计算点的凸包区域
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

    # Calculate R² based on area overlap
    r_squared_area = mean_overlap_area / original_area
    return min(max(r_squared_area, 0), 1)  # Ensure R² is in [0, 1]

# 定义异常值清理函数
def clean_data(df, columns, threshold=3):
    for col in columns:
        df = df[np.abs(zscore(df[col])) < threshold]
    return df.reset_index(drop=True)

def hour_min_to_float(hour_min_str):
    hour, minute = map(int, hour_min_str.split(":"))
    return hour + minute / 60.0

import re
def scientific_notation_with_superscript(value, precision=3):
    # 如果值为零，直接返回零字符串格式
    if value == 0:
        return f"{value:.{precision}f}"
    
    # 格式化为科学记数法，并去掉指数部分的前导零
    formatted_value = f"{value:.{precision}e}"
    formatted_value = formatted_value.replace('e+', 'x10^').replace('e-', 'x10^-')
    formatted_value = re.sub(r'x10\^([-+]?)0*(\d+)', r'x10^\1\2', formatted_value)  # 保留负号
    
    # 将指数部分的数字替换为 Unicode 上标
    superscript_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    match = re.search(r'x10\^([-+]?\d+)', formatted_value)
    if match:
        exponent = match.group(1).translate(superscript_map)
        formatted_value = re.sub(r'x10\^[-+]?\d+', f'x10{exponent}', formatted_value)
         # 如果是 x10⁰ 则移除
        if 'x10⁰' in formatted_value:
            formatted_value = formatted_value.replace('x10⁰', '')
    else:
        # 如果没有 x10^，直接返回数值部分
        formatted_value = formatted_value.split('x10')[0]
         # 如果是 x10⁰ 则移除
        if 'x10⁰' in formatted_value:
            formatted_value = formatted_value.replace('x10⁰', '')
        
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

# 从 dataall 中筛选出 JH 和 CM 地点的数据
data_jh = dataall[dataall['place'] == 'JH'].copy()
data_cm = dataall[dataall['place'] == 'CM'].copy()
data = dataall.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()

#----End Of Cell----



def create_models_for_location(data_location):
    # 计算每小时的均值
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

    # 计算交互变量
    dataj["HNO3"] = dataj["RH"] * dataj["NOx"]
    dataj["H2SO4"] = dataj["RH"] * dataj["SO2"]
    dataj["H2SO403"] = dataj["RH"] * dataj["SO2"] * dataj["O3"]
    dataj["HNO3O3"] = dataj["RH"] * dataj["NOx"] * dataj["O3"]
    dataj["O3hv"] = dataj["O3"] * dataj["hv"]

    variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]

    # 模型 1：直接使用观测到的 Isoprene 浓度
    dataj["BVOCs"] = datare["Isoprene"]
    for var in variables_to_regress:
        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene"]

    # 模型 2：使用温度拟合 Isoprene 的均值和方差，再以方差为权重拟合 SOA
    data_grouped = data_location.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    data_grouped["Concentration"] = data_grouped["Isoprene"]
    grouped_by_hour = data_grouped.groupby('Hour')
    normality_df = normal_distribution_fit_and_test(grouped_by_hour["Isoprene"])
    normality_df['T'] = data_grouped.groupby('Hour').mean(numeric_only=True)["Temperature"]

    # 提取数据
    T = normality_df['T'].values
    mean_values = normality_df['Mean'].astype("float").values
    std_dev_values = normality_df['Standard Deviation'].astype("float").values ** 2

    # 去除离群值
    mean_values_filtered = remove_outliers(mean_values)
    std_dev_values_filtered = remove_outliers(std_dev_values)
    T_filtered_mean = T[np.isin(mean_values, mean_values_filtered)]
    T_filtered_std_dev = T[np.isin(std_dev_values, std_dev_values_filtered)]

    # 拟合模型
    params_mean, _ = curve_fit(mean_relation, T_filtered_mean, mean_values_filtered, method='trf', maxfev=10000)
    params_std_dev, _ = curve_fit(std_dev_relation, T_filtered_std_dev, std_dev_values_filtered, method='trf')

    # 计算 fitted Isoprene 和权重
    mean_isoprene = mean_relation(dataj["T"], *params_mean)
    std_dev_isoprene = std_dev_relation(dataj["T"], *params_std_dev)
    dataj["Isoprene_fitted"] = mean_isoprene
    weights = 1 / std_dev_isoprene  # 使用方差的倒数作为权重

    for var in variables_to_regress:
        dataj[var + "_BVOCs"] = dataj[var] * dataj["Isoprene_fitted"]

    X2 = dataj[[var + "_BVOCs" for var in variables_to_regress]]
    Y2 = dataj["SOA"]
    model2 = sm.WLS(Y2, X2, weights=weights).fit(cov_type='HC3')

    # 输出模型
    return model2, params_mean, params_std_dev

#----End Of Cell----

modelall,params_mean,params_std_dev=create_models_for_location(dataall)

#----End Of Cell----

data_location=dataall.copy()
params_mean=params_mean
params_std_dev=params_std_dev
model=modelall

# Build Hour_Min climatology for CS scaling using the same linear kernel as in the stochastic script.
datare_cs = data_location.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
dataj_cs = pd.DataFrame({
    "Hour_Min": datare_cs["Hour_Min"],
    "T": datare_cs["Temperature"],
    "hv": datare_cs["Radiation"],
    "RH": datare_cs["Humidity"],
    "O3": datare_cs["O3"],
    "NOx": datare_cs["NOx"],
    "SO2": datare_cs["SO2"],
    "SOA": datare_cs["SOA"],
    "K": 1,
    "Isoprene": datare_cs["Isoprene"],
})
dataj_cs["HNO3"] = dataj_cs["RH"] * dataj_cs["NOx"]
dataj_cs["H2SO4"] = dataj_cs["RH"] * dataj_cs["SO2"]
dataj_cs["H2SO403"] = dataj_cs["RH"] * dataj_cs["SO2"] * dataj_cs["O3"]
dataj_cs["HNO3O3"] = dataj_cs["RH"] * dataj_cs["NOx"] * dataj_cs["O3"]
dataj_cs["O3hv"] = dataj_cs["O3"] * dataj_cs["hv"]
variables_to_regress_cs = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]
dataj_cs["Isoprene_fitted"] = mean_relation(dataj_cs["T"], *params_mean)
for var in variables_to_regress_cs:
    dataj_cs[var + "_BVOCs"] = dataj_cs[var] * dataj_cs["Isoprene_fitted"]
X_base_cs = dataj_cs[[var + "_BVOCs" for var in variables_to_regress_cs]]
Y_obs_cs = dataj_cs["SOA"]
Y_base_cs = modelall.predict(X_base_cs)

cs_cs = pd.merge(
    datare_cs[["Hour_Min"]],
    cs_hour_min_mean,
    on="Hour_Min",
    how="left"
)
dataj_cs["CS"] = cs_cs["CS"].values

def fit_cs_scaling_threshold(cs_series, ratio_series):
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

ratio_cs = (Y_obs_cs / Y_base_cs.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
beta_max_cs_thr, cs0_cs_thr = fit_cs_scaling_threshold(dataj_cs["CS"], ratio_cs)
if not np.isfinite(beta_max_cs_thr) or not np.isfinite(cs0_cs_thr):
    k_env_mean_thr = 1.0
else:
    k_env_series_thr = beta_max_cs_thr * dataj_cs["CS"] / (dataj_cs["CS"] + cs0_cs_thr)
    k_env_mean_thr = float(k_env_series_thr.mean())

# 计算每小时的均值
datare = data_location.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
dataj = pd.DataFrame({
    "T": datare["Temperature"].mean(),
    "hv": datare["Radiation"].mean(),
    "RH": datare["Humidity"].mean(),
    "O3": datare["O3"].mean(),
    "NOx": datare["NOx"].mean(),
    "SO2": datare["SO2"].mean(),
    "SOA": datare["SOA"].mean(),
    "K": 1,
    'Isoprene': datare["Isoprene"].mean()
}, index=[0])

# 计算交互变量
dataj["HNO3"] = dataj["RH"] * dataj["NOx"]
dataj["H2SO4"] = dataj["RH"] * dataj["SO2"]
dataj["H2SO403"] = dataj["RH"] * dataj["SO2"] * dataj["O3"]
dataj["HNO3O3"] = dataj["RH"] * dataj["NOx"] * dataj["O3"]
dataj["O3hv"] = dataj["O3"] * dataj["hv"]

variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]

#----End Of Cell----

from scipy.stats import norm

T_values = np.linspace(dataall["Temperature"].mean(), dataall["Temperature"].mean() + 2 * dataall["Temperature"].std(), 15)
thresholds = [
    dataall["SOA"].mean() + dataall["SOA"].std(),
    dataall["SOA"].mean() + 2 * dataall["SOA"].std(),
    dataall["SOA"].mean() + 3 * dataall["SOA"].std(),
]
num_simulations = 1000
simulation_size = 100
probability_results = {}

#----End Of Cell----

# Monte Carlo simulation
for threshold in thresholds:
    probability_results[threshold] = {f"N>={i}": [] for i in range(1, 9)}
    for T in T_values:
        exceedances = []
        for _ in range(num_simulations):
            simulated_BVOCs = norm.rvs(
                loc=mean_relation(T, *params_mean),
                scale=np.sqrt(std_dev_relation(T, *params_std_dev)),
                size=simulation_size
            )
            dataj_repeated = dataj.loc[dataj.index.repeat(simulation_size // len(dataj))].copy()
            dataj_repeated['Isoprene'] = np.tile(simulated_BVOCs, len(dataj_repeated) // simulation_size)
            # Update dependent variables
            for var in variables_to_regress:
                dataj_repeated[var + "_BVOCs"] = dataj_repeated[var] * dataj_repeated["Isoprene"]
            X = dataj_repeated[[var + "_BVOCs" for var in variables_to_regress]]
            simulated_SOA_base = model.predict(X)
            simulated_SOA = simulated_SOA_base * k_env_mean_thr
            # Count exceedances
            exceedances.append(np.sum(simulated_SOA > threshold))
        for i in range(1, 9):
            probability_results[threshold][f"N>={i}"].append(np.mean(np.array(exceedances) >= i))


#----End Of Cell----

from palettable.tableau import Tableau_20
colors = Tableau_20.mpl_colors
# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()
threshold_labels = ["Mean + 1 standard deviation", "Mean + 2 standard deviation", "Mean + 3 standard deviation"]

for idx, (threshold, ax) in enumerate(zip(thresholds, axes)):
    for i in range(1, 9):
        ax.plot(
            T_values,
            probability_results[threshold][f"N>={i}"],
            label=f"N>={i}",
            color=colors[i - 1],
            marker="o",
            markersize=6,
            linewidth=2
        )
    # Set titles and axis labels
    ax.set_title(f"Threshold: {threshold_labels[idx]}", pad=10, fontsize=16, fontweight='bold')
    ax.set_xlabel("Temperature (T)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Probability", fontsize=15, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5)
    ax.legend( loc="lower right", frameon=False, handlelength=2.5, borderpad=0.5,  prop={'size': 14, 'weight': 'bold'})
    ax.grid(visible=True, linestyle='--', alpha=0.7)

    # Add SOA threshold text at the top of the plot
    if idx!=2:
        ax.text(
            0.75, 0.7, f"SOA Threshold = {threshold:.2f}", 
            fontsize=15, weight='bold', ha='center', va='bottom', transform=ax.transAxes
        )
    else:
        ax.text(
            0.3, 0.7, f"SOA Threshold = {threshold:.2f}", 
            fontsize=15, weight='bold', ha='center', va='bottom', transform=ax.transAxes
        )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

# Adjust figure title, layout, and save
plt.tight_layout()
plt.subplots_adjust()
plt.savefig(FIGURE_DIR / "SOA_extreme_exceedance.png", dpi=500, bbox_inches='tight')
plt.show()

#----End Of Cell----


#----End Of Cell----

