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
from pathlib import Path

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_JH = BUNDLE_ROOT / "groupedjhS.csv"
DATA_CM = BUNDLE_ROOT / "groupedcmS.csv"
DATA_JH_SOA = BUNDLE_ROOT / "groupedjhSOA.csv"
DATA_CM_SOA = BUNDLE_ROOT / "groupedcmSOA.csv"
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

#----End Of Cell----

random.seed(20231125)
np.random.seed(20231125)

datajh = pd.read_csv(DATA_JH)
datacm = pd.read_csv(DATA_CM)
datajhsoa = pd.read_csv(DATA_JH_SOA)
datacmsoa = pd.read_csv(DATA_CM_SOA)

#----End Of Cell----

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

from scipy.optimize import curve_fit


# 定义与均值 Mean 的关系的函数
def mean_relation(T, Q0, a, v0):
    return Q0 + (a * T ** 2) / 2 + T * v0


# 定义与方差 variance 的关系的函数
def std_dev_relation(T, k, sigma0):
    return ((k ** 2) * (T ** 3)) / 3 + k * (T ** 2) * sigma0 + T * (sigma0 ** 2)


from scipy.stats import zscore
from scipy.stats import t


# 定义去除离群值的函数（使用 Z-score 方法）
def remove_outliers(data):
    z_scores = zscore(data)
    return data[(np.abs(z_scores) < 3)]  # 通常使用 3 作为 Z-score 的阈值


# 定义加权二次关系模型
def weighted_quadratic_relation(T, y, weights):
    # 增加常数项和二次项
    T_with_const = sm.add_constant(np.column_stack((T, T**2)))
    # 加权最小二乘法（WLS）
    model_wls = sm.WLS(y, T_with_const, weights=weights).fit()
    return model_wls.rsquared  # 返回加权的 R² 值

def clean_data(df, columns, threshold=3):
    for col in columns:
        df = df[np.abs(zscore(df[col])) < threshold]
    return df.reset_index(drop=True)

#----End Of Cell----

from scipy.stats import (
    gaussian_kde,
    norm,
    uniform,
    gamma,
    weibull_min,
    pearson3  # 用于对数皮尔逊III型分布
)

def kl_divergence(p, q, x_vals):
    """
    计算两个概率密度之间的KL散度
    """
    return np.sum(p * np.log(p / (q + 1e-10))) * (x_vals[1] - x_vals[0])

def adaptive_penalty_coefficient(sim_var, real_var):
    variance_ratio = sim_var / real_var
    if variance_ratio > 1:
        return np.exp(variance_ratio)
        # return 0
    else:
        # return np.log(variance_ratio+1) / np.log(100)  # 如果模拟方差不超过真实方差，则轻微惩罚
        return 0  

def monte_carlo_density_r_squared(
    T,
    X,
    model,
    mean_params,
    std_dev_params,
    real_data,
    num_simulations=1000,
    distribution='uniform'  # 新增参数，默认为 'uniform'
):
    """
    基于概率密度的蒙特卡罗模拟方法计算调整后的 R²，考虑自适应的方差惩罚。

    T: 自变量（例如温度）
    mean_params: 拟合的均值函数参数
    std_dev_params: 拟合的方差函数参数
    real_data: 原始观测数据
    num_simulations: 模拟次数
    distribution: 零模型的分布类型，支持 'normal', 'uniform', 'gamma', 'weibull', 'logpearson3'
    """
    # 计算观测数据的概率密度
    density_obs = gaussian_kde(real_data)
    x_vals = np.linspace(np.min(real_data), np.max(real_data), 100)
    p_obs = density_obs(x_vals)
    
    simv=[]
    ov=[]

    # 根据指定的分布类型计算零模型的概率密度
    if distribution == 'normal':
        mean_obs, std_obs = np.mean(real_data), np.std(real_data)
        p_null = norm.pdf(x_vals, loc=mean_obs, scale=std_obs)
    elif distribution == 'uniform':
        a, b = np.min(real_data), np.max(real_data)
        p_null = uniform.pdf(x_vals, loc=a, scale=b - a)
    elif distribution == 'gamma':
        params = gamma.fit(real_data, floc=0)  # 固定 loc=0，避免负值
        shape, loc, scale = params
        p_null = gamma.pdf(x_vals, a=shape, loc=loc, scale=scale)
    elif distribution == 'weibull':
        params = weibull_min.fit(real_data, floc=0)  # 固定 loc=0
        c, loc, scale = params
        p_null = weibull_min.pdf(x_vals, c=c, loc=loc, scale=scale)
    elif distribution == 'logpearson3':
        # 对数据取对数
        positive_data = real_data[real_data > 0]  # 确保数据为正
        log_data = np.log(positive_data)
        # 拟合 Pearson III 型分布
        skew, loc, scale = pearson3.fit(log_data)
        # 计算对数空间的概率密度
        log_x_vals = np.log(x_vals[x_vals > 0])  # x_vals 也要为正
        p_null_log = pearson3.pdf(log_x_vals, skew=skew, loc=loc, scale=scale)
        # 转换回原始空间的概率密度
        p_null = np.zeros_like(x_vals)
        p_null[x_vals > 0] = (1 / x_vals[x_vals > 0]) * p_null_log
    else:
        raise ValueError(f"不支持的分布类型：{distribution}")

    kl_null = kl_divergence(p_obs, p_null, x_vals)

    # 模拟路径的KL散度和自适应惩罚
    kl_total = 0
    penalty_total = 0
    
    
    # # 动态计算每个数据点的 k
    # k_values = model.predict(X) / mean_relation(T, *mean_params)
    # 
    # # 使用 std_dev_relation 计算 VOCs 的方差
    # std_dev_soa = k_values * np.sqrt(std_dev_relation(T, *std_dev_params))
    meansoa=model.predict(X)
    residuals = meansoa - model.predict(X)
    std_dev_soa = residuals
    

    for _ in range(num_simulations):
        # 生成模拟路径
        simulated_path = np.random.normal(
            loc=meansoa,
            scale=std_dev_soa,
            size=len(T)
        )
      
        # 计算模拟路径的概率密度
        density_sim = gaussian_kde(simulated_path)
        p_sim = density_sim(x_vals)
        
        # 计算观测数据与模拟数据的KL散度
        kl_sim = kl_divergence(p_obs, p_sim, x_vals)
        kl_total += kl_sim
        
        # 计算模拟路径的方差，并计算自适应惩罚
        sim_var = np.var(simulated_path)
        real_var = np.var(real_data)
        penalty = adaptive_penalty_coefficient(sim_var, real_var)
        penalty_total += penalty
        simv.append(sim_var)
        ov.append(real_var)

    # 计算平均KL散度和平均自适应惩罚
    kl_avg = kl_total / num_simulations
    mean_penalty = penalty_total / num_simulations
    # 计算调整后的 R²，考虑自适应的方差惩罚
    adjusted_r_squared = (1 - (kl_avg / kl_null)) / (1 + mean_penalty)
    adjusted_r_squared = np.clip(adjusted_r_squared, 0, 1)  # 限制 R² 在 [0, 1] 之间
    print(mean_penalty)

    return adjusted_r_squared

#----End Of Cell----

def crps_normal_analytical(mu, sigma, y):
    """
    References: Gneiting and Raftery (2007, Journal of the American Statistical
    Association); Hersbach (2000, Weather and Forecasting).
    Equation: CRPS(N(mu,sigma^2), y) = sigma * [w * (2*Phi(w) - 1)
    + 2*phi(w) - 1/sqrt(pi)], where w = (y - mu) / sigma.
    Parameters: mu are predictive means; sigma are predictive standard
    deviations; y are observed SOA concentrations.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(sigma, 1e-12)
    w = (y - mu) / sigma
    return sigma * (w * (2.0 * norm.cdf(w) - 1.0) + 2.0 * norm.pdf(w) - 1.0 / np.sqrt(np.pi))

def crps_skill_score_gaussian(mu, sigma, y, hours=None):
    """
    References: same as crps_normal_analytical, using CRPS-based skill score
    CRPSS = 1 - CRPS_model / CRPS_ref.
    Equation: CRPS_model and CRPS_ref are temporal averages of analytical
    CRPS values computed for a Gaussian stochastic SOA process and a
    climatological normal reference N(mu_ref, sigma_ref^2), respectively.
    If hours are provided, CRPS is first computed at the Hour_Min-aggregated
    scale and then averaged within each Hour group so that every hour
    contributes equally.
    Parameters: mu and sigma are predictive means and standard deviations
    at the Hour_Min scale; y are observed SOA concentrations at the same
    Hour_Min times; hours is an optional array of Hour identifiers.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if not (mu.shape == sigma.shape == y.shape):
        raise ValueError("mu, sigma and y must have the same length.")

    sigma = np.maximum(sigma, 1e-12)
    crps_model_values = crps_normal_analytical(mu, sigma, y)

    mu_ref = float(np.mean(y))
    sigma_ref = float(np.std(y))
    if sigma_ref <= 0:
        return float(np.nanmean(crps_model_values)), np.nan

    crps_ref_values = crps_normal_analytical(mu_ref, sigma_ref, y)

    if hours is not None:
        hours_arr = np.asarray(hours)
        if hours_arr.shape[0] != y.shape[0]:
            raise ValueError("hours must have the same length as y.")
        unique_hours = np.unique(hours_arr)
        per_hour_model = []
        per_hour_ref = []
        for h in unique_hours:
            mask_h = hours_arr == h
            if not np.any(mask_h):
                continue
            m_h = float(np.nanmean(crps_model_values[mask_h]))
            r_h = float(np.nanmean(crps_ref_values[mask_h]))
            if np.isfinite(m_h) and np.isfinite(r_h) and r_h > 0:
                per_hour_model.append(m_h)
                per_hour_ref.append(r_h)
        if per_hour_model:
            crps_model = float(np.mean(per_hour_model))
            crps_ref = float(np.mean(per_hour_ref))
        else:
            crps_model = float(np.nanmean(crps_model_values))
            crps_ref = float(np.nanmean(crps_ref_values))
    else:
        crps_model = float(np.nanmean(crps_model_values))
        crps_ref = float(np.nanmean(crps_ref_values))

    if crps_ref <= 0:
        return crps_model, np.nan

    crpss = 1.0 - crps_model / crps_ref
    return crps_model, crpss

#----End Of Cell----

# 加载已保存的 KMeans 模型；若文件不存在则退化为按顺序编号的聚类标签
try:
    kmeans = joblib.load('kmeans.joblib')
    chemical_clusters = dict(zip(VOCs, kmeans.labels_))
except FileNotFoundError:
    # Fallback: use index-based cluster labels when pretrained model is not available
    chemical_clusters = {name: idx for idx, name in enumerate(VOCs)}

def analyze_chemical_T(data, chemicals):
    # 计算每小时的平均值
    datare = data.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()

    # 准备 DataFrame
    dataj = pd.DataFrame({
        "Time": pd.to_datetime(datare["Hour_Min"]),
        "T": datare["Temperature"],
        "hv": datare["Radiation"],
        "RH": datare["Humidity"],
        "O3": datare.O3,
        "NOx": datare.NOx,
        "SO2": datare.SO2,
        "SOA": datare.SOA,
        "K": 1,
    })

    results = []  # 存储每种化学物质的结果

    for chemical in chemicals:
        
        cluster_label = chemical_clusters.get(chemical, None)

        data_grouped = data.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
        data_grouped["Concentration"] = data_grouped[chemical]
        grouped_by_hour = data_grouped.groupby('Hour')
        normality_df = normal_distribution_fit_and_test(grouped_by_hour[chemical])
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
        params_mean, covariance_mean = curve_fit(mean_relation, T_filtered_mean, mean_values_filtered, method='trf',
                                                 maxfev=10000)
        params_std_dev_filtered, covariance_std_dev = curve_fit(std_dev_relation, T_filtered_std_dev,
                                                                std_dev_values_filtered, method='trf')

        # 添加化学物质浓度（VOC）
        dataj["VOC"] = mean_relation(dataj['T'], *params_mean)
        dataj["VOC_std"] = std_dev_relation(dataj['T'], *params_std_dev_filtered)
        weights = 1/dataj["VOC_std"]
        weights = weights / np.sum(weights)
        # 计算交互变量
        dataj["HNO3"] = dataj["RH"] * dataj["NOx"]
        dataj["H2SO4"] = dataj["RH"] * dataj["SO2"]
        dataj["H2SO403"] = dataj["RH"] * dataj["SO2"] * dataj["O3"]
        dataj["HNO3O3"] = dataj["RH"] * dataj["NOx"] * dataj["O3"]
        dataj["O3hv"] = dataj["O3"] * dataj["hv"]
        
        

        # 将这些变量乘以VOC
        variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]
        for var in variables_to_regress:
            dataj[var + "_VOC"] = dataj[var] * dataj["VOC"]

        # 选择变量进行线性回归
        X = dataj[[var + "_VOC" for var in variables_to_regress]]
        # X=sm.add_constant(X)
        Y = dataj["SOA"]

        # 进行线性回归
        model = sm.WLS(Y, X, weights=weights).fit(cov_type='HC3')

        # 预测值
        y_pred = model.predict(X)
        # 计算 MSE
        mse = mean_squared_error(Y, y_pred)

        # 计算 RMSE
        rmse = np.sqrt(mse)

        # 计算 NMSE（基于方差归一化的 NMSE）
        var_y = np.var(Y)  # 目标变量 Y 的方差
        nmse = mse / var_y  # 方差归一化的 NMSE

        # 从模型对象中提取信息并存储结果
        coefficients_pvalues = {f'{param}': f'{coef:.4f} (p={pval:.3f})' for param, coef, pval in
                                zip(X.columns, model.params, model.pvalues)}

        
        # rtotal=  monte_carlo_density_r_squared(data_grouped['Temperature'].values,
        # X.values, model,params_mean, params_std_dev_filtered,Y.values.T
        # )
        
        # # 逐步删除 VIF 高的变量
        # max_vif = 10  # VIF 阈值
        # removed_features = []  # 记录被删除的特征
        # while True:
        #     vif = pd.DataFrame()
        #     vif["feature"] = X.columns
        #     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        #     
        #     # 找到最大 VIF 值的特征
        #     max_vif_value = vif["VIF"].max()
        #     max_vif_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
        #     
        #     # 检查是否需要删除特征
        #     if max_vif_value > max_vif:
        #         X = X.drop(columns=[max_vif_feature])
        #         removed_features.append(max_vif_feature)
        #     else:
        #         break
        # 
        # # 执行加权最小二乘回归
        # model = sm.WLS(Y, X, weights=weights).fit(cov_type='HC3')
        # 
        # # 预测值与评价指标
        # y_pred = model.predict(X)
        # mse = mean_squared_error(Y, y_pred)
        # rmse = np.sqrt(mse)
        # var_y = np.var(Y)
        # nmse = mse / var_y
        # 
        # # 提取系数和 p 值，未选择的特征标记为 None
        # coefficients_pvalues = {
        #     feature: f'{model.params[feature]:.4f} (p={model.pvalues[feature]:.3f})' if feature in model.params else None
        #     for feature in [var + "_VOC" for var in variables_to_regress]
        # }
        
        dataj["VOCo"] = datare[chemical] 
        
        # 将这些变量乘以VOC
        variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]
        for var in variables_to_regress:
            dataj[var + "_VOC"] = dataj[var] * dataj["VOCo"]
        
        # 选择变量进行线性回归
        X2 = dataj[[var + "_VOC" for var in variables_to_regress]]
        # X=sm.add_constant(X)
        Y2 = dataj["SOA"]
        
        # 进行线性回归
        model2 = sm.WLS(Y2, X2, weights=weights).fit(cov_type='HC3')
        

        

 
        result = {
            'Chemical': chemical,
            'Cluster': cluster_label+1,
            'Average Concentration': np.mean(dataj["VOC"]),
            # 'R2 Total':rtotal,
            'Concentration Std Dev': np.std(dataj["VOC"]),
            'R2': model.rsquared,
            'R2BVOCs':model2.rsquared,
            'MSE': mse,
            'RMSE': rmse,
            'NMSE': nmse,  # 归一化 MSE
            'AIC':model.aic,
            'P-Value': model.f_pvalue,
            **coefficients_pvalues
        }
        results.append(result)

    return pd.DataFrame(results)

#----End Of Cell----

# 计算每种VOC的平均浓度
average_concentrations = dataall[VOCs].mean()
# 计算所有VOCs的平均总浓度
total_concentration = average_concentrations.sum()
# 计算每种VOC占总浓度的百分比
concentration_percentage = (average_concentrations / total_concentration) * 100
# 筛选出占比大于2%的VOCs
selected_chemicals = concentration_percentage[concentration_percentage > 2].index.tolist()

#----End Of Cell----

chemicals = selected_chemicals

#----End Of Cell----

# 处理 JH 地点的化学物质数据
results_jh = analyze_chemical_T(dataall[dataall.place == 'JH'], chemicals)
results_df_jh = pd.DataFrame(results_jh)
results_df_jh['Place'] = 'JH'  # 添加地点信息

# 处理 CM 地点的化学物质数据
results_cm = analyze_chemical_T(dataall[dataall.place == 'CM'], chemicals)
results_df_cm = pd.DataFrame(results_cm)
results_df_cm['Place'] = 'CM'  # 添加地点信息

#----End Of Cell----

# 合并 JH 和 CM 的结果
results_df = pd.concat([results_df_jh, results_df_cm], axis=0)

results_df.set_index(['Chemical', 'Cluster', 'Place'], inplace=True)

jh_r2_mean = results_df.xs('CM', level='Place')['R2']
sorted_index = jh_r2_mean.sort_values(ascending=False).index.get_level_values('Chemical')
results_df_sorted = results_df.loc[sorted_index]

# 显示结果
results_df_sorted

#----End Of Cell----


#----End Of Cell----


#----End Of Cell----


#----End Of Cell----

chemical = 'Isoprene'

# 分别处理 JH 和 CM 地点的数据
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
places = ['JH', 'CM']
results = {}

for i,place in enumerate(places):
    data=dataall[dataall.place == place]
     
    # 计算每小时的平均值
    datare = data.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    
    # 准备 DataFrame
    dataj = pd.DataFrame({
        "Time": pd.to_datetime(datare["Hour_Min"]),
        "T": datare["Temperature"],
        "hv": datare["Radiation"],
        "RH": datare["Humidity"],
        "O3": datare.O3,
        "NOx": datare.NOx,
        "SO2": datare.SO2,
        "SOA": datare.SOA,
        "K": 1
    })
    
     
    data_grouped = data.groupby(["Hour_Min"]).mean(numeric_only=True).reset_index()
    data_grouped["Concentration"] = data_grouped[chemical]
    grouped_by_hour = data_grouped.groupby('Hour')
    normality_df = normal_distribution_fit_and_test(grouped_by_hour[chemical])
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
    params_mean, covariance_mean = curve_fit(mean_relation, T_filtered_mean, mean_values_filtered, method='trf',
                                             maxfev=10000)
    params_std_dev_filtered, covariance_std_dev = curve_fit(std_dev_relation, T_filtered_std_dev,
                                                            std_dev_values_filtered, method='trf')
    
    # 添加化学物质浓度（VOC）
    dataj["VOC"] = mean_relation(dataj['T'], *params_mean)
    dataj["VOC_std"] = std_dev_relation(dataj['T'], *params_std_dev_filtered)
    weights = 1/dataj["VOC_std"]
    
    # 计算交互变量
    dataj["HNO3"] = dataj["RH"] * dataj["NOx"]
    dataj["H2SO4"] = dataj["RH"] * dataj["SO2"]
    dataj["H2SO403"] = dataj["RH"] * dataj["SO2"] * dataj["O3"]
    dataj["HNO3O3"] = dataj["RH"] * dataj["NOx"] * dataj["O3"]
    dataj["O3hv"] = dataj["O3"] * dataj["hv"]
    
    # 将这些变量乘以VOC
    variables_to_regress = ["HNO3", "H2SO4", "H2SO403", "HNO3O3", "O3hv", "K", "hv"]
    for var in variables_to_regress:
        dataj[var + "_VOC"] = dataj[var] * dataj["VOC"]
    
    # 选择变量进行线性回归
    X = dataj[[var + "_VOC" for var in variables_to_regress]]
    Y = dataj["SOA"]
    
    
    
     # 逐步删除 VIF 高的变量
    max_vif = 10  # VIF 阈值
    removed_features = []  # 记录被删除的特征
    while True:
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # 找到最大 VIF 值的特征
        max_vif_value = vif["VIF"].max()
        max_vif_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
        
        # 检查是否需要删除特征
        if max_vif_value > max_vif:
            X = X.drop(columns=[max_vif_feature])
            removed_features.append(max_vif_feature)
        else:
            break

    # 执行加权最小二乘回归
    model = sm.WLS(Y, X, weights=weights).fit(cov_type='HC3')

    
    # 进行线性回归
    model = sm.WLS(Y, X, weights=weights).fit(cov_type='HC3')
    
    # 预测值
    y_pred = model.predict(X)
    # 计算 MSE
    mse = mean_squared_error(Y, y_pred)
    
    # 计算 RMSE
    rmse = np.sqrt(mse)
    
    # 计算 NMSE（基于方差归一化的 NMSE）
    var_y = np.var(Y)  # 目标变量 Y 的方差
    nmse = mse / var_y  # 方差归一化的 NMSE
    
    # 从模型对象中提取信息并存储结果
    coefficients_pvalues = {f'{param}': f'{coef:.4f} (p={pval:.3f})' for param, coef, pval in
                            zip(X.columns, model.params, model.pvalues)}
   
    meansoa = model.predict(X)

    # 将 WLS 预测均值与温度响应的方差模型组合为 SOA 随机过程的标准差
    k_values = model.predict(X) / mean_relation(data_grouped['Temperature'], *params_mean)
    std_dev_soa = k_values * np.sqrt(
        std_dev_relation(data_grouped['Temperature'], *params_std_dev_filtered)
    ).values

    ax_hour = axes[i]

    def hour_min_to_float(hour_min_str):
        hour, minute = map(int, hour_min_str.split(":"))
        return hour + minute / 60.0

    data_grouped['Hour_Float'] = data_grouped['Hour_Min'].apply(hour_min_to_float)

    # 观测 SOA 的 Hour_Min 数据点
    ax_hour.scatter(
        data_grouped['Hour_Float'],
        data_grouped['SOA'],
        color='#F1D77E',
        alpha=0.3,
        marker='o',
        s=20,
        zorder=10,
    )

    # Monte Carlo 随机过程样本路径
    ns = 50
    cmap = sns.light_palette("grey", as_cmap=True)
    palette = [cmap(x) for x in np.linspace(0.2, 0.85, ns)]
    for j in range(ns):
        simulated_path = norm.rvs(
            loc=meansoa,
            scale=std_dev_soa,
            size=data_grouped['Temperature'].shape[0],
        )
        ax_hour.plot(
            data_grouped['Hour_Float'],
            simulated_path,
            color=palette[j],
            alpha=0.5,
        )

    # 基于 Hour_Min 尺度的 SOA 随机过程计算 CRPS 及相关诊断量
    mu_hourmin = np.asarray(meansoa, dtype=float)
    sigma_hourmin = np.asarray(std_dev_soa, dtype=float)
    y_hourmin = data_grouped['SOA'].values.astype(float)
    hours_hourmin = data_grouped['Hour'].values

    crps_value, crpss = crps_skill_score_gaussian(
        mu_hourmin,
        sigma_hourmin,
        y_hourmin,
        hours=hours_hourmin,
    )

    bias_abs = float(np.mean(np.abs(mu_hourmin - y_hourmin)))
    lower_90 = mu_hourmin - 1.645 * sigma_hourmin
    upper_90 = mu_hourmin + 1.645 * sigma_hourmin
    coverage_90 = float(
        np.mean((y_hourmin >= lower_90) & (y_hourmin <= upper_90))
    )

    spread_ratios = []
    for h_val in np.unique(hours_hourmin):
        mask_h = hours_hourmin == h_val
        y_h = y_hourmin[mask_h]
        if y_h.size < 2:
            continue
        obs_sd_h = float(np.std(y_h, ddof=1))
        if obs_sd_h <= 0:
            continue
        sigma_h = float(np.median(sigma_hourmin[mask_h]))
        spread_ratios.append(sigma_h / obs_sd_h)
    if spread_ratios:
        r_spread = float(np.median(spread_ratios))
    else:
        r_spread = np.nan

    ax_hour.text(
        0.05,
        0.9,
        f'{place}:\nCRPS = {crps_value:.3f}, CRPSS = {crpss:.3f}\n'
        r'$|\mathrm{bias}|$'
        f' = {bias_abs:.2f}, '
        r'$R_{\mathrm{spread}}$'
        f' = {r_spread:.2f}\n'
        r'$C_{90}$'
        f' = {coverage_90:.2f}',
        transform=ax_hour.transAxes,
        verticalalignment='top',
        fontsize=8, bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'),
        zorder=100,
    )

    ax_hour.set_xlabel("Hour")
    ax_hour.set_ylabel(r"SOA ($\mu g/m^3$)")
    ax_hour.set_xticks([0, 4, 8, 12, 16, 20, 24])

# 添加子图间距并保存图形
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'SOA_stochastic_process.png', dpi=500, bbox_inches='tight')
plt.show()

#----End Of Cell----

