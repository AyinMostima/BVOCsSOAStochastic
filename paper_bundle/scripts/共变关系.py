import pandas as pd
import numpy as np
import random
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
plt.rcParams['figure.facecolor'] = 'white'  # 设置全局图形背景颜色为白色
plt.rcParams['axes.facecolor'] = 'white'    # 设置全局绘图区背景颜色为白色

from pathlib import Path

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_JH = BUNDLE_ROOT / "groupedjhS.csv"
DATA_CM = BUNDLE_ROOT / "groupedcmS.csv"
DATA_JH_SOA = BUNDLE_ROOT / "groupedjhSOA.csv"
DATA_CM_SOA = BUNDLE_ROOT / "groupedcmSOA.csv"

#----End Of Cell----

np.random.seed(20231125)
random.seed(20231125)

datajh = pd.read_csv(DATA_JH)
datacm = pd.read_csv(DATA_CM)
datajhsoa = pd.read_csv(DATA_JH_SOA)
datacmsoa = pd.read_csv(DATA_CM_SOA)

#----End Of Cell----

datajh['place']='JH'
datacm['place']='CM'

#----End Of Cell----


from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.barycenters import softdtw_barycenter
transfer = MinMaxScaler(feature_range=(0, 1))
from kneed import KneeLocator

#----End Of Cell----

VOCs = ['Methyl Mercaptan', '1,3-Butadiene', 'Butene', 'Acetone/Butane', 'n-Propanol', 'Dimethyl Sulfide/Ethyl Mercaptan', 'Chloroethane', 'Isoprene', 'Pentene', 'Pentane/Isopentane', 'Dimethylformamide', 'Ethyl Formate', 'Carbon Disulfide/Propyl Mercaptan', 'Benzene', 'Cyclohexene', 'Hexene/Methylcyclopentane', 'n-Hexane/Dimethylbutane', 'Ethyl Sulfide/Butyl Mercaptan', 'Toluene', 'Aniline', 'Dimethyl Disulfide', '1,1-Dichloroethylene', 'Methylcyclohexane', 'n-Heptane', 'Triethylamine', 'n-Propyl Acetate', 'Diethylene Triamine', 'Styrene', 'Xylene/Ethylbenzene', '1,3-Dichloropropene', 'n-Octane', 'n-Butyl Acetate', 'Hexyl Mercaptan', 'Xylenol', 'Trichloroethylene', 'Diethylbenzene', 'Methyl Benzoate', 'Trimethyl Phosphate', 'n-Decanol', 'Dichlorobenzene', 'Diethyl Aniline', 'Undecane', 'Tetrachloroethylene', 'n-Dodecane', 'Dibromomethane', '1,2,4-Trichlorobenzene', 'n-Tridecane', '1,2-Dibromoethane']

#----End Of Cell----

columns_to_keep = ['甲硫醇浓度', '1,3-丁二烯浓度', '丁烯浓度', '丙酮、丁烷浓度', '正丙醇浓度', '甲硫醚、乙硫醇浓度',
                   '氯乙烷浓度', '异戊二烯浓度', '戊烯浓度', '戊烷、异戊烷浓度', '二甲基甲酰胺浓度', '甲酸乙酯浓度',
                   '二硫化碳、丙硫醇浓度', '苯浓度', '环己烯浓度', '己烯、甲基环戊烷浓度', '正己烷、二甲基丁烷浓度',
                   '乙硫醚、丁硫醇浓度',
                   '甲苯浓度', '苯胺浓度', '二甲基二硫醚浓度', '1,1-二氯乙烯浓度', '甲基环己烷浓度', '正庚烷浓度',
                   '三乙胺浓度',
                   '乙酸正丙酯浓度', '二亚乙基三胺浓度', '苯乙烯浓度', '二甲苯、乙苯浓度', '1,3-二氯丙烯浓度',
                   '正辛烷浓度',
                   '乙酸正丁酯浓度', '己硫醇浓度', '二甲苯酚浓度', '三氯乙烯浓度', '二乙基苯浓度', '苯甲酸甲酯浓度',
                   '磷酸三甲酯浓度',
                   '正癸醇浓度', '二氯苯浓度', '二乙基苯胺浓度', '十一烷浓度', '四氯乙烯浓度', '正十二烷浓度',
                   '二溴甲烷浓度',
                   '1,2,4-三氯苯浓度', '正十三烷浓度', '1,2-二溴乙烷浓度', 'Hour_Min','Time','place']

# Keep only the specified columns and merge the data
data = pd.concat([datajh[columns_to_keep], datacm[columns_to_keep]], axis=0)
data.columns = ['Methyl Mercaptan', '1,3-Butadiene', 'Butene', 'Acetone/Butane', 'n-Propanol', 'Dimethyl Sulfide/Ethyl Mercaptan', 'Chloroethane', 'Isoprene', 'Pentene', 'Pentane/Isopentane', 'Dimethylformamide', 'Ethyl Formate', 'Carbon Disulfide/Propyl Mercaptan', 'Benzene', 'Cyclohexene', 'Hexene/Methylcyclopentane', 'n-Hexane/Dimethylbutane', 'Ethyl Sulfide/Butyl Mercaptan', 'Toluene', 'Aniline', 'Dimethyl Disulfide', '1,1-Dichloroethylene', 'Methylcyclohexane', 'n-Heptane', 'Triethylamine', 'n-Propyl Acetate', 'Diethylene Triamine', 'Styrene', 'Xylene/Ethylbenzene', '1,3-Dichloropropene', 'n-Octane', 'n-Butyl Acetate', 'Hexyl Mercaptan', 'Xylenol', 'Trichloroethylene', 'Diethylbenzene', 'Methyl Benzoate', 'Trimethyl Phosphate', 'n-Decanol', 'Dichlorobenzene', 'Diethyl Aniline', 'Undecane', 'Tetrachloroethylene', 'n-Dodecane', 'Dibromomethane', '1,2,4-Trichlorobenzene', 'n-Tridecane', '1,2-Dibromoethane', 'Hour_Min','Time','place']

#----End Of Cell----

data=data.groupby(['Hour_Min']).mean(numeric_only=True).reset_index()

#----End Of Cell----

# 提取小时信息
data['Hour'] = data['Hour_Min'].apply(lambda x: int(x.split(':')[0]))

#----End Of Cell----

# data=data.groupby(['Hour']).mean().reset_index()

#----End Of Cell----

data

#----End Of Cell----


#----End Of Cell----

from scipy.interpolate import interp1d
def transform_to_same_length(x,  max_length):
    n = x.shape[0]
    # x的形状(n,var,length)

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, 1), dtype=np.float64)


    mts = x
    curr_length = n
    idx = np.array(range(curr_length))
    idx_new = np.linspace(0, idx.max(), max_length)
    # linear interpolation
    f = interp1d(idx, mts, kind='cubic')
    new_ts = f(idx_new)

    return new_ts

#----End Of Cell----

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataclu=[]
for i in VOCs:
    dataclu.append(transform_to_same_length(data[i].values,250))
# 对 dataclu 中的每个时间序列进行归一化
normalized_dataclu = []
for series in dataclu:
    # series 的形状是 (250, 1)，需要进行归一化
    normalized_series = scaler.fit_transform(series.reshape(-1, 1))
    normalized_dataclu.append(normalized_series)
dataclu=normalized_dataclu

#----End Of Cell----

from tqdm import tqdm
MODEL_PATH = BUNDLE_ROOT / "kmeans.joblib"
if MODEL_PATH.exists():
    kmeans = joblib.load(MODEL_PATH)
else:
    inertias = []
    max_clusters = 10  # 尝试最多10个簇
    for n_clusters in tqdm(range(1, max_clusters + 1), desc="Clustering Progress"):
        kmeans_tmp = TimeSeriesKMeans(n_clusters=n_clusters, n_init=5, max_iter=100, metric="dtw", n_jobs=-1)
        kmeans_tmp.fit(dataclu)
        inertias.append(kmeans_tmp.inertia_)

    kneedle = KneeLocator(range(1, max_clusters + 1), inertias, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow or 3
    kmeans = TimeSeriesKMeans(n_clusters=optimal_k, n_init=50, max_iter=1000, metric="dtw", n_jobs=-1, random_state=132)
    kmeans.fit(dataclu)
    joblib.dump(kmeans, MODEL_PATH)

#----End Of Cell----

# 找到所有标签为 2 的索引
indices = np.where(kmeans.labels_ == 1)[0]

# 使用这些索引从 VOCs 列表中获取对应的 VOC 名称
matched_vocs = [VOCs[i] for i in indices]

print(f"标签为 2 的 VOC 是: {matched_vocs}")

#----End Of Cell----

motifs=dataclu
model=kmeans
nclu=3
import matplotlib.lines as mlines
from palettable.tableau import Tableau_20
from palettable.tableau import PurpleGray_12
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from matplotlib.patches import ConnectionPatch

#----End Of Cell----

# 计算每个 VOC 的浓度均值，并获取排名前15的 VOCs
mean_concentrations = {voc: np.mean(data[voc]) for voc in VOCs}
top_15_vocs = sorted(mean_concentrations, key=mean_concentrations.get, reverse=True)[:20]

# 获取 Tableau_20 调色板
colors = Tableau_20.mpl_colors
# 创建子图
fig, ax = plt.subplots(nclu, 1, sharex=True, figsize=(8, 8))
fig.subplots_adjust(hspace=0.3)  # 调整子图之间的间距

# 计算每个聚类中的 VOCs 数量
vocs_count_by_cluster = [sum(1 for label in model.labels_ if label == yi) for yi in range(nclu)]
texts_by_ax = [[] for _ in range(nclu)]  # 创建一个列表来存储每个 ax 的文本对象
# 绘制每个 motif，逐个子图进行调整
for yi in range(nclu):
    for idx, (motif, label) in enumerate(zip(motifs, model.labels_)):
        if label == yi:  # 仅绘制当前子图的曲线
            if VOCs[idx] in top_15_vocs:
                # 使用对应的颜色和加粗线条绘制
                color_idx = top_15_vocs.index(VOCs[idx]) % len(colors)
                ax[yi].plot(motif, alpha=0.8, lw=1.7, c=colors[color_idx], label=VOCs[idx])

                # 添加箭头和文本标注
                x_position = len(motif) // 2  # 在曲线的中间位置添加标注
                y_position = motif[x_position]  # 获取曲线中间点的 y 值
                text = ax[yi].text(x_position, y_position, VOCs[idx], fontsize=14, color=colors[color_idx], fontweight='bold')
                text.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground='white'), path_effects.Normal()])
                texts_by_ax[yi].append(text)  # 将文本对象保存到对应的 ax 的列表中
            else:
                # 默认使用灰色和较细的线条绘制
                ax[yi].plot(motif, alpha=0.6, lw=1.2, c="gray", linestyle=":")
    
    
     # 添加每个聚类的 VOCs 数量注释
    ax[yi].text(0.01, 0.95, f"{vocs_count_by_cluster[yi]} VOCs", transform=ax[yi].transAxes,
                fontsize=16, fontweight='bold', color='black', ha='left', va='top')
    

    ts = texts_by_ax[yi]

    # 防御式：确保 ts 是 list 且每个元素是 Text
    if ts and len(ts) >= 2:
        ys = np.array([float(t.get_position()[1]) for t in ts], dtype=float)

        order = np.argsort(ys)  # numpy array of ints

        # 逐个整数索引访问，绝不把 list 当 index
        for k, j in enumerate(order):
            j = int(j)
            x, y = ts[j].get_position()
            if k % 2 == 0:
                ts[j].set_position((x - 35, y))   # 左移
            else:
                ts[j].set_position((x + 35, y))   # 右移

        # 底部两条兜底：同样逐个 int
        bottom2 = np.argsort(ys)[:2]
        for j in bottom2:
            j = int(j)
            x, y = ts[j].get_position()
            ts[j].set_position((x - 30, y))


    

    # 第一阶段：只排字，不画箭头（更容易收敛）
    adjust_text(
        texts_by_ax[yi],
        ax=ax[yi],
        force_text=(1.5, 2.8),
        expand_text=(1.10, 1.60),
        lim=4000,
        precision=0.01,
    )

    # 第二阶段：保持风格加箭头，轻微调整
    adjust_text(
        texts_by_ax[yi],
        ax=ax[yi],
        force_text=(0.6, 1.2),
        expand_text=(1.05, 1.20),
        lim=1500,
        precision=0.02,
        arrowprops=dict(
            arrowstyle='->',
            color='#9467bd',
            lw=2,
            alpha=0.7,
            shrinkA=18,   # 由于退化成 annotate，这里必须显著增大
            shrinkB=12,
        ),
    )

    # 文字始终在最上层
    for t in texts_by_ax[yi]:
        t.set_zorder(6)

    # 箭头压到文字下面，但在线条上面
    for p in ax[yi].patches:
        p.set_zorder(5)
        p.set_clip_on(False)




# 绘制聚类中心
for i, center in enumerate(model.cluster_centers_):
    ax[i].set_title(f"Cluster {i+1}", fontsize=14, fontweight='bold', pad=10)
    ax[i].plot(center.ravel(), c="#B95756", alpha=0.8, lw=2.7, label="Cluster Center", linestyle='dashdot')

    # 删除 Y 轴标签
    ax[i].set_yticklabels([])

    # 仅在中间的图上标注 Y 轴标签
    if i == 1:
        ax[i].set_ylabel("Concentration (Scale)", fontsize=16, fontweight='bold')

    # 去除所有网格线
    ax[i].grid(axis='x')



# 设置 X 轴刻度为 24 小时分布的间隔，每 2 小时
hours = np.linspace(0, 24, 13)
ax[-1].set_xticks(np.linspace(0, 250, len(hours)))
ax[-1].set_xticklabels([f"{int(hour)}:00" for hour in hours], fontsize=16,rotation=45)

# 创建图例并将其放在右侧外部
gray_line = mlines.Line2D([], [], color='gray', label='Other VOCs', linestyle=':')
red_line = mlines.Line2D([], [], color='#B95756', label='Cluster Center', linestyle='dashdot')
handles = [gray_line, red_line] + [mlines.Line2D([], [], color=colors[top_15_vocs.index(voc) % len(colors)], lw=2.5, label=voc) for voc in top_15_vocs]
fig.legend(handles=handles, frameon=False, loc="center right", fontsize=12, bbox_to_anchor=(1.4, 0.5),ncol=1)

# # # 优化布局
plt.tight_layout()
plt.savefig(FIGURE_DIR / "VOC_covariance_clusters.png", dpi=500, bbox_inches='tight')

plt.show()

#----End Of Cell----

# # 创建子图
# fig, ax = plt.subplots(nclu, 1, sharex=True, figsize=(10, 10))
# fig.subplots_adjust(hspace=0.3)  # 调整子图之间的间距
# 
# # 绘制每个 motif
# for motif, label in zip(motifs, model.labels_):
#     ax[label].plot(motif, alpha=0.6, lw=1.2, c="gray", linestyle='--')
# 
# # 绘制聚类中心
# for i, center in enumerate(model.cluster_centers_):
#     ax[i].set_title(f"Cluster {i+1}", fontsize=14, fontweight='bold', pad=10)
#     ax[i].plot(center.ravel(), c="#B95756", alpha=0.8, lw=2.5, label="Cluster Center")
# 
#     # 删除 Y 轴标签
#     ax[i].set_yticklabels([])
#     
#     # 仅在中间的图上标注 Y 轴标签
#     if i == 1:
#         ax[i].set_ylabel("Concentration (Scale)", fontsize=16, fontweight='bold')
#     
#     # 去除所有网格线
#     ax[i].grid(False)
# 
# # 设置 X 轴刻度为 24 小时分布的间隔，每 2 小时
# hours = np.linspace(0, 24, 13)  # 生成 13 个点，从 0 到 24 小时
# ax[-1].set_xticks(np.linspace(0, 250, len(hours)))  # 250 个数据点平均分配到 24 小时
# ax[-1].set_xticklabels([f"{int(hour)}:00" for hour in hours], fontsize=16)  # 将刻度标签设为小时
# 
# # 添加自定义图例并放在右侧
# gray_line = mlines.Line2D([], [], color='gray', label='Original Data', linestyle='--')
# red_line = mlines.Line2D([], [], color='#B95756', label='Cluster Center')
# fig.legend(handles=[gray_line, red_line], frameon=False, loc="upper center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.05))
# 
# # 优化布局
# plt.tight_layout()
# plt.show()

#----End Of Cell----


#----End Of Cell----


#----End Of Cell----

