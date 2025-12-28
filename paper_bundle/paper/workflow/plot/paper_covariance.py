from __future__ import annotations

import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from scipy.interpolate import interp1d

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT

MODEL_PATH = BUNDLE_ROOT / "kmeans.joblib"
DATA_JH = BUNDLE_ROOT / "groupedjhS.csv"
DATA_CM = BUNDLE_ROOT / "groupedcmS.csv"
SCRIPT_PATH = BUNDLE_ROOT / "scripts" / "共变关系.py"
TARGET_NAME = FIGURE_DIR / "VOC_covariance_clusters.png"


def transform_to_same_length(x: np.ndarray, max_length: int) -> np.ndarray:
    n = x.shape[0]
    ucr_x = np.zeros((n, max_length, 1), dtype=np.float64)
    mts = x
    curr_length = n
    idx = np.array(range(curr_length))
    idx_new = np.linspace(0, idx.max(), max_length)
    f = interp1d(idx, mts, kind="cubic")
    new_ts = f(idx_new)
    return new_ts


def ensure_kmeans_model() -> None:
    if MODEL_PATH.exists():
        return
    datajh = pd.read_csv(DATA_JH)
    datacm = pd.read_csv(DATA_CM)
    data = pd.concat([datajh, datacm], axis=0)
    vocs = [
        "Methyl Mercaptan",
        "1,3-Butadiene",
        "Butene",
        "Acetone/Butane",
        "n-Propanol",
        "Dimethyl Sulfide/Ethyl Mercaptan",
        "Chloroethane",
        "Isoprene",
        "Pentene",
        "Pentane/Isopentane",
        "Dimethylformamide",
        "Ethyl Formate",
        "Carbon Disulfide/Propyl Mercaptan",
        "Benzene",
        "Cyclohexene",
        "Hexene/Methylcyclopentane",
        "n-Hexane/Dimethylbutane",
        "Ethyl Sulfide/Butyl Mercaptan",
        "Toluene",
        "Aniline",
        "Dimethyl Disulfide",
        "1,1-Dichloroethylene",
        "Methylcyclohexane",
        "n-Heptane",
        "Triethylamine",
        "n-Propyl Acetate",
        "Diethylene Triamine",
        "Styrene",
        "Xylene/Ethylbenzene",
        "1,3-Dichloropropene",
        "n-Octane",
        "n-Butyl Acetate",
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
    ]
    voc_data = data.iloc[:, 1 : 1 + len(vocs)].copy()
    voc_data.columns = vocs
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataclu = []
    for col in vocs:
        series = transform_to_same_length(voc_data[col].values, 250)
        normalized = scaler.fit_transform(series.reshape(-1, 1))
        dataclu.append(normalized)
    kmeans = TimeSeriesKMeans(
        n_clusters=3,
        n_init=50,
        max_iter=1000,
        metric="dtw",
        n_jobs=-1,
        random_state=132,
    )
    kmeans.fit(dataclu)
    joblib.dump(kmeans, MODEL_PATH)


def main() -> None:
    ensure_kmeans_model()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    subprocess.run([sys.executable, str(SCRIPT_PATH)], cwd=BUNDLE_ROOT, check=True, env=env)
    if not TARGET_NAME.exists():
        raise FileNotFoundError(f"Expected covariance figure missing: {TARGET_NAME}")
    print("Covariance figure generated as PNG.")


if __name__ == "__main__":
    sys.exit(main())
