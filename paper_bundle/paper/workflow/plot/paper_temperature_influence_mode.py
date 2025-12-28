from __future__ import annotations

import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde, norm, entropy

THIS_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))

from paper.workflow.lib.paper_paths import BUNDLE_ROOT, FIGURE_DIR, PAPER_ROOT  # noqa: E402


def _ensure_optional_deps() -> None:
    try:
        import plotnine  # noqa: F401
        import mizani  # noqa: F401
        import statsmodels  # noqa: F401
        import skmisc  # noqa: F401
    except ImportError:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "plotnine==0.15.2",
                "mizani==0.14.3",
                "statsmodels==0.14.6",
            ],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "scikit-misc==0.5.2"], check=True)


def compute_density(df: pd.DataFrame, section_label: str, rat: float = 1.0, scale: float = 1.0) -> pd.DataFrame:
    if len(df["res"]) > 1:
        density = gaussian_kde(df["res"])
        xs = np.linspace(df["res"].min(), df["res"].max(), 1000)
        ys = density(xs)
        rescaled_x = max(df["x"]) - ys * rat
        mean_y = xs
        return pd.DataFrame({"x": rescaled_x, "y": mean_y / scale, "type": "Empirical", "group": f"{section_label}_empirical"})
    return pd.DataFrame()


def add_normal_lines(df: pd.DataFrame, section_label: str, rat: float = 1.0, scale: float = 1.0) -> pd.DataFrame:
    if len(df["res"]) > 1:
        xs = np.linspace(df["res"].min(), df["res"].max(), 1000)
        ys = norm.pdf(xs, df["res"].mean(), df["res"].std())
        rescaled_x = max(df["x"]) - ys * rat
        mean_y = xs
        return pd.DataFrame({"x": rescaled_x, "y": mean_y / scale, "type": "Normal", "group": f"{section_label}_normal"})
    return pd.DataFrame()


def remove_outliers(df: pd.DataFrame, columns: list[str], iqr_factor: float = 1.5) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"The column '{column}' does not exist in the DataFrame.")

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df


def compute_kl_divergence(series: pd.Series) -> float:
    if len(series) > 1:
        empirical_kde = gaussian_kde(series)
        xs = np.linspace(series.min(), series.max(), 1000)
        empirical_pdf = empirical_kde(xs)
        empirical_pdf /= np.sum(empirical_pdf)

        mean = series.mean()
        std = series.std()
        normal_pdf = norm.pdf(xs, mean, std)
        normal_pdf /= np.sum(normal_pdf)

        return float(entropy(empirical_pdf, normal_pdf))
    return float("nan")


def main() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MATPLOTLIBRC"] = str(PAPER_ROOT / "matplotlibrc")
    os.environ.update(env)

    _ensure_optional_deps()
    from plotnine import (  # noqa: E402
        aes,
        element_blank,
        element_text,
        geom_path,
        geom_point,
        geom_smooth,
        geom_text,
        geom_vline,
        ggplot,
        labs,
        scale_x_continuous,
        scale_y_continuous,
        theme,
        theme_bw,
    )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue LT Pro", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Helvetica Neue LT Pro"
    plt.rcParams["mathtext.it"] = "Helvetica Neue LT Pro:italic"
    plt.rcParams["mathtext.bf"] = "Helvetica Neue LT Pro:bold"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.unicode_minus"] = False

    datajh = pd.read_csv(BUNDLE_ROOT / "groupedjhS.csv")
    datacm = pd.read_csv(BUNDLE_ROOT / "groupedcmS.csv")

    datajh["place"] = "JH"
    datacm["place"] = "CM"
    data = pd.concat([datajh, datacm], axis=0)

    data.columns = [
        "Time",
        "TVOC",
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
        "0.25um",
        "0.28um",
        "0.30um",
        "0.35um",
        "0.40um",
        "0.45um",
        "0.50um",
        "0.58um",
        "0.65um",
        "0.70um",
        "0.80um",
        "1.00um",
        "1.30um",
        "1.60um",
        "2.00um",
        "2.50um",
        "3.00um",
        "3.50um",
        "4.00um",
        "5.00um",
        "6.50um",
        "7.50um",
        "8.50um",
        "10.00um",
        "12.50um",
        "15.00um",
        "17.50um",
        "20.00um",
        "25.00um",
        "30.00um",
        "32.00um",
        "PM10",
        "PM2.5",
        "PM1",
        "SO2",
        "NOx",
        "NO",
        "NO2",
        "CO",
        "O3",
        "NO2.1",
        "NegativeOxygenIons",
        "Radiation",
        "Temperature",
        "Humidity",
        "WindSpeed",
        "Hour_Min_Sec",
        "Hour_Min",
        "Hour",
        "Month",
        "Day",
        "Datetime",
        "seconds",
        "place",
    ]

    variables = ["Isoprene", "1,1-Dichloroethylene", "n-Tridecane"]
    places = ["JH", "CM"]
    breaksp = [7, 7]
    ratp = [0.7, 0.7, 0.3]
    colorcenter = ["#96C37D", "#C497B2", "#F3D266"]

    plots = []
    for i, place in enumerate(places):
        for j, var in enumerate(variables):
            datp = data[data.place == place].groupby("Hour_Min").mean(numeric_only=True).reset_index()
            xy = [var, "Temperature"]

            x = datp[xy[1]].copy()
            y = datp[xy[0]].copy()
            dat = pd.DataFrame({"x": x, "y": y})
            dat = remove_outliers(dat, ["x", "y"])

            breaks = np.linspace(dat["x"].min(), dat["x"].max(), breaksp[i])
            dat["section"] = pd.cut(dat["x"], breaks)

            dat["res"] = dat["y"]
            grouped_sections = dat.groupby("section", observed=False)
            densities = pd.concat([compute_density(group, str(label), ratp[j]) for label, group in grouped_sections])
            normal_lines = pd.concat([add_normal_lines(group, str(label), ratp[j]) for label, group in grouped_sections])
            densities = pd.concat([densities, normal_lines])

            kl_divs = dat.groupby("section", observed=False)["y"].apply(compute_kl_divergence).reset_index()
            kl_divs.columns = ["section", "kl_divergence"]
            kl_divs["kl_label"] = kl_divs["kl_divergence"].apply(lambda v: f"KLD={v:.2f}" if not np.isnan(v) else "")
            kl_divs["section_mid"] = kl_divs["section"].apply(lambda v: v.mid if pd.notnull(v) else np.nan).astype(float)

            plot = (
                ggplot(dat, aes("x", "y"))
                + geom_point(color=colorcenter[j], alpha=0.3)
                + geom_smooth(method="loess", se=False, color=colorcenter[j], size=1.5)
                + geom_path(
                    data=densities,
                    mapping=aes("x", "y", color="type", group="group"),
                    size=1,
                    linetype="dashdot",
                )
                + geom_vline(xintercept=breaks, linetype="dashed")
            )

            if var == variables[2] and i == 0:
                plot = plot + labs(x="T ($^\\circ$C)", y=f"{var} ($\\mu$g/m$^3$)")
            elif var == variables[2] and i == 1:
                plot = plot + labs(x="T ($^\\circ$C)", y="")
            elif var != variables[2] and i == 0:
                plot = plot + labs(x="", y=f"{var} ($\\mu$g/m$^3$)")
            else:
                plot = plot + labs(x="", y="")

            plot = (
                plot
                + scale_x_continuous(
                    breaks=np.linspace(dat["x"].min(), dat["x"].max(), 6),
                    labels=lambda xs: [f"{val:.2f}" for val in xs],
                )
                + scale_y_continuous(
                    breaks=np.linspace(dat["y"].min(), dat["y"].max(), 5),
                    labels=lambda ys: [f"{val:.2f}" for val in ys],
                )
                + theme_bw()
                + theme(
                    figure_size=(4.6, 3.0),
                    legend_position=(0.3, 0.9),
                    legend_title=element_blank(),
                    legend_key=element_blank(),
                    legend_background=element_blank(),
                    text=element_text(family="Helvetica Neue LT Pro"),
                    axis_text_x=element_text(
                        family="Helvetica Neue LT Pro",
                        size=11,
                        weight="bold",
                        angle=45,
                        ha="right",
                    ),
                    axis_text_y=element_text(family="Helvetica Neue LT Pro", size=11, weight="bold"),
                    axis_title_x=element_text(family="Helvetica Neue LT Pro", size=13, weight="bold"),
                    axis_title_y=element_text(family="Helvetica Neue LT Pro", size=13, weight="bold"),
                    legend_text=element_text(family="Helvetica Neue LT Pro", size=12, weight="bold"),
                )
            )

            plot = plot + geom_text(
                data=kl_divs,
                mapping=aes(
                    x="section_mid",
                    y=float(dat["y"].min() + 0.3 * (dat["y"].max() - dat["y"].min())),
                    label="kl_label",
                ),
                ha="center",
                va="top",
                size=12,
                family="Helvetica Neue LT Pro",
                fontweight="bold",
                angle=45,
            )

            plots.append(plot)

    if len(plots) != 6:
        raise RuntimeError(f"Expected 6 panels but got {len(plots)}")

    def _render_panel(p, dpi: int = 500) -> Image.Image:
        fig = p.draw()
        fig.set_size_inches(4.6, 3.0)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    panels = [_render_panel(p) for p in plots]
    col_width = max(img.size[0] for img in panels)
    row_height = max(img.size[1] for img in panels)
    combined = Image.new("RGB", (col_width * 2, row_height * 3), (255, 255, 255))

    def _paste(panel: Image.Image, col: int, row: int) -> None:
        x0 = col * col_width + (col_width - panel.size[0]) // 2
        y0 = row * row_height + (row_height - panel.size[1]) // 2
        combined.paste(panel, (x0, y0))

    # Panel order matches the legacy script:
    # Left column: JH (Isoprene, 1,1-Dichloroethylene, n-Tridecane)
    # Right column: CM (Isoprene, 1,1-Dichloroethylene, n-Tridecane)
    for row in range(3):
        _paste(panels[row], 0, row)
        _paste(panels[row + 3], 1, row)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "VOC_temperature_influence_mode.png"
    combined.save(str(output_path), dpi=(500, 500))
    print(f"Saved temperature influence mode figure: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
