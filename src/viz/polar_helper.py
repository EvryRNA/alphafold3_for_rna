import os
import numpy as np
from typing import List, Dict

import pandas as pd
import plotly.express as px

from src.enum import SUB_METRICS, ALL_MODELS, OLD_TO_NEW


class PolarHelper:
    def __init__(self, in_paths: Dict):
        self.df = self.read_df(in_paths)

    def get_mean_metrics(
        self, in_path: str, metrics: List = SUB_METRICS, models: List = ALL_MODELS
    ):
        """
        Return the mean per metric from a directory with .csv files
        :param in_path:
        :return:
        """
        files = [x for x in os.listdir(in_path) if x.endswith(".csv")]
        scores = {model: {metric: [] for metric in metrics} for model in models}
        if "casp" in in_path:
            files = [
                f"{x}.csv"
                for x in [
                    "R1107",
                    "R1108",
                    "R1116",
                    "R1117",
                    "R1149",
                    "R1156",
                    "R1189",
                    "R1190",
                ]
            ]
            models = models.copy()
            models.remove("mcsym")
            metrics = metrics.copy()
        for n_file in files:
            in_df = os.path.join(in_path, n_file)
            df = pd.read_csv(in_df, index_col=[0])
            for model in models:
                out = self.get_metrics_from_model(df, model, metrics)
                for c_score, metric in zip(out, metrics):
                    scores[model][metric].append(c_score)
        scores = self.get_mean_scores(scores)
        return scores

    def get_mean_scores(self, scores: Dict):
        for model, values in scores.items():
            for metric_name, metric in values.items():
                scores[model][metric_name] = np.nanmean(metric)
        return scores

    def get_metrics_from_model(self, df: pd.DataFrame, model: str, metrics: List):
        names = [x for x in df.index if model in x]
        df.rename(columns=OLD_TO_NEW, inplace=True)
        df = df.loc[names].mean(axis=0)
        output = []
        for metric in metrics:
            if metric in df:
                output.append(df[metric])
            else:
                output.append(np.nan)
        return output

    def read_df(self, in_paths: Dict):
        """
        Read the dataset results
        """
        df = {"Metric": [], "Dataset": [], "Metric (value)": [], "Model": []}
        for dataset, d_path in in_paths.items():
            c_scores = self.get_mean_metrics(
                d_path, metrics=["INF-WC", "INF-NWC", "INF-STACK"]
            )
            for model, values in c_scores.items():
                metric = list(values.values())
                n = len(metric)
                metric_name = list(values.keys())
                df["Metric"].extend(metric_name)
                df["Dataset"].extend([dataset] * n)
                df["Metric (value)"].extend(metric)
                df["Model"].extend([model] * n)
        df = pd.DataFrame(df)
        df = df.replace(OLD_TO_NEW)
        return df

    def save_table(self):
        df = self.df.pivot(
            index="Model", columns=["Dataset", "Metric"], values="Metric (value)"
        )
        save_path = os.path.join(
            "data", "plots", "polar", "inf_interactions.tex"
        )
        df_latex = df.to_latex(float_format="%.2f")
        with open(save_path, "w") as f:
            f.write(df_latex)

    def viz(self):
        self.save_table()
        df = (
            self.df[["Metric", "Model", "Metric (value)"]]
            .groupby(["Model", "Metric"])
            .mean()
            .reset_index()
        )
        metric_order = ["INF-WC", "INF-NWC", "INF-STACK"]
        df["Metric"] = df["Metric"].astype(
            pd.CategoricalDtype(categories=metric_order, ordered=True)
        )
        df = df.sort_values(by="Metric").reset_index(drop=True)
        colors = ["#83b8d6", "#621038", "#ef8927"]
        # Remove Challenge-best
        df = df[df["Model"] != "Challenge-best"]
        fig = px.bar_polar(
            df,
            r="Metric (value)",
            theta="Model",
            color="Metric",
            template="plotly_white",
            color_discrete_sequence=colors,
            range_r=[0, 3],
        )
        fig = self._clean_polar_viz(fig)
        fig.update_layout(legend_title_text="Metric:")
        save_path = os.path.join(
            "data", "plots", "polar", "all_inf_interactions.png"
        )
        fig.write_image(save_path, scale=4, width=1000, height=600)

    def _clean_polar_viz(self, fig):
        new_polars = {
            "polar": dict(
                radialaxis=dict(
                    showline=False,
                    showgrid=True,
                    linewidth=1,
                    linecolor="black",
                    gridcolor="black",
                    gridwidth=1,
                    showticklabels=True,
                    dtick=1,
                ),
                angularaxis=dict(
                    linewidth=1,
                    visible=True,
                    linecolor="black",
                    showline=True,
                    gridcolor="black",
                ),
                radialaxis_tickfont_size=14,
                bgcolor="white",
            )
        }
        fig.update_layout(
            legend=dict(
                orientation="h",
                bgcolor="#f3f3f3",
                bordercolor="Black",
                borderwidth=1,
                font=dict(size=20),
                x=0.25,
                y=-0.1,
            ),
        )
        fig.update_layout(margin=dict(l=0, r=0, b=50, t=50))
        fig.update_layout(font_size=22)
        fig.update_layout(
            **new_polars,
            showlegend=True,
        )
        return fig

    @staticmethod
    def get_viz():
        benchmarks = ["CASP_RNA", "RNA_PUZZLES", "RNASOLO"]
        in_paths = {
            name: os.path.join("data", "output", name) for name in benchmarks
        }
        viz_polar = PolarHelper(in_paths)
        viz_polar.viz()
