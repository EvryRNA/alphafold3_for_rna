import os
from typing import Dict, List
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

from src.enum import (
    SUB_METRICS,
    ALL_MODELS,
    OLD_TO_NEW,
    COLORS_MAPPING,
    DESC_METRICS,
    ORDER_MODELS,
)
from src.utils.utils import update_bar_plot


class BarHelper:
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

    def read_df(self, in_paths: Dict):
        """
        Read the dataset results
        """
        df = {"Metric": [], "Dataset": [], "Metric (value)": [], "Model": []}
        for dataset, d_path in in_paths.items():
            c_scores = self.get_mean_metrics(d_path)
            for model, values in c_scores.items():
                metric = list(values.values())
                n = len(metric)
                metric_name = list(values.keys())
                df["Metric"].extend(metric_name)
                df["Dataset"].extend([dataset] * n)
                df["Metric (value)"].extend(metric)
                df["Model"].extend([model] * n)
        df = pd.DataFrame(df)
        df = self.normalize_metrics(df)
        df = df.replace(OLD_TO_NEW)
        return df

    def normalize_metrics(self, df, desc_metrics: List = DESC_METRICS):
        metrics, datasets = df["Metric"].unique(), df["Dataset"].unique()
        for metric in metrics:
            mask = df["Metric"] == metric
            metric_values = df.loc[mask, "Metric (value)"].values.reshape(-1, 1)
            if metric_values.shape[0] == 0:
                continue
            non_nan_metrics = metric_values[~np.isnan(metric_values)].reshape(-1, 1)
            if len(non_nan_metrics) == 0:
                continue
            scaler = MinMaxScaler().fit(X=non_nan_metrics)
            norm_metric = scaler.transform(X=metric_values).reshape(-1).tolist()
            if metric in desc_metrics:
                norm_metric = [x if np.isnan(x) else 1 - x for x in norm_metric]
            df.loc[mask, "Metric (value)"] = norm_metric
        return df

    def viz(self):
        datasets = self.df["Dataset"].unique()
        for i, dataset in enumerate(datasets):
            self.viz_dataset(dataset)

    def viz_dataset(self, dataset: str):
        """Plot the polar distribution for a dataset."""
        width, height = 1000, 600
        if dataset != "All":
            df = self.df[self.df["Dataset"] == dataset]
        else:
            df = (
                self.df[["Metric", "Model", "Metric (value)"]]
                .groupby(["Model", "Metric"])
                .mean()
                .reset_index()
            )
        fig = px.bar(
            df,
            y="Model",
            x="Metric (value)",
            color="Metric",
            color_discrete_map=COLORS_MAPPING,
            orientation="h",
            category_orders={
                "Metric": list(COLORS_MAPPING.keys()),
                "Model": ORDER_MODELS,
            },
            labels={"Metric (value)": "Normalized metrics"},
            range_x=[0, 9],
        )
        fig = update_bar_plot(fig)
        save_path = os.path.join("data", "plots", "bar", dataset + ".png")
        fig.write_image(save_path, scale=2, width=width, height=height)

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

    @staticmethod
    def get_viz():
        benchmarks = ["CASP_RNA", "RNA_PUZZLES", "RNASOLO", "RNA3DB", "RNA3DB_LONG"]
        in_paths = {
            name: os.path.join("data", "output", name) for name in benchmarks
        }
        viz_polar = BarHelper(in_paths)
        viz_polar.viz()
