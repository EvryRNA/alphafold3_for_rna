from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src.utils.enum import DESC_METRICS, SUB_METRICS
from scipy.stats import wilcoxon


class ComputePValue:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def normalize_metrics(self, df: pd.DataFrame, desc_metrics: List = DESC_METRICS):
        """
        Normalize the metrics by the min-max over all the datasets
        :param df: dataframe with the metric values
        :param desc_metrics: descending metrics to reverse
        :return: new normalized dataframe
        """
        metrics, datasets = df["Metric_name"].unique(), df["Dataset"].unique()
        for metric in metrics:
            mask = (
                (df["Metric_name"] == metric)
                & df["Metric"].notna()
                & np.isfinite(df["Metric"])
            )
            metric_values = df.loc[mask, "Metric"].values.reshape(-1, 1)
            if metric_values.shape[0] == 0:
                continue
            scaler = MinMaxScaler().fit(X=metric_values)
            norm_metric = scaler.transform(X=metric_values).reshape(-1).tolist()
            if metric in desc_metrics:
                norm_metric = [x if np.isnan(x) else 1 - x for x in norm_metric]
            df.loc[mask, "Metric (value)"] = norm_metric
        return df

    def compute_wilcoxon(
        self, df: pd.DataFrame, dataset: List, methods=["best", "alphafold3"]
    ):
        """
        Compute wilcox test for the given dataset and methods
        """
        df = df[df["Dataset"].isin(dataset)]
        df = df[df["Method"].isin(methods)].reset_index(drop=True)
        df_group = (
            df[["Name", "Method", "Metric (value)"]]
            .groupby(["Name", "Method"])
            .sum()
            .reset_index()
        )
        names = df_group.loc[df_group["Method"] == methods[1], "Name"]
        df_group = df_group[df_group["Name"].isin(names)]
        x = df_group[df_group["Method"] == methods[0]]["Metric (value)"].values
        y = df_group[df_group["Method"] == methods[1]]["Metric (value)"].values
        stat, p_value = wilcoxon(x, y)
        print(f"{dataset} Wilcoxon test: {p_value} for {methods[0]} > {methods[1]}")
        if len(methods) > 2:
            z = df_group[df_group["Method"] == methods[2]]["Metric (value)"].values
            stat, p_value = wilcoxon(x, z)
            print(f"{dataset} Wilcoxon test: {p_value} for {methods[0]} > {methods[2]}")

    def run(self):
        """
        Show the statistical test for the different datasets
        """
        df = self.normalize_metrics(self.df)
        df = df[df["Metric_name"].isin(SUB_METRICS)]
        datasets = [
            "RNA_PUZZLES",
            "CASP_RNA",
            "RNASOLO",
            "RNA3DB",
            "RNA3DB",
            "RNA3DB",
            "CASP_RNA",
        ]
        methods = [
            None,
            ["alphafold3", "best", "trrosettarna"],
            ["alphafold3", "trrosettarna"],
            ["alphafold3", "rhofold"],
            ["alphafold3", "alphafold3c"],
            ["rhofold", "alphafold3c"],
            ["alphafold3", "isrna"],
        ]
        for dataset, method in zip(datasets, methods):
            if method is None:
                self.compute_wilcoxon(df, [dataset])
            else:
                self.compute_wilcoxon(df, [dataset], method)
