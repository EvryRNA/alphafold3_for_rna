from typing import List

import pandas as pd

from src.utils.enum import SUB_METRICS, OLD_TO_NEW, ORDER_MODELS, DESC_METRICS
import os


class TableHelper:
    def __init__(self, df: pd.DataFrame, save_dir: str):
        self.df = df
        self.save_dir = save_dir

    def run(self):
        df = self.df.replace(OLD_TO_NEW)
        df = df[df["Metric_name"].isin(SUB_METRICS)]
        df = df[df["Method"].isin(ORDER_MODELS)]
        result = (
            df.groupby(["Dataset", "Method", "Metric_name"])["Metric"]
            .agg(["mean", "std"])
            .reset_index()
        )
        result.rename(
            columns={"mean": "Mean", "std": "Standard Deviation"}, inplace=True
        )
        datasets = df["Dataset"].unique()
        for dataset in datasets:
            self.show_table(result[result["Dataset"] == dataset], dataset)

    def show_context_vs_no_context(self):
        df = self.df.replace(OLD_TO_NEW)
        df = df[df["Metric_name"].isin(SUB_METRICS)]
        df = df[df["Method"].isin(["AlphaFold 3 (DL)", "AlphaFold 3 (Context)"])]
        df = df[df["Dataset"] == "RNA3DB_0"]
        df.index.name = "Names"
        names = df[(df["Method"] == "AlphaFold 3 (Context)")]["Name"].unique()
        df = df[df["Name"].isin(names)]
        df_group = (
            df[["Name", "Method", "Metric_name", "Metric"]]
            .groupby(["Name", "Method", "Metric_name"])
            .mean()
        )
        pivot_df = df_group.pivot_table(
            index="Name", columns=["Metric_name", "Method"], values="Metric"
        )
        pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
        pivot_df.reset_index(inplace=True)
        n_pivot_df = self.merge_data(pivot_df)
        main_metrics = ["RMSD", "MCQ", "TM-score", "INF-ALL", "lDDT"]
        self.save_data_to_latex(n_pivot_df, main_metrics, "main_metrics")

    def merge_data(self, df):
        context_columns = [col for col in df.columns if "AlphaFold 3 (Context)" in col]
        dl_columns = [col for col in df.columns if "AlphaFold 3 (DL)" in col]
        context_columns.sort()
        dl_columns.sort()
        merged_data = {"Name": df["Name"]}
        for context_col, dl_col in zip(context_columns, dl_columns):
            metric_name = context_col.split("_")[0]
            new_col_name = metric_name
            merged_data[new_col_name] = (
                df[context_col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
                + "/"
                + df[dl_col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
            )
            merged_data[new_col_name] = merged_data[new_col_name].str.strip("/")
        merged_df = pd.DataFrame(merged_data)
        merged_df.index = df["Name"].apply(
            lambda x: x.lower().replace(".csv", "").replace("_", "\_").upper()
        )
        merged_df = merged_df.rename({"εRMSD": "$\epsilon$RMSD"}, axis=1)
        return merged_df

    def save_data_to_latex(self, df: pd.DataFrame, metrics: List, name: str):
        """
        Save into latex format the dataframe for the given metrics
        :param metrics: metrics to consider
        :param name: name of the file to save
        """
        c_metrics = [metric.replace("εRMSD", "$\epsilon$RMSD") for metric in metrics]
        df = df[c_metrics]
        latex_df = df.to_latex(escape=False, column_format="l" + "c" * len(SUB_METRICS))
        save_dir = os.path.join(self.save_dir, "tables", "context_vs_no_context")
        save_path = os.path.join(save_dir, name + ".tex")
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_df)

    def show_table(self, df: pd.DataFrame, dataset: str):
        """
        Get the table with the mean values and standard deviation for each method and metric
        """
        df["Mean/Std"] = (
            df["Mean"].round(2).astype(str)
            + " ± "
            + df["Standard Deviation"].round(2).astype(str)
        )
        pivot_table = df.pivot(index="Method", columns="Metric_name", values="Mean/Std")
        pivot_table.columns.name = None
        order_model = [model for model in ORDER_MODELS if model in pivot_table.index]
        pivot_table = pivot_table.loc[order_model, SUB_METRICS]
        for col in pivot_table.columns:
            if col in DESC_METRICS:
                best_indices = (
                    pivot_table[col].apply(lambda x: float(x.split(" ± ")[0])).idxmin()
                )
            else:
                best_indices = (
                    pivot_table[col].apply(lambda x: float(x.split(" ± ")[0])).idxmax()
                )
            pivot_table.loc[best_indices, col] = (
                r"\textbf{" + pivot_table.loc[best_indices, col] + "}"
            )
        latex_df = pivot_table.to_latex(
            escape=False, column_format="l" + "c" * len(SUB_METRICS)
        )
        save_dir = os.path.join(self.save_dir, "tables", "dataset_metrics")
        save_path = os.path.join(save_dir, dataset + ".tex")
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_df)
