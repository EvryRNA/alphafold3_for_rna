import os
from typing import Dict
import numpy as np

import pandas as pd
import plotly.express as px

from src.utils.enum import COLOR_DATASET, OLD_TO_NEW
from src.utils.utils import update_fig_box_plot


class NWCHelper:
    def __init__(
        self,
        df: pd.DataFrame,
        save_dir,
        interactions_csv_path: str = os.path.join("data", "info", "interactions.csv"),
    ):
        self.df = df
        self.save_dir = save_dir
        self.df_inter = pd.read_csv(interactions_csv_path)

    def get_viz(self):
        self.show_nwc_dataset_df()
        self.plot_wc_vs_nwc(category_size="RMSD")
        self.plot_wc_vs_nwc(x="INF-WC", y="INF-STACK", category_size="RMSD")

    def _get_nb_nwc(self, count_dict: Dict, name: str):
        out = {}
        out = {**out, **{f"Mean {name}": []}}
        for dataset, values in count_dict.items():
            mean_value, std_value = np.mean(values), np.std(values)
            out[f"Mean {name}"].append(f"{mean_value:.1f}")
        return out

    def show_nwc_dataset_df(self):
        """
        Show the percentage of interactions for each dataset
        """
        df = self.df_inter.copy()
        df = df.replace(OLD_TO_NEW)
        info = pd.read_csv(os.path.join("data", "info", "info_datasets.csv"))
        df = df.merge(info[["RNA", "Length"]], on="RNA")
        df["Number of interactions"] = df["Number of interactions"] / df["Length"]
        df = (
            df[["Dataset", "Interaction type", "Number of interactions"]]
            .groupby(["Dataset", "Interaction type"])
            .mean()
        )
        df_reset = df.reset_index()
        df_pivot = df_reset.pivot(
            index="Dataset", columns="Interaction type", values="Number of interactions"
        )
        df_pivot = df_pivot.round(2)
        df_latex = df_pivot.to_latex(float_format="%.2f")
        save_dir = os.path.join("data", "plots", "tables", "nwc")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "interactions.tex")
        with open(save_path, "w") as f:
            f.write(df_latex)

    def plot_wc_vs_nwc(
        self, x: str = "INF-WC", y="INF-NWC", category_size: str = "TM-score"
    ):
        df = self.df[self.df["Method"].isin(["alphafold3", "alphafold3c"])]
        df = df[df["Metric_name"].isin([x, y, category_size])]
        df.loc[df["Method"] == "alphafold3c", "Dataset"] = "RNA3DB_0 (Context)"
        df = df.reset_index(drop=True)
        pivot_df = df.pivot_table(
            index=["Name", "Dataset"],
            columns="Metric_name",
            values="Metric",
        ).reset_index()
        pivot_df = pivot_df.dropna()
        pivot_df.replace(OLD_TO_NEW, inplace=True)
        fig = px.scatter(
            pivot_df,
            x=x,
            y=y,
            color="Dataset",
            color_discrete_map=COLOR_DATASET,
            hover_data=["Name"],
            size=category_size,
            range_y=[-0.05, 1.05],
        )
        fig.update_traces(
            marker=dict(line=dict(width=2, color="black")),
            selector=dict(mode="markers"),
        )
        fig = update_fig_box_plot(fig, marker_size=None)
        fig.update_layout(
            legend=dict(
                x=0.1,
                y=-0.12,
                orientation="h",
            ),
        )
        fig.update_layout(
            legend=dict(
                borderwidth=1.5,
            ),
        )
        fig.update_layout(font=dict(family="Computer Modern", size=24))
        save_path = os.path.join(
            self.save_dir, "figures", "nwc", f"{x}_{y}_by_{category_size}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path, width=1000, height=1000, scale=2)
