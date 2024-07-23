import os
from typing import Dict
import numpy as np

import pandas as pd
import plotly.express as px

from src.enum import COLOR_DATASET, COLOR_INTERACTIONS, OLD_TO_NEW
from src.utils.utils import update_fig_box_plot


class NWCHelper:
    def __init__(
        self,
        df: pd.DataFrame,
        save_dir,
        interactions_csv_path: str = os.path.join(
            "data", "info", "interactions.csv"
        ),
    ):
        self.df = df
        self.save_dir = save_dir
        self.df_inter = pd.read_csv(interactions_csv_path)

    def get_viz(self):
        self.show_missing_interactions()
        self.show_nwc_dataset()
        self.plot_wc_vs_nwc(category_size="RMSD")
        self.plot_wc_vs_nwc(x="INF-WC", y="INF-STACK", category_size="RMSD")

    def show_missing_interactions(self):
        datasets = self.df_inter["Dataset"].unique()
        interactions = self.df_inter["Interaction type"].unique()
        for dataset in datasets:
            to_print = f"{dataset}: "
            for interaction in interactions:
                df = self.df_inter[
                    (self.df_inter["Dataset"] == dataset)
                    & (self.df_inter["Interaction type"] == interaction)
                ]
                df_non_null = df[df["Number of interactions"] > 0]
                count_null = df_non_null.shape[0] / df.shape[0]
                to_print += f"{interaction}: {count_null:.2f}, "
            print(to_print)

    def _get_nb_nwc(self, count_dict: Dict, name: str):
        out = {}
        out = {**out, **{f"Mean {name}": []}}
        for dataset, values in count_dict.items():
            mean_value, std_value = np.mean(values), np.std(values)
            out[f"Mean {name}"].append(f"{mean_value:.1f}")
        return out

    def show_nwc_dataset(self):
        df = self.df_inter.copy()
        df = df.replace(OLD_TO_NEW)
        df["Number of interactions"] = [1 + x for x in df["Number of interactions"]]
        fig = px.box(
            df,
            x="Dataset",
            y="Number of interactions",
            color="Interaction type",
            log_y=True,
            color_discrete_map=COLOR_INTERACTIONS,
            category_orders={"Dataset": list(COLOR_DATASET.keys())},
        )
        fig = update_fig_box_plot(fig, legend_coordinates=(0.24, -0.1))
        fig.update_layout(
            font=dict(
                size=16,
            )
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
            ),
        )
        fig.update_xaxes(
            showline=True,
            linewidth=1.2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickformat=".0f",
            tickfont_size=20,
        )
        fig.update_layout(xaxis_title=None)
        fig.update_yaxes(tickfont_size=20)
        fig.update_layout(legend_title_text="Interaction type:")
        save_path = os.path.join(
            self.save_dir, "nwc", "nwc_stack_wc_box_distribution.png"
        )
        fig.write_image(save_path, width=800, height=500, scale=4)

    def plot_wc_vs_nwc(
        self, x: str = "INF-WC", y="INF-NWC", category_size: str = "TM-score"
    ):
        df = self.df[[x, y, category_size, "Dataset"]]
        df = df.dropna()
        df["Name"] = [name.lower() for name in df.index]
        df.replace(OLD_TO_NEW, inplace=True)
        fig = px.scatter(
            df,
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
            self.save_dir, "nwc", f"{x}_{y}_by_{category_size}.png"
        )
        fig.write_image(save_path, width=1000, height=1000, scale=2)
