import os
import pandas as pd
from typing import List
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from src.enum import COLORS_MAPPING, OLD_TO_NEW, ASC_METRICS
from src.utils.utils import update_fig_box_plot


class BoxHelper:
    def __init__(self, df: pd.DataFrame, save_dir: str, family_df_path: str):
        self.df = df
        self.save_dir = save_dir
        self.family_df = pd.read_csv(family_df_path)
        self.family_df.index = self.family_df["RNA"].map(lambda x: x.lower())

    def run_supplementary(self):
        self.get_viz(
            metrics=["εRMSD", "DI", "P-VALUE", "CAD", "GDT-TS"],
            name="supp_box_datasets",
        )
        self.get_viz(
            metrics=["INF-STACK", "INF-WC", "INF-NWC", "INF-NWC", "INF-ALL"],
            name="inf_datasets",
        )

    def get_viz(
        self,
        metrics=["RMSD", "MCQ", "TM-score", "lDDT", "INF-ALL"],
        name="box_datasets",
    ):
        df = self.convert_to_box_dataset(metrics)
        df = df.replace(OLD_TO_NEW)
        df = df[df["Dataset"] == "RNA3DB_Long"]
        df.loc[~df["Family"].isin(["RIBOSOME"]), "Family"] = "Other"
        fig = make_subplots(rows=1, cols=len(metrics), horizontal_spacing=0.06)
        for i, metric in enumerate(metrics):
            c_df = df[df["Metric_name"] == metric]
            fig.add_trace(
                go.Box(
                    y=c_df["Metric"],
                    x=c_df["Family"],
                    name=metric,
                    fillcolor=COLORS_MAPPING.get(metric),
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker_size=2,
                    line_width=2,
                    marker=dict(size=2, color="rgb(0, 0, 0)"),
                    notched=False,
                ),
                row=1,
                col=i + 1,
            )
            if metric in ASC_METRICS:
                fig.update_yaxes(range=[0, 1], row=1, col=i + 1)
            fig.update_yaxes(title_text=metric, row=1, col=i + 1, title_standoff=0)
        fig.update_yaxes(row=1, col=1, title_standoff=10)
        fig.update_traces(boxpoints="all", jitter=0.3)
        fig = update_fig_box_plot(
            fig, marker_size=None, legend_coordinates=(0.65, -0.2)
        )
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(title_text="RNA family", row=1, col=3)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=45)
        fig.update_traces(
            marker=dict(line=dict(width=2, color="black")),
            selector=dict(mode="markers"),
        )
        save_path = os.path.join(self.save_dir, "box", f"{name}.png")
        fig.write_image(save_path, width=1200, height=350, scale=3)

    def convert_to_box_dataset(self, metrics: List):
        out = {"Metric": [], "Metric_name": [], "Dataset": [], "Family": []}
        c_df = self.df.copy()
        c_df["εRMSD"] = c_df["BARNABA-eRMSD"]
        for row, data in c_df.iterrows():
            for col in metrics:
                out["Metric"].append(data[col])
                out["Metric_name"].append(col)
                out["Dataset"].append(data["Dataset"])
                out["Family"].append(self.family_df.loc[row]["Family"])
        df = pd.DataFrame(out)
        return df
