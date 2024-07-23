import os
from typing import List

import pandas as pd
import plotly.express as px

from src.enum import COLOR_DATASET, OLD_TO_NEW, ASC_METRICS
from src.utils.utils import update_fig_box_plot
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ScatterHelper:
    def __init__(self, df: pd.DataFrame, save_dir: str):
        self.df = df
        self.save_dir = save_dir

    def get_viz(self):
        self.run_metrics_vs_seq_len(
            ["TM-score", "lDDT", "INF-ALL", "RMSD", "MCQ"],
            name="main_metrics",
            facet_col_wrap=3,
        )
        self._run_mean_test()

    def _get_mean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ranges = np.arange(0, 800, 25)
        out = pd.DataFrame([])
        for index in range(len(ranges) - 1):
            nt_begin, nt_end = ranges[index], ranges[index + 1]
            c_df = df[
                (df["Sequence length"] >= nt_begin) & (df["Sequence length"] < nt_end)
            ]
            if len(c_df) != 0:
                c_df.drop(["RNA", "Sequence length", "Dataset"], axis=1, inplace=True)
                c_df_min = c_df.groupby(["Method", "Metric_name"]).min().reset_index()
                c_df_max = c_df.groupby(["Method", "Metric_name"]).max().reset_index()
                c_df_min = c_df_min.rename(columns={"Metric": "Metric (min)"})
                c_df_min["Metric (max)"] = c_df_max["Metric"]
                c_df_best = self.get_best_value(c_df_min)
                c_df_best["Sequence length"] = (nt_begin + nt_end) / 2
                out = out._append(c_df_best)
        return out

    def get_best_value(self, df: pd.DataFrame):
        best_value = []
        for row, data in df.iterrows():
            metric = data["Metric_name"]
            if metric in ASC_METRICS:
                best_value.append(data["Metric (max)"])
            else:
                best_value.append(data["Metric (min)"])
        df["Metric"] = best_value
        return df

    def get_window_figure(
        self,
        metrics: List,
        n_rows: int,
        cols: int,
        v_space: float = 0.1,
        h_space: float = 0.08,
    ):
        all_df = self._get_all_dfs(metric=metrics)
        all_df = all_df[all_df["Metric"] < 200]
        all_df = all_df.replace({"BARNABA-eRMSD": "εRMSD"})
        if "BARNABA-eRMSD" in metrics:
            metrics[metrics.index("BARNABA-eRMSD")] = "εRMSD"
        new_df = self._get_mean_data(all_df)
        COLORS = {
            "Challenge-best": "#984ea4",
            "AlphaFold 3": "#dc3911",
            "Template-based": "green",
            "Deep Learning": "#1f77b4",
            "Ab initio": "goldenrod",
        }
        fig = make_subplots(
            rows=n_rows, cols=cols, vertical_spacing=v_space, horizontal_spacing=h_space
        )
        positions = [(i, j) for i in range(1, n_rows + 1) for j in range(1, cols + 1)]
        for position, metric in zip(positions, metrics):
            for method in new_df["Method"].unique():
                c_df = new_df[
                    (new_df["Method"] == method) & (new_df["Metric_name"] == metric)
                ]
                x = c_df["Sequence length"].tolist()
                c_color = COLORS.get(method)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=c_df["Metric"],
                        line_color=c_color,
                        name=method,
                        showlegend=metric == "DI",
                    ),
                    row=position[0],
                    col=position[1],
                )
            fig.update_yaxes(
                title_text=metric,
                title_standoff=0 if position[1] == 2 else 5,
                row=position[0],
                col=position[1],
            )
            if metric in ASC_METRICS:
                fig.update_yaxes(range=[-0.1, 1.1], row=position[0], col=position[1])
            fig.update_xaxes(
                tick0=50, dtick=100, row=position[0], col=position[1], range=[0, 800]
            )
        fig.update_traces(mode="lines+markers")
        fig = update_fig_box_plot(fig)
        fig.update_annotations(font_size=8)
        fig.update_layout(font=dict(size=10))
        fig.update_traces(marker=dict(size=7, symbol="circle"))
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.18,
                x=0.05,
            )
        )
        fig.update_layout(legend=dict(orientation="h"))
        fig.update_layout(legend_title_text="Type of approach:")
        return fig

    def _run_mean_test_inf(self, metrics, save_name):
        fig = self.get_window_figure(metrics, n_rows=1, cols=3, h_space=0.08)
        fig.update_layout(font=dict(size=16))
        for i in range(1, len(metrics) + 1):
            fig["layout"]["xaxis{}".format(i)]["title"] = "Sequence length (nt)"
        save_path = os.path.join(self.save_dir, "scatter", save_name)
        width = 1000
        height = 250
        fig.write_image(save_path, width=width, height=height, scale=4)

    def _run_mean_test_supp(self, metrics, save_name):
        fig = self.get_window_figure(
            metrics, n_rows=3, cols=2, v_space=0.05, h_space=0.07
        )
        for i in range(5, len(metrics) + 1):
            fig["layout"]["xaxis{}".format(i)]["title"] = "Sequence length (nt)"
        fig.update_yaxes(range=[-0.1, 1.1], row=2, col=1)
        fig.update_layout(font=dict(size=16))
        fig.update_layout(showlegend=False)
        save_path = os.path.join(self.save_dir, "scatter", save_name)
        width = 1000
        height = 700
        fig.write_image(save_path, width=width, height=height, scale=4)

    def _run_mean_test(
        self,
        metrics=["RMSD", "MCQ", "TM-score", "lDDT", "INF-ALL"],
        save_name="mean_metrics_vs_seq_len_all_models_scatter.png",
    ):
        n_rows = 3 if len(metrics) > 3 else 1
        # cols = len(metrics)//n_rows
        cols = 2
        fig = self.get_window_figure(metrics, n_rows, cols)
        for i in range(4, 6):
            try:
                fig["layout"]["xaxis{}".format(i)]["title"] = "Sequence length (nt)"
            except KeyError:
                pass
        fig.update_layout(font=dict(size=14))
        # Add margin
        fig.update_layout(margin=dict(l=50, r=0, t=0, b=0))
        # Show legend
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.18,
                x=0.05,
            )
        )
        save_path = os.path.join(self.save_dir, "scatter", save_name)
        width = 1000
        height = 500
        fig.write_image(save_path, width=width, height=height, scale=4)

    def _run_test(self):
        metric = ["DI", "MCQ", "BARNABA-eRMSD", "lDDT", "TM-score", "GDT-TS"]
        all_df = self._get_all_dfs(metric=metric)
        all_df = all_df.replace({"BARNABA-eRMSD": "εRMSD"})
        COLORS = {
            "Challenge-best": "#C738BD",
            "AlphaFold3": "#FF7F3E",
            "Template-based": "#00b938",
            "Deep Learning": "#9cdcff",
            "Ab initio": "#c0a085",
        }
        fig = px.scatter(
            all_df,
            x="Sequence length",
            y="Metric",
            color="Method",
            color_discrete_map=COLORS,
            labels={"Sequence length": "Sequence length (nt)"},
            hover_data=["RNA"],
            facet_col="Metric_name",
            facet_col_wrap=3,
            facet_col_spacing=0.03,
        )
        fig = update_fig_box_plot(fig, legend_coordinates=(0, -0.1))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(showticklabels=True, matches=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.14,
                x=0.18,
            )
        )
        save_path = os.path.join(self.save_dir, "all_metrics_vs_seq_len_all_models.png")
        fig.write_image(save_path, width=1200, height=600, scale=2)

    def _get_all_dfs(self, metric: List, top_n=5):
        datasets = ["CASP_RNA", "RNA_PUZZLES"]
        NAME_TO_APPROACHES = {
            "best": "Challenge-best",
            "alphafold3": "AlphaFold 3",
            "vfold": "Template-based",
            "mcsym": "Template-based",
            "rhofold": "Deep Learning",
            "simrna": "Ab initio",
            "rnajp": "Ab initio",
            "rnacomposer": "Template-based",
            "3drna": "Template-based",
            "isrna": "Ab initio",
            "vfoldpipeline": "Template-based",
            "trrosettarna": "Deep Learning",
        }
        all_df = {
            "Dataset": [],
            "RNA": [],
            "Sequence length": [],
            "Method": [],
            "Metric": [],
            "Metric_name": [],
        }
        for dataset in datasets:
            c_dir = os.path.join("data", "output", dataset)
            rnas = [rna for rna in os.listdir(c_dir) if rna.endswith(".csv")]
            for rna in rnas:
                rna_dir = os.path.join(c_dir, rna)
                df = pd.read_csv(rna_dir, index_col=[0])
                df.index = df.index.map(lambda x: x.split("_")[1])
                seq_len = self.df[self.df["Name"] == rna]["Sequence length"].values[0]
                df["Approach"] = df.index.map(
                    lambda x: NAME_TO_APPROACHES.get(x, "Others")
                )
                df = self._get_top_df(metric, 1, df)
                for row in df.iterrows():
                    name = NAME_TO_APPROACHES.get(row[0], None)
                    if name is not None:
                        all_df["Dataset"].extend([dataset] * len(metric))
                        all_df["RNA"].extend([rna.replace(".csv", "")] * len(metric))
                        all_df["Sequence length"].extend([seq_len] * len(metric))
                        all_df["Metric"].extend(row[1][metric])
                        all_df["Metric_name"].extend(metric)
                        method = name
                        all_df["Method"].extend([method] * len(metric))
        all_df = pd.DataFrame(all_df)
        return all_df

    def _get_top_df(self, metric: str, top_n: int, df: pd.DataFrame):
        df = df.sort_values(metric, ascending=True)
        new_df = pd.DataFrame()
        approaches = df["Approach"].unique()
        for approach in approaches:
            c_df = df[df["Approach"] == approach]
            c_df = c_df.head(top_n)
            new_df = new_df._append(c_df)
        return new_df

    def run_supplementary(self):
        self.run_metrics_vs_seq_len(
            ["BARNABA-eRMSD", "DI", "P-VALUE", "CAD", "GDT-TS"],
            name="supp_vs_len",
            facet_col_wrap=3,
        )
        self.run_metrics_vs_seq_len(
            ["INF-WC", "INF-NWC", "INF-STACK", "INF-ALL", "CLASH"],
            name="inf",
            facet_col_wrap=3,
        )
        self._run_mean_test_supp(
            metrics=["BARNABA-eRMSD", "DI", "P-VALUE", "CAD", "GDT-TS"],
            save_name="supp_mean_metrics_vs_seq_len.png",
        )
        self._run_mean_test_supp(
            metrics=["INF-WC", "INF-NWC", "INF-STACK", "INF-ALL", "RMSD"],
            save_name="supp_mean_metrics_vs_seq_len_inf.png",
        )

    def _run_metric_vs_seq_len(self, metric: str):
        df = self.df.copy()
        df["Name"] = df.index
        df.drop(
            df[
                (df["Dataset"] == "RNA3DB_LONG") & (~df["Name"].isin(RNA_NON_RIBOSOME))
            ].index,
            inplace=True,
        )
        df = df.replace(OLD_TO_NEW)
        df = df[~df["Dataset"].isin(["RNA3DB_Long"])]
        fig = px.scatter(
            df,
            x="Sequence length",
            y=metric,
            color="Dataset",
            color_discrete_map=COLOR_DATASET,
            labels={"Sequence length": "Sequence length (nt)"},
            hover_data=["Name"],
        )
        fig = update_fig_box_plot(fig, legend_coordinates=(0.87, -0.15))
        fig.update_traces(
            marker=dict(line=dict(width=2, color="black")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend=dict(
                x=0, y=-0.15, orientation="h", bordercolor="black", borderwidth=1.5
            ),
        )
        save_path = os.path.join(self.save_dir, "scatter", f"{metric}_vs_seq_len.png")
        fig.write_image(save_path, width=700, height=600, scale=3)

    def run_metrics_vs_seq_len(
        self,
        metrics: List,
        name: str,
        facet_col_wrap: int = 3,
        height: int = 600,
        legend_coordinates=(0.8, -0.15),
    ):
        df = self.df.copy()
        df["Name"] = df.index
        df = self.convert_df_to_len_plot(df, metrics)
        df = df.replace(OLD_TO_NEW)
        fig = px.scatter(
            df,
            x="Sequence length",
            y="Metric",
            color="Dataset",
            facet_col="Metric name",
            facet_col_wrap=facet_col_wrap,
            color_discrete_map=COLOR_DATASET,
            labels={"Sequence length": "Sequence length (nt)", "Metric": ""},
            hover_data=["Name"],
            facet_row_spacing=0.1,
            facet_col_spacing=0.05,
            range_x=[5, 6000],
            log_x=True,
        )
        fig = update_fig_box_plot(fig, legend_coordinates=legend_coordinates)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True, matches=None))
        fig.update_traces(
            marker=dict(line=dict(width=2, color="black")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend=dict(
                orientation="h", y=-0.18, x=0.1, bordercolor="black", borderwidth=1.5
            )
        )
        fig.update_layout(legend_title_text="Dataset:")
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.replace("Metric name=", ""))
        )
        fig.update_xaxes(minor=dict(ticks="inside", ticklen=10, showgrid=True))
        save_path = os.path.join(self.save_dir, "scatter", f"{name}_vs_seq_len.png")
        fig.write_image(save_path, width=1600, height=height, scale=2)

    def convert_df_to_len_plot(self, df: pd.DataFrame, metrics: List):
        out = {
            "Sequence length": [],
            "Metric": [],
            "Dataset": [],
            "Name": [],
            "Metric name": [],
        }
        out["Sequence length"] = list(df["Sequence length"].values) * len(metrics)
        out["Dataset"] = list(df["Dataset"].values) * len(metrics)
        out["Name"] = list(df["Name"].values) * len(metrics)
        for metric in metrics:
            out["Metric"] += df[metric].values.tolist()
            out["Metric name"] += [metric] * len(df)
        return pd.DataFrame(out)
