import os
from typing import List

import pandas as pd
import plotly.express as px

from src.utils.enum import COLOR_DATASET, OLD_TO_NEW, ASC_METRICS, DESC_METRICS
from src.utils.utils import update_fig_box_plot
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ScatterHelper:
    def __init__(self, df: pd.DataFrame, save_dir: str):
        self.df = df
        self.save_dir = save_dir

    def get_viz_mean(self):
        self.run_mean_plot()
        self.get_viz_mean_supp()

    def get_viz_mean_supp(self):
        metrics = [
            ["BARNABA-eRMSD", "DI", "P-VALUE", "CAD", "GDT-TS"],
            ["INF-WC", "INF-NWC", "INF-STACK", "INF-ALL", "RMSD"],
        ]
        names = [
            "supp_mean_metrics_vs_seq_len.png",
            "supp_mean_metrics_vs_seq_len_inf.png",
        ]
        for metric, name in zip(metrics, names):
            fig = self.get_window_figure(metric, 3, 2, v_space=0.05, h_space=0.05)
            path_to_save = os.path.join(self.save_dir, "figures", "scatter_mean", name)
            fig.write_image(path_to_save, width=1000, height=500, scale=4)

    def get_viz(self):
        self.run_metrics_vs_seq_len(
            ["TM-score", "lDDT", "INF-ALL", "RMSD", "MCQ"],
            name="main_metrics",
            facet_col_wrap=3,
            height=1000,
        )
        self.run_supplementary()

    def run_supplementary(self):
        metrics = [
            ["CAD", "DI", "P-VALUE", "CAD", "GDT-TS", "BARNABA-eRMSD", "LCS-10"],
            ["INF-NWC", "INF-WC", "INF-STACK", "INF-ALL"],
        ]
        for metric in metrics:
            self.run_metrics_vs_seq_len(
                metric,
                name="".join(metric),
                facet_col_wrap=3,
                is_supp=True,
            )

    def _get_df_diff(self, metrics: List):
        df = self.df.copy()
        df = df[df["Metric_name"].isin(metrics)]
        df = df[df["Method"].isin(["alphafold3", "alphafold3c"])]
        df = df.replace(OLD_TO_NEW)
        df = df[df["Dataset"] == "RNA3DB_0"]
        df.index.name = "Names"
        names = df[(df["Method"] == "AlphaFold 3 (Context)")]["Name"].unique()
        df = df[df["Name"].isin(names)]
        pivot_df = df.pivot_table(
            index=["Name", "Dataset", "Sequence length", "Metric_name"],
            columns="Method",
            values="Metric",
            aggfunc="first",
        ).reset_index()
        pivot_df["Difference"] = (
            pivot_df["AlphaFold 3 (Context)"] - pivot_df["AlphaFold 3 (DL)"]
        )
        new_df = pivot_df[
            ["Name", "Dataset", "Sequence length", "Metric_name", "Difference"]
        ]
        return new_df

    def get_viz_diff(
        self,
        metrics=["RMSD", "MCQ", "TM-score", "lDDT", "INF-ALL"],
        save_name="diff_metrics.png",
    ):
        df = self._get_df_diff(metrics)
        df["Positive"] = df["Difference"] > 0
        df = df[df["Name"] != "8hsr_r.csv"]
        df.loc[df["Metric_name"].isin(DESC_METRICS), "Positive"] = ~df["Positive"]
        n_rna = len(df["Name"].unique())
        df_pos = (
            df[["Metric_name", "Positive"]].groupby("Metric_name").sum().reset_index()
        )
        df_pos["Positive"] = df_pos["Positive"] / n_rna
        df_pos["Positive"] = df_pos["Positive"].apply(lambda x: round(x, 3))
        print(df_pos)
        fig = px.scatter(
            df,
            x="Sequence length",
            y="Difference",
            color="Dataset",
            facet_col="Metric_name",
            facet_col_wrap=3,
            color_discrete_map=COLOR_DATASET,
            labels={"Sequence length": "Sequence length (nt)", "Metric": ""},
            hover_data=["Name"],
            facet_row_spacing=0.1,
            facet_col_spacing=0.05,
            category_orders={"Metric_name": metrics},
        )
        fig = update_fig_box_plot(fig, legend_coordinates=(0, 0))
        fig.update_traces(marker=dict(opacity=1))
        target_datasets = ["RNA3DB_0", "RNA3DB_0 (Context)"]
        fig.for_each_trace(
            lambda trace: trace.update(marker=dict(opacity=0.5))
            if trace.name in target_datasets
            else ()
        )
        fig.for_each_yaxis(
            lambda y: y.update(
                showticklabels=True,
                matches=None,
                autorange="reversed" if (y.anchor in ["x4", "x5"]) else None,
            )
        )
        for i in range(0, 3):
            for j in range(0, 3):
                try:
                    fig.add_hline(
                        y=0,
                        line_dash="dot",
                        row=i,
                        col=j,
                        line_color="red",
                        line_width=2,
                    )
                except IndexError:
                    pass
        fig.update_traces(
            marker=dict(symbol="x", line=dict(width=1, color="black")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend=dict(
                orientation="h", y=-0.18, x=0.1, bordercolor="black", borderwidth=1.5
            )
        )
        fig.update_layout(legend_title_text="Dataset:")
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.replace("Metric_name=", ""))
        )
        fig.update_xaxes(minor=dict(ticks="inside", ticklen=10, showgrid=True))
        fig.update_traces(marker=dict(size=9))
        path_to_save = os.path.join(self.save_dir, "figures", "diff", save_name)
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        fig.write_image(path_to_save, width=1200, height=600, scale=4)

    def run_mean_plot(
        self,
        metrics=["RMSD", "MCQ", "TM-score", "lDDT", "INF-ALL"],
        save_name="mean_metrics_vs_seq_len_all_models_scatter_25.png",
    ):
        fig = self.get_window_figure(metrics, 3, 2, v_space=0.05, h_space=0.05)
        path_to_save = os.path.join(self.save_dir, "figures", "scatter_mean", save_name)
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        fig.write_image(path_to_save, width=1000, height=500, scale=4)

    def run_metrics_vs_seq_len(
        self,
        metrics: List,
        name: str,
        facet_col_wrap: int = 3,
        height: int = 600,
        legend_coordinates=(0.8, -0.15),
        is_supp: bool = False,
    ):
        df = self.df.copy()
        df = df.replace({"LCS-TA-COVERAGE-10": "LCS-10"})
        df = df[df["Metric_name"].isin(metrics)]
        df = df[df["Method"].isin(["alphafold3", "alphafold3c"])]
        df.loc[df["Method"] == "alphafold3c", "Dataset"] = "RNA3DB_0 (Context)"
        df = df.replace(OLD_TO_NEW)
        df = df[
            ~df["Name"].isin(
                [
                    "8gh6_r.csv",
                    "8bvj_b.csv",
                    "8hsr_r.csv",
                ]
            )
        ]
        fig = px.scatter(
            df,
            x="Sequence length",
            y="Metric",
            color="Dataset",
            facet_col="Metric_name",
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
        fig.update_traces(marker=dict(opacity=1))
        target_datasets = ["RNA3DB_0", "RNA3DB_0 (Context)"]
        fig.for_each_trace(
            lambda trace: trace.update(marker=dict(opacity=0.5))
            if trace.name in target_datasets
            else ()
        )
        fig.for_each_yaxis(
            lambda y: y.update(
                showticklabels=True,
                matches=None,
                autorange="reversed" if (y.anchor in ["x4", "x6"]) else None,
            )
        )
        fig.update_traces(
            marker=dict(line=dict(width=1, color="black")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend=dict(
                orientation="h", y=-0.18, x=0.1, bordercolor="black", borderwidth=1.5
            )
        )
        fig.update_layout(legend_title_text="Dataset:")
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.replace("Metric_name=", ""))
        )
        fig.update_xaxes(minor=dict(ticks="inside", ticklen=10, showgrid=True))
        fig.update_traces(marker=dict(size=9))
        save_dir = os.path.join(self.save_dir, "figures", "scatter")
        os.makedirs(save_dir, exist_ok=True)
        if is_supp:
            save_path = os.path.join(save_dir, f"supp_{name}_vs_seq_len.png")
        else:
            save_path = os.path.join(save_dir, f"{name}_vs_seq_len.png")
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

    def get_window_figure(
        self,
        metrics: List,
        n_rows: int,
        cols: int,
        v_space: float = 0.1,
        h_space: float = 0.08,
    ):
        all_df = self._get_all_dfs(metric=metrics)
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
            desc_metrics = DESC_METRICS + ["εRMSD"]
            if metric in desc_metrics:
                fig.update_yaxes(
                    autorange="reversed", row=position[0], col=position[1], tick0=0
                )
            if metric in ASC_METRICS:
                fig.update_yaxes(range=[-0.1, 1.1], row=position[0], col=position[1])
            fig.update_xaxes(
                tick0=50, dtick=100, row=position[0], col=position[1], range=[0, 800]
            )
        fig.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.8)))
        fig.update_traces(mode="lines+markers")
        fig = update_fig_box_plot(fig)
        fig.update_annotations(font_size=8)
        fig.update_layout(font=dict(size=14))
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
                df = df.rename(columns={"LCS-TA-COVERAGE-10": "LCS-10"})
                seq_len = self.df[self.df["Name"] == rna]["Sequence length"].values[0]
                df["Approach"] = df.index.map(
                    lambda x: NAME_TO_APPROACHES.get(x, "Others")
                )
                df = self._get_top_df(metric, df)
                for row in df.iterrows():
                    name = row[1]["Approach"]
                    if name in list(NAME_TO_APPROACHES.values()):
                        all_df["Dataset"].extend([dataset] * len(metric))
                        all_df["RNA"].extend([rna.replace(".csv", "")] * len(metric))
                        all_df["Sequence length"].extend([seq_len] * len(metric))
                        all_df["Metric"].extend(row[1][metric])
                        all_df["Metric_name"].extend(metric)
                        method = name
                        all_df["Method"].extend([method] * len(metric))
        all_df = pd.DataFrame(all_df)
        return all_df

    def _get_top_df(self, metric: str, df: pd.DataFrame):
        condition_test = not all(m in df.columns for m in metric)
        if condition_test:
            return df
        df = df.sort_values(metric, ascending=True)
        df_min = df.groupby("Approach").min()[metric]
        df_max = df.groupby("Approach").max()[metric]
        new_df = pd.DataFrame()
        for m in metric:
            if m in DESC_METRICS:
                new_df[m] = df_min[m]
            else:
                new_df[m] = df_max[m]
        new_df = new_df.reset_index()
        return new_df

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
