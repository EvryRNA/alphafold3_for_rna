import os
import pandas as pd


class AlphaFoldHelper:
    def __init__(self, in_dir: str, info_dataset_path: str, save_path: str):
        """
        :param in_dir: directory where are saved the scores for each RNA
        :param info_dataset_path: path to a dataframe with the name, length
                    and sequence of RNA structures
        :param save_path: where to save the final dataframe
        """
        self.in_dir = in_dir
        self.info_df = pd.read_csv(info_dataset_path, index_col=[0], sep=",")
        self.info_df.index = self.info_df["RNA"].map(lambda x: x.lower())
        self.save_path = save_path

    def merge_af_predictions(
        self, to_remove_metrics=["QS-score", "LCS-TA-COVERAGE", "LCS-TA-RESIDUES"]
    ) -> pd.DataFrame:
        """
        Merge all the predictions from AlphaFold into one dataframe

        """
        out = {"Name": [], "Dataset": [], "Sequence length": []}
        datasets = [
            dataset for dataset in os.listdir(self.in_dir) if dataset != ".DS_Store"
        ]
        for dataset in datasets:
            in_dir = os.path.join(self.in_dir, dataset)
            rnas = [rna for rna in os.listdir(in_dir) if rna != ".DS_Store"]
            for rna in rnas:
                rna_name = rna.replace(".csv", "").lower()
                df = pd.read_csv(os.path.join(in_dir, rna), index_col=[0]).sort_values(
                    by="RMSD"
                )
                names = [name.split("_")[1] for name in df.index]
                index_af = names.index("alphafold3")
                content = df.iloc[index_af, :]
                for metric in content.index:
                    if metric not in to_remove_metrics:
                        out[metric] = out.get(metric, []) + [content[metric]]
                out["Name"].append(rna)
                out["Dataset"].append(dataset)
                seq_len = int(self.info_df.loc[rna_name]["Length"])
                out["Sequence length"].append(seq_len)
        out_df = pd.DataFrame(out)
        out_df.to_csv(self.save_path, index=False)
        return out_df


if __name__ == "__main__":
    params = {
        "in_dir": os.path.join("data", "output"),
        "info_dataset_path": os.path.join("data", "info", "info_datasets.csv"),
        "save_path": os.path.join("data", "info", "all_datasets.csv"),
    }
    af_helper = AlphaFoldHelper(**params)
    af_helper.merge_af_predictions()
