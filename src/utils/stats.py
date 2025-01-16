import os
from typing import List, Tuple

import pandas as pd
import tqdm

from src.utils.utils import get_sequence_from_pdb


class Stats:
    def __init__(self, pdb_dir: str, info_path: str):
        self.pdb_dir = pdb_dir
        self.info_path = info_path

    def get_stats(self):
        """
        Get the min, max and mean length of the sequences for the different datasets
        :return:
        """
        df = self.get_seq_data()
        for dataset in df["Dataset"].unique():
            c_df = df[df["Dataset"] == dataset]
            len_min = c_df["Length"].min()
            len_max = c_df["Length"].max()
            len_mean = c_df["Length"].mean()
            print(
                f"{dataset} : {len(c_df)} structures: Min: {len_min} | Max: {len_max} | Mean: {len_mean:.2f}"
            )

    def get_seq_data(self):
        if os.path.exists(self.info_path):
            return pd.read_csv(self.info_path)
        else:
            return self._get_seq_data()

    def _get_seq_data(self):
        datasets = [name for name in os.listdir(self.pdb_dir) if name != ".DS_Store"]
        output = {"Dataset": [], "RNA": [], "Sequence": [], "Length": []}
        for dataset in datasets:
            in_path = os.path.join(self.pdb_dir, dataset, "NATIVE")
            rnas, sequences = self.get_seq(in_path)
            output["Dataset"] += [dataset] * len(rnas)
            output["RNA"] += rnas
            output["Sequence"] += sequences
            output["Length"] += [len(x) for x in sequences]
        output = pd.DataFrame(output)
        output.to_csv(self.info_path, index=False)
        return output

    def get_seq(self, in_path: str) -> Tuple[List[str], List[str]]:
        """
        Return the RNA names and the sequences from the .pdb files
        :param in_path: path to a directory with .pdb files
        :return: the RNA names and the associated sequences
        """
        pdb_files = [
            os.path.join(in_path, name)
            for name in os.listdir(in_path)
            if name.endswith(".pdb")
        ]
        rnas, sequences = [], []
        for in_rna in tqdm.tqdm(pdb_files):
            rna = os.path.basename(in_rna).split(".")[0]
            sequence = get_sequence_from_pdb(in_rna)
            rnas.append(rna)
            sequences.append(sequence)
        return rnas, sequences


if __name__ == "__main__":
    params = {
        "pdb_dir": os.path.join("data", "pdb"),
        "info_path": os.path.join("data", "info", "info_datasets.csv"),
    }
    stats = Stats(**params)
    stats.get_stats()
