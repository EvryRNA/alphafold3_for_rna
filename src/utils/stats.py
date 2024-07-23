import os
import pandas as pd
import tqdm

from src.utils.utils import get_sequence_from_pdb
from selenium import webdriver
from selenium.webdriver.common.by import By


class Stats:
    def __init__(self, pdb_dir: str, info_path: str, family_path: str):
        self.pdb_dir = pdb_dir
        self.info_path = info_path
        self.family_path = family_path

    def get_family(self):
        if os.path.exists(self.family_path):
            return pd.read_csv(self.family_path)
        else:
            return self._get_family()

    def _get_family(self):
        df = self.get_seq_data()
        names = [name.split("_")[0].upper() for name in list(df["RNA"])]
        families = self.get_family_from_name(names)
        df["Family"] = families
        df.to_csv(self.family_path, index=False)
        return df

    def get_family_from_name(self, names):
        URL = "https://www.rcsb.org/structure/"
        XPATH_title = "/html/body/div[1]/div[3]/div[2]/div[1]/div/div[1]/h4/span"
        XPATH = '//*[@id="header_classification"]/strong/a'
        browser = webdriver.Firefox()
        f_names = []
        for name in tqdm.tqdm(names):
            browser.get(URL + name)
            try:
                title = browser.find_elements(By.XPATH, XPATH_title)[0].text
                family = browser.find_elements(By.XPATH, XPATH)[0].text
                if family == "RIBOSOME" or "ribo" in title.lower():
                    f_names.append("RIBOSOME")
                else:
                    f_names.append(family)
            except IndexError:
                f_names.append("Unknown")
        return f_names

    def get_stats(self):
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

    def get_seq(self, in_path: str):
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
        "family_path": os.path.join("data", "info", "all_datasets_family.csv"),
    }
    stats = Stats(**params)
    stats.get_stats()
    family = stats.get_family()
