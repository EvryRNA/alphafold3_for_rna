import os
import pandas as pd
from typing import Tuple, List
import tqdm
from lib.rna_assessment.RNA_normalizer.structures.pdb_struct import PDBStruct


class CountInteractions:
    def __init__(
        self,
        save_dir: str,
        in_dir: str = os.path.join("data", "pdb"),
        mc_annotate_bin=os.path.join("lib", "rna_assessment", "MC-Annotate"),
    ):
        self.save_dir = save_dir
        self.in_dir = in_dir
        self.mc_annotate_bin = mc_annotate_bin

    def run(self):
        datasets = [name for name in os.listdir(self.in_dir) if name != ".DS_Store"]
        output = {
            "Dataset": [],
            "Interaction type": [],
            "Number of interactions": [],
            "RNA": [],
        }
        for dataset in datasets:
            in_path = os.path.join(self.in_dir, dataset, "NATIVE")
            interactions, counts, rnas = self.count_interactions(in_path)
            output["Dataset"] += [dataset] * len(interactions)
            output["Interaction type"] += interactions
            output["Number of interactions"] += counts
            output["RNA"] += rnas
        output = pd.DataFrame(output)
        output.to_csv(self.save_dir, index=False)
        return output

    def count_interactions(self, in_path: str) -> Tuple[List, List, List]:
        """
        Count the number of interactions in a directory of .pdb files
        :return: types of interactions, counts of interactions and RNA name
        """
        files = [
            os.path.join(in_path, name)
            for name in os.listdir(in_path)
            if name.endswith(".pdb")
        ]
        interactions, counts, rnas = [], [], []
        for in_file in tqdm.tqdm(files):
            rna = os.path.basename(in_file).split(".")[0]
            wc, nwc, stack = self.get_interactions(in_file)
            interactions.extend(["WC", "nWC", "STACK"])
            counts.extend([wc, nwc, stack])
            rnas.extend([rna] * 3)
        return interactions, counts, rnas

    def get_interactions(self, in_file: str):
        """
        Return the number of WC, nWC and STACK interactions in a .pdb file
        :param in_file:
        :return:
        """
        structure = PDBStruct(self.mc_annotate_bin)
        structure.load(in_file)
        interactions = ["PAIR_2D", "PAIR_3D", "STACK"]
        output = []
        for interaction in interactions:
            output.append(len(structure.get_interactions(interaction)))
        return output


if __name__ == "__main__":
    save_dir = os.path.join("data", "info", "interactions.csv")
    count_interactions = CountInteractions(save_dir)
    count_interactions.run()
