from typing import List
import os
import tqdm

from src.align.utils import align_structure


class AlignHelper:
    def __init__(self, pdb_path: str, datasets: List):
        self.pdb_path = pdb_path
        self.datasets = datasets

    def run(self):
        """
        Align the different structures with the native structure
        """
        for dataset in self.datasets:
            pdb_dir = os.path.join(self.pdb_path, dataset)
            native_dir = os.path.join(pdb_dir, "NATIVE")
            pred_dir = os.path.join(pdb_dir, "PREDS_CLEAN")
            aligned_dir = os.path.join(pdb_dir, "ALIGNED")
            os.makedirs(aligned_dir, exist_ok=True)
            self.align_dataset(native_dir, pred_dir, aligned_dir)

    def align_dataset(self, native_dir: str, pred_dir: str, aligned_dir: str):
        """
        Loop over the different predictions and align them with the native structure
        :return:
        """
        preds = [name for name in os.listdir(pred_dir) if name != ".DS_Store"]
        for pred in tqdm.tqdm(preds):
            native = os.path.join(native_dir, pred + ".pdb")
            pred_folder = os.path.join(pred_dir, pred)
            out_dir = os.path.join(aligned_dir, pred)
            os.makedirs(out_dir, exist_ok=True)
            self.align_structures(native, pred_folder, out_dir)

    def align_structures(self, native_path: str, pred_folder: str, out_dir: str):
        """
        Align the structures from the prediction folder with the native structure
        """
        preds = [name for name in os.listdir(pred_folder) if name.endswith(".pdb")]
        for pred in preds:
            out_path = os.path.join(out_dir, pred)
            pred_path = os.path.join(pred_folder, pred)
            if not os.path.exists(out_path):
                align_structure(native_path, pred_path, out_path)


if __name__ == "__main__":
    params = {
        "pdb_path": os.path.join("ata", "pdb"),
        "datasets": ["CASP_RNA", "RNA3DB", "RNA3DB_LONG", "RNA_PUZZLES", "RNASOLO"],
    }
    align_helper = AlignHelper(**params)
    align_helper.run()
