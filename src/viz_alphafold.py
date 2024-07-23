import os


from src.benchmark.alphafold_helper import AlphaFoldHelper
from src.viz.bar_helper import BarHelper
from src.viz.box_helper import BoxHelper
from src.viz.nwc_helper import NWCHelper
from src.viz.polar_helper import PolarHelper
from src.viz.scatter_helper import ScatterHelper


class VizAlphaFold:
    def __init__(self):
        self.df = self.get_merged_df()
        self.save_dir = os.path.join("data", "plots")
        self.family_df_path = os.path.join(
            "data", "info", "all_datasets_family.csv"
        )

    def get_merged_df(self):
        params = {
            "in_dir": os.path.join("data", "output"),
            "info_dataset_path": os.path.join(
                "data", "info", "info_datasets.csv"
            ),
            "save_path": os.path.join(
                "data", "info", "all_datasets.csv"
            ),
        }
        af_helper = AlphaFoldHelper(**params)
        df = af_helper.merge_af_predictions()
        df.index = df["Name"].apply(lambda x: x.lower().replace(".csv", ""))
        return df

    def run(self):
        nwc_helper = NWCHelper(self.df, self.save_dir)
        nwc_helper.get_viz()
        box_helper = BoxHelper(self.df, self.save_dir, self.family_df_path)
        box_helper.get_viz()
        box_helper.run_supplementary()
        scatter_helper = ScatterHelper(self.df, self.save_dir)
        scatter_helper.get_viz()
        scatter_helper.run_supplementary()
        BarHelper.get_viz()
        PolarHelper.get_viz()


if __name__ == "__main__":
    viz_alphafold = VizAlphaFold()
    viz_alphafold.run()
