import os

from src.benchmark.models_helper import ModelsHelper
from src.viz.bar_helper import BarHelper
from src.viz.compute_p_value import ComputePValue
from src.viz.nwc_helper import NWCHelper
from src.viz.scatter_helper import ScatterHelper
from src.viz.table_helper import TableHelper


class VizAlphaFold:
    def __init__(self):
        self.save_dir = os.path.join("data", "plots")
        self.family_df_path = os.path.join("data", "info", "all_datasets_family.csv")

    def get_merged_model_df(self):
        """
        Get the merged model dataframe: metric for each model for each RNA and each dataset.
        :return:
        """
        params = {
            "in_dir": os.path.join("data", "output"),
            "info_dataset_path": os.path.join("data", "info", "info_datasets.csv"),
            "save_path": os.path.join("data", "info", "models_all_datasets.csv"),
        }
        model_helper = ModelsHelper(**params)
        df = model_helper.merge_af_predictions()
        df.index = df["Name"].apply(lambda x: x.lower().replace(".csv", ""))
        return df

    def run(self):
        df = self.get_merged_model_df()
        table_helper = TableHelper(df, self.save_dir)
        table_helper.run()
        table_helper.show_context_vs_no_context()
        ComputePValue(df).run()
        nwc_helper = NWCHelper(df, self.save_dir)
        nwc_helper.get_viz()
        scatter_helper = ScatterHelper(df, self.save_dir)
        scatter_helper.get_viz()
        scatter_helper.get_viz_mean()
        scatter_helper.get_viz_diff()
        BarHelper.get_viz()


if __name__ == "__main__":
    viz_alphafold = VizAlphaFold()
    viz_alphafold.run()
