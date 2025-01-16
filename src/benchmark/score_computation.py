import os


METRICS = "RMSD,P-VALUE,INF,DI,MCQ,TM-SCORE,CAD,BARNABA,CLASH,GDT-TS,lDDT,LCS-TA"

DOCKER_COMMAND = (
    "docker run -it -v ${PWD}/data/:/app/data "
    + "sayby77/rnadvisor --pred_path $PRED_PATH "
    + "--native_path $NATIVE_PATH --result_path $OUTPUT_PATH "
    + f'--all_scores={METRICS} --params=\'{{"mcq_threshold": 10, "mcq_mode": 2}}\''
)


class ScoreComputation:
    def __init__(self, native_paths: str, preds_paths: str, output_path: str):
        self.native_paths = native_paths
        self.preds_paths = preds_paths
        self.output_path = output_path

    def run_benchmark(self):
        """
        Compute scores for all the predictions.
        """
        os.makedirs(self.output_path, exist_ok=True)
        for challenge in os.listdir(self.native_paths):
            if "RNA3DB" in self.native_paths:
                pred_path = os.path.join(
                    self.preds_paths, challenge.lower().replace(".pdb", "")
                )
            else:
                pred_path = os.path.join(
                    self.preds_paths, challenge.replace(".pdb", "")
                )
            if os.path.isdir(pred_path):
                native_path = os.path.join(self.native_paths, challenge)
                if "RNA3DB" in self.native_paths:
                    output_path = os.path.join(
                        self.output_path, challenge.lower().replace(".pdb", ".csv")
                    )
                else:
                    output_path = os.path.join(
                        self.output_path, challenge.replace(".pdb", ".csv")
                    )
                if not os.path.exists(output_path):
                    self.compute_challenge(native_path, pred_path, output_path)

    def compute_challenge(
        self,
        native_path: str,
        pred_path: str,
        output_path: str,
    ):
        """
        Run the docker command to compute all the metrics
        :return:
        """
        command = (
            DOCKER_COMMAND.replace("$PRED_PATH", pred_path)
            .replace("$NATIVE_PATH", native_path)
            .replace("$OUTPUT_PATH", output_path)
        )
        os.system(command)


if __name__ == "__main__":
    prefix = os.path.join("data", "pdb")
    out_dir = os.path.join("data", "output")
    datasets = ["RNA_PUZZLES", "RNASOLO", "CASP_RNA", "RNA3DB", "RNA3DB_LONG"]
    for dataset in datasets:
        NATIVE_PATHS = os.path.join(prefix, dataset, "NATIVE")
        PREDS_PATHS = os.path.join(prefix, dataset, "PREDS")
        OUTPUT_PATH = os.path.join(out_dir, dataset)
        score_computation = ScoreComputation(NATIVE_PATHS, PREDS_PATHS, OUTPUT_PATH)
        score_computation.run_benchmark()
