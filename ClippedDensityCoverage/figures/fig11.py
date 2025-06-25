import os
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pandas as pd
import csv

from ClippedDensityCoverage.metrics.call_all_metrics import call_evaluate_all
from ClippedDensityCoverage.tests.tests import plot_single_metric


def make_fig11(save_path_csv):
    df = pd.read_csv(save_path_csv)
    metrics = df.columns[1:]
    ks = df["k"]
    for metric in metrics:
        plot_single_metric(
            x_values=ks,
            mean_values=df[metric],
            std_values=None,
            metric_name=metric,
            xlabel="k",
            ylabel="Score",
            xticks=ks,
            xticklabels=ks,
            ylim=(-0.01, 1.01),
            save_path=save_path_csv.replace(
                "result_eval.csv", f"k_{metric}.pdf"
            ),
        )


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)

    save_path_csv = save_path + "fig11_result_eval.csv"

    metrics_of_interest = cfg.metrics_of_interest

    initial_data = np.load(cfg.data_path)["reps"]
    gen_data = np.load(cfg.gen_data_file)["reps"]

    with open(save_path_csv, "w", newline="") as csvfile:
        fieldnames = ["k"] + metrics_of_interest
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for nearest_k in tqdm([2, 3, 5, 7, 10]):
        res = call_evaluate_all(
            real_data=initial_data,
            synthetic_data=gen_data,
            metrics_of_interest=metrics_of_interest,
            nearest_k=nearest_k,
            n_jobs=cfg.n_jobs,
        )

        with open(save_path_csv, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {
                    "k": nearest_k,
                    **{metric: res[metric] for metric in metrics_of_interest},
                }
            )

    make_fig11(save_path_csv)


if __name__ == "__main__":
    main()
