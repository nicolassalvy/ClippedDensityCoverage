import os
import numpy as np
import hydra
from omegaconf import DictConfig

from ClippedDensityCoverage.metrics.call_all_metrics import call_evaluate_all
import csv


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)

    save_path_csv = save_path + f"result_eval_{cfg.dataset}.csv"
    real_data = np.load(cfg.data_path)["reps"]
    with open(save_path_csv, "w", newline="") as csvfile:
        fieldnames = ["file_name"] + cfg.metrics_of_interest
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    gen_data_dir = cfg.gen_data_dir
    generated_models = os.listdir(gen_data_dir)
    generated_files = [
        os.path.join(gen_model, "embed", cfg.embedding_file_name)
        for gen_model in generated_models
        if os.path.isdir(os.path.join(gen_data_dir, gen_model))
    ]

    for gen_data_file in generated_files:
        gen_data = np.load(os.path.join(gen_data_dir, gen_data_file))["reps"]
        print(f"--------{gen_data_file}--------")

        res = call_evaluate_all(
            real_data=real_data,
            synthetic_data=gen_data,
            metrics_of_interest=cfg.metrics_of_interest,
            nearest_k=cfg.nearest_k,
            n_jobs=cfg.n_jobs,
            timing=True,
        )

        with open(save_path_csv, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {
                    "file_name": gen_data_file,
                    **{
                        metric: res[metric]
                        for metric in cfg.metrics_of_interest
                    },
                }
            )


if __name__ == "__main__":
    main()
