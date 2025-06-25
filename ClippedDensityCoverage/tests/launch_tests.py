import numpy as np
import hydra
import os
from omegaconf import DictConfig

from ClippedDensityCoverage.tests.tests import (
    make_toy_dataset,
    translation_test,
    ood_proportion_test,
    simultaneous_mode_dropping,
)


@hydra.main(
    version_base=None,
    config_path="../../",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # Create save paths
    save_path = cfg.save_path
    if save_path is not None:
        save_path_toy = save_path + "toy/"
        save_path_real = save_path + "real/"
        os.makedirs(save_path_toy, exist_ok=True)
        os.makedirs(save_path_real, exist_ok=True)

    n_jobs = cfg.n_jobs
    nearest_k = cfg.nearest_k
    N = cfg.N
    d = cfg.d
    n_repeat = cfg.n_repeat

    metrics_of_interest = cfg.metrics_of_interest
    metric_categories = cfg.metric_categories

    categories = [metric_categories[metric] for metric in metrics_of_interest]

    # SYNTHETIC DATA TESTS
    # Matched real & synthetic out-of-distribution sample proportion test
    if cfg.toy_ood_both_proportion_test:
        real_data, _ = make_toy_dataset(n=N * 2, d=d, n_modes=1)

        ood_proportion_test(
            [real_data],
            metrics_of_interest=metrics_of_interest,
            categories=categories,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_toy,
            where="both",
            n_repeat=n_repeat,
        )

    # Synthetic out-of-distribution sample proportion test
    if cfg.toy_ood_synthetic_proportion_test:
        real_data, _ = make_toy_dataset(n=N * 2, d=d, n_modes=1)

        ood_proportion_test(
            [real_data],
            metrics_of_interest=metrics_of_interest,
            categories=categories,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_toy,
            n_repeat=n_repeat,
        )

    # Mode dropping simultaneous
    if cfg.toy_simultaneous_mode_dropping_test:
        real_data, _ = make_toy_dataset(n=N * 20, d=d, n_modes=10)

        simultaneous_mode_dropping(
            real_data,
            metrics_of_interest=metrics_of_interest,
            categories=categories,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_toy,
            n_repeat=n_repeat,
        )

    # Translation test
    if cfg.toy_translation_test:
        real_data, _ = make_toy_dataset(n=N * 2, d=d, n_modes=1)

        translation_test(
            [real_data],
            n_translations=21,
            metrics_of_interest=metrics_of_interest,
            categories=categories,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_toy,
            n_repeat=n_repeat,
        )

    # LOAD REAL DATA (CIFAR10)
    if np.any(
        [
            cfg.real_ood_both_proportion_test,
            cfg.real_ood_synthetic_proportion_test,
            cfg.real_simultaneous_mode_dropping_test,
        ]
    ):
        real_data_real_full = np.load(cfg.data_path)["reps"]

        real_data_real_full_list = [
            real_data_real_full[i * 5000 : (i + 1) * 5000] for i in range(10)
        ]

    if np.any(
        [
            cfg.real_ood_both_proportion_test,
            cfg.real_ood_synthetic_proportion_test,
        ]
    ):  # load noise
        noise_reps = np.load(cfg.noise_path)["reps"]

    # REAL synthetic out-of-distribution sample proportion test
    if cfg.real_ood_synthetic_proportion_test:
        ood_proportion_test(
            real_data=real_data_real_full_list,
            oods=noise_reps,
            metrics_of_interest=metrics_of_interest,
            categories=categories,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_real,
            where="synthetic",
            n_repeat=n_repeat,
        )

    # REAL matched real & synthetic out-of-distribution sample proportion test
    if cfg.real_ood_both_proportion_test:
        ood_proportion_test(
            real_data=real_data_real_full_list,
            oods=noise_reps,
            metrics_of_interest=metrics_of_interest,
            categories=categories,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_real,
            where="both",
            n_repeat=n_repeat,
        )

    # REAL Mode dropping simultaneous for fidelity metrics
    if cfg.real_simultaneous_mode_dropping_test:
        metrics_of_interest_mode_dropping = [
            metric
            for metric in metrics_of_interest
            if metric_categories[metric] == "fidelity"
        ]
        categories_mode_dropping = ["fidelity"] * len(
            metrics_of_interest_mode_dropping
        )
        simultaneous_mode_dropping(
            real_data_real_full_list,
            metrics_of_interest=metrics_of_interest_mode_dropping,
            categories=categories_mode_dropping,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            save_path=save_path_real,
            n_repeat=n_repeat,
        )


if __name__ == "__main__":
    main()
