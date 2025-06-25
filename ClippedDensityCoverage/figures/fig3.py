# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from ClippedDensityCoverage.tests.tests import make_toy_dataset
from ClippedDensityCoverage.metrics.Clipped_Density_Coverage import (
    ClippedDensityCoverage,
)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    # Parameters
    N = cfg.N
    save_path = cfg.save_path
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    x_s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_clipped_coverage_scores = []
    all_corrected_scores = []

    for repeat in tqdm(range(cfg.n_repeat)):
        real_data, synthetic_data = make_toy_dataset(n=N, d=cfg.d, n_modes=1)

        clippedDensityCoverage = ClippedDensityCoverage(
            real_data=real_data,
            K=cfg.nearest_k,
            n_jobs=cfg.n_jobs,
        )

        clipped_coverage_score = []
        for x in x_s:
            M_x = int(N * (1 - x))
            n_oods = N - M_x

            syn_data_oods = np.copy(synthetic_data)
            if n_oods > 0:
                syn_data_oods[:n_oods] *= 10

            clipped_coverage_score.append(
                clippedDensityCoverage.ClippedCoverage(
                    synthetic_data=syn_data_oods, _without_g=True
                )
            )

        all_clipped_coverage_scores.append(clipped_coverage_score)
        all_corrected_scores.append(
            clippedDensityCoverage._g(
                clipped_coverage_score,
                clippedDensityCoverage.f_coverage_values,
            )
        )

    all_clipped_coverage_scores = np.array(all_clipped_coverage_scores)
    all_corrected_scores = np.array(all_corrected_scores)

    mean_clipped = np.mean(all_clipped_coverage_scores, axis=0)
    std_clipped = np.std(all_clipped_coverage_scores, axis=0)
    mean_corrected = np.mean(all_corrected_scores, axis=0)
    std_corrected = np.std(all_corrected_scores, axis=0)

    f_values = clippedDensityCoverage.f_coverage_values

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.errorbar(
        x_s,
        mean_clipped,
        yerr=std_clipped,
        linestyle="-",
        marker="o",
        label="With clipping",
    )

    plt.plot(
        1 - np.arange(0, N + 1) / N,
        f_values,
        linestyle="--",
        linewidth=10,
        label="Theoretical",
    )

    plt.errorbar(
        x_s,
        mean_corrected,
        yerr=std_corrected,
        linestyle="--",
        marker="o",
        label="ClippedCoverage",
    )

    plt.xlabel("Proportion of bad samples")

    plt.grid(True, linestyle="--", alpha=0.7)

    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(
        [handles[1], handles[0], handles[2]],
        [
            "ClippedCoverage$_{\\text{unnorm}}$",
            "Theoretical",
            "ClippedCoverage",
        ],
    )

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + "fig3.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
