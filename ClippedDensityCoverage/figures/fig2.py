# %% Imports
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update(
    {
        "font.size": 28,
        "lines.linewidth": 5,
        "lines.markersize": 10,
    }
)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    # Parameters
    N = 3
    d = 2
    k = 2
    n_jobs = cfg.n_jobs
    save_path = cfg.save_path
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Data
    rng = np.random.default_rng()
    data = rng.normal(size=(N, d))
    data -= data.mean(axis=0)

    data_neighbors = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(
        data
    )

    distances, _ = data_neighbors.kneighbors(data)
    radius = distances[:, -1]

    synthetic_data = rng.normal(size=(N, d)) / 5

    synthetic_data[0] /= np.linalg.norm(synthetic_data[0])
    synthetic_data[0] *= 5

    # Plotting
    figure_step = 3

    plt.figure(figsize=(4, 4))

    if figure_step >= 1:
        for i in range(N):
            circle = Circle(
                (data[i, 0], data[i, 1]),
                radius=radius[i],
                fill=False,
                color="green",
                alpha=1,
            )
            plt.gca().add_patch(circle)

    plt.plot(data[:, 0], data[:, 1], "x", alpha=1)

    if figure_step >= 2:
        plt.plot(synthetic_data[:, 0], synthetic_data[:, 1], "o", color="red")

    if figure_step >= 3:
        # Create a separate figure for the legend
        legend_fig = plt.figure(figsize=(4, 1))
        legend_lines = [
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="blue",
                linestyle="None",
                markersize=30,
                label="Initial data",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="red",
                linestyle="None",
                markersize=30,
                label="Synthetic data",
            ),
        ]
        legend_fig.legend(handles=legend_lines, loc="center", ncol=1)
        legend_fig.tight_layout()
        if save_path is not None:
            legend_fig.savefig(
                f"{save_path}fig2_legend.pdf", bbox_inches="tight"
            )
        plt.close(legend_fig)

    plt.gca().set_aspect("equal")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(f"{save_path}fig2.pdf")
    plt.show()


if __name__ == "__main__":
    main()
