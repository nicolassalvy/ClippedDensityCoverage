"""To use after fig6_compute.py"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from ClippedDensityCoverage.tests.tests import format_metric_name

plt.rcParams.update(
    {
        "font.size": 28,
        "lines.linewidth": 5,  # make the lines thicker
        "lines.markersize": 10,
    }
)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    save_path = cfg.save_path
    df = pd.read_csv(save_path + f"result_eval_{cfg.dataset}.csv")
    save_dir = save_path + f"fig6_{cfg.dataset}/"
    os.makedirs(save_dir, exist_ok=True)

    df["model"] = df["file_name"].apply(lambda x: x.split("/")[0])

    y_axis_name = cfg.dataset == "CIFAR-10"

    model_to_category = {
        "ACGAN-Mod": "GAN",
        "ADM": "Diffusion",
        "ADM-dropout": "Diffusion",
        "ADMG": "Diffusion",
        "ADMG-ADMU": "Diffusion",
        "BigGAN": "GAN",
        "Consistency-set1": "Consistency",
        "Consistency-set2": "Consistency",
        "DDPM": "Diffusion",
        "Diff-ProjGAN": "GAN",
        "DiT-XL-2": "Diffusion",
        "DiT-XL-2-guided": "Diffusion",
        "Efficient-vdVAE": "VAE",
        "GigaGAN": "GAN",
        "iDDPM": "Diffusion",
        "InsGen": "GAN",
        "LDM": "Diffusion",
        "LOGAN": "GAN",
        "LSGM-ODE": "Diffusion",
        "Mask-GIT": "Transformer",
        "MHGAN": "GAN",
        "NVAE": "VAE",
        "PFGMPP": "Diffusion",
        "Projected-GAN": "GAN",
        "RESFLOW": "Normalizing Flow",
        "ReACGAN": "GAN",
        "RQ-Transformer": "Transformer",
        "StyleGAN": "GAN",
        "StyleGAN-XL": "GAN",
        "StyleGAN2-ada": "GAN",
        "StyleNAT": "GAN",
        "StyleSwin": "GAN",
        "Unleash-Trans": "Transformer",
        "WGAN-GP": "GAN",
        "iDDPM-DDIM": "Diffusion",
    }

    category_to_color = {
        "Consistency": "#8c564b",  # brown
        "Diffusion": "#2ca02c",  # green
        "Normalizing Flow": "#9467bd",  # purple
        "GAN": "#1f77b4",  # blue
        "Transformer": "#ff7f0e",  # orange
        "VAE": "#d62728",  # red
    }

    markers = ["o", "s", "^", "d", "*", "p", "h", "v", ">", "<", "X", "P"]

    metrics_of_interest = cfg.metrics_of_interest
    metric_categories = cfg.metric_categories
    categories = [metric_categories[metric] for metric in metrics_of_interest]

    fidelity_metrics = [
        metric
        for i, metric in enumerate(metrics_of_interest)
        if categories[i] == "fidelity"
    ]
    coverage_metrics = [
        metric
        for i, metric in enumerate(metrics_of_interest)
        if categories[i] == "coverage"
    ]

    for fidelity_metric, coverage_metric in zip(
        fidelity_metrics, coverage_metrics
    ):
        plt.figure(figsize=(6, 6))
        # Create dictionaries to track used markers per category
        category_markers = {
            category: [0] for category in model_to_category.values()
        }
        model_points = {}  # Store points by category and model

        plt.grid(zorder=0)
        plt.axhline(y=1, color="gray", linestyle="-", linewidth=3, zorder=1)
        plt.axvline(x=1, color="gray", linestyle="-", linewidth=3, zorder=1)

        for i, row in df.iterrows():
            model = row["model"]
            category = model_to_category.get(model, "Other")
            color = category_to_color.get(category, "gray")

            category_markers[category].append(
                category_markers[category][-1] + 1
            )
            marker_idx = category_markers[category][-1] % len(markers)

            point = plt.scatter(
                row[fidelity_metric],
                row[coverage_metric],
                marker=markers[marker_idx],
                color=color,
                zorder=2,
            )

            if category not in model_points:
                model_points[category] = {}
            if model not in model_points[category]:
                model_points[category][model] = point

        plt.xlabel(format_metric_name(fidelity_metric))
        if y_axis_name:
            plt.ylabel(format_metric_name(coverage_metric))

        max_val_fidelity = df[fidelity_metric].max() + 0.03
        max_val_coverage = df[coverage_metric].max() + 0.03
        max_val = max(max_val_fidelity, max_val_coverage)
        plt.xlim(-0.03, max(1.03, max_val_fidelity))
        plt.ylim(-0.03, max(1.03, max_val_coverage))
        if max_val <= 1.2:
            plt.gca().set_aspect("equal")
            tick_every = 0.2
            ticks = np.arange(0, max(1.03, max_val), tick_every)
            plt.xticks(ticks)
            plt.yticks(ticks)

        plt.subplots_adjust(
            left=0.15 if not y_axis_name else 0.2,
            right=0.95,
            top=0.95,
            bottom=0.15,
        )

        plt.savefig(
            f"{save_dir}{fidelity_metric}_vs_{coverage_metric}.pdf",
        )

        # Create separate figures for the legends
        fig_model_legend = plt.figure(figsize=(5, 8))
        model_handles = []

        sorted_categories = sorted(model_points.keys())
        for category in sorted_categories:
            # Sort models within each category alphabetically
            sorted_models = sorted(model_points[category].keys())
            for model in sorted_models:
                point = model_points[category][model]
                point.set_label(model)
                model_handles.append(point)

        fig_model_legend.legend(
            handles=model_handles, loc="center", frameon=False
        )
        plt.axis("off")

        fig_model_legend.savefig(
            f"{save_dir}legend_models.pdf", bbox_inches="tight"
        )

        fig_category_legend = plt.figure(figsize=(10, 1.5))
        sorted_categories = sorted(category_to_color.keys())
        category_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=category_to_color[category],
                markersize=15,
                label=category,
            )
            for category in sorted_categories
        ]
        fig_category_legend.legend(
            handles=category_handles,
            loc="center",
            ncol=len(sorted_categories),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        plt.axis("off")

        fig_category_legend.savefig(
            f"{save_dir}legend_categories.pdf", bbox_inches="tight"
        )

        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
