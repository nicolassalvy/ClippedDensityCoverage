import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ClippedDensityCoverage.metrics.call_all_metrics import call_evaluate_all

plt.rcParams.update(
    {
        "font.size": 28,
        "lines.linewidth": 5,
        "lines.markersize": 10,
    }
)


def format_metric_name(name):
    if name.startswith("Clipped"):
        return name + " (ours)"
    elif name == "alpha-Precision":
        return "$\\alpha$-Precision"
    elif name == "beta-Recall":
        return "$\\beta$-Recall"
    return name


def plot_single_metric(
    x_values,
    mean_values,
    std_values,
    metric_name,
    xlabel=None,
    ylabel=None,
    save_path=None,
    ideal_x=None,
    ideal_y=None,
    ideal_label="Ideal",
    ideal_style="--",
    ideal_color="green",
    ylim=None,
    xticks=None,
    xticklabels=None,
    xscale=None,
    show_legend=False,
):
    fig, ax = plt.subplots(figsize=(12, 8))
    if std_values is None:
        ax.plot(
            x_values,
            mean_values,
            marker="o",
            label=format_metric_name(metric_name),
        )
    else:
        ax.errorbar(
            x_values,
            mean_values,
            yerr=std_values,
            marker="o",
            capsize=5,
            label=format_metric_name(metric_name),
        )
    if ideal_x is not None and ideal_y is not None:
        ax.plot(
            ideal_x,
            ideal_y,
            linestyle=ideal_style,
            color=ideal_color,
            label=ideal_label,
        )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if xscale:
        ax.set_xscale(xscale)
    if show_legend:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_multiple_normalized_metrics_with_legend(
    x_values,
    normalized_means,
    normalized_stds,
    metric_names,
    xlabel=None,
    ylabel=None,
    save_path=None,
    ideal_x=None,
    ideal_y=None,
    ideal_label="Ideal",
    ideal_style="--",
    ideal_color="black",
    xticks=None,
    xticklabels=None,
    xscale=None,
):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
    markers = ["o", "s", "^", "D", "v", "p", "*", "X", "<", ">"]
    plot_handles = []
    ideal_handle = None

    for i, metric in enumerate(metric_names):
        (line,) = ax.plot(
            x_values,
            normalized_means[i],
            label=format_metric_name(metric),
            marker=markers[i % len(markers)],
            color=colors[i],
            markersize=8,
            alpha=0.8,
            linewidth=2.5,
        )
        plot_handles.append(line)
        if normalized_stds is not None:
            ax.fill_between(
                x_values,
                normalized_means[i] - normalized_stds[i],
                normalized_means[i] + normalized_stds[i],
                color=colors[i],
                alpha=0.2,
            )

    if ideal_x is not None and ideal_y is not None:
        (ideal_line,) = ax.plot(
            ideal_x,
            ideal_y,
            linestyle=ideal_style,
            label=ideal_label,
            color=ideal_color,
            linewidth=2.5,
        )
        ideal_handle = ideal_line

    ax.grid(True, linestyle="--", alpha=0.7)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if xscale:
        ax.set_xscale(xscale)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    if save_path:
        main_plot_filename = f"{save_path}_PLOT.pdf"
        plt.savefig(main_plot_filename, bbox_inches="tight")
    plt.show()

    if plot_handles:
        handles_for_legend = plot_handles
        labels_for_legend = [h.get_label() for h in plot_handles]
        if ideal_handle:
            handles_for_legend.append(ideal_handle)
            labels_for_legend.append(ideal_handle.get_label())

        num_items = len(handles_for_legend)
        legend_fig_height = max(1, num_items * 0.3)
        legend_fig_width = 5

        fig_legend = plt.figure(figsize=(legend_fig_width, legend_fig_height))
        fig_legend.legend(
            handles_for_legend, labels_for_legend, loc="center", frameon=False
        )
        plt.axis("off")

        if save_path:
            legend_filename = f"{save_path}_LEGEND.pdf"
            fig_legend.savefig(legend_filename, bbox_inches="tight")
        plt.show()
        plt.close(fig_legend)

    plt.close(fig)


def make_toy_dataset(n=50000, d=8, n_modes=1, seed=42):
    rng = np.random.default_rng(seed)

    if n_modes == 1:
        return (rng.normal(size=(n, d)) for _ in range(2))

    mode_centers = rng.normal(scale=5, size=(n_modes, d))
    n_samples_per_mode = n // n_modes
    datasets = []
    for _ in range(2):
        datasets.append(
            [
                rng.normal(size=(n_samples_per_mode, d)) + mode_center
                for mode_center in mode_centers
            ]
        )
    return datasets[0], datasets[1]


def split_data_randomly(real_data, repeat=0, return_list=False):
    rng = np.random.RandomState(repeat)

    real_data_repeat = []
    synthetic_data_repeat = []
    for i in range(len(real_data)):
        indices = np.arange(len(real_data[i]))
        rng.shuffle(indices)
        split_idx = len(indices) // 2
        real_data_repeat.append(real_data[i][indices[:split_idx]])
        synthetic_data_repeat.append(real_data[i][indices[split_idx:]])

    mode_order = np.arange(len(real_data))
    rng.shuffle(mode_order)

    real_data_repeat = [real_data_repeat[i] for i in mode_order]
    synthetic_data_repeat = [synthetic_data_repeat[i] for i in mode_order]

    if return_list:
        return real_data_repeat, synthetic_data_repeat

    real_data_repeat = np.concatenate(real_data_repeat)
    synthetic_data_repeat = np.concatenate(synthetic_data_repeat)
    return real_data_repeat, synthetic_data_repeat


def ood_proportion_test(
    real_data,
    metrics_of_interest,
    categories=None,
    oods=None,
    nearest_k=5,
    n_jobs=8,
    save_path=None,
    where="synthetic",
    n_repeat=3,
):
    """Tests with out-of-distribution (ood) samples."""
    if where == "synthetic":
        ood_proportions = np.linspace(0, 1, 11)
    elif where == "both":
        ood_proportions = np.linspace(0, 0.03, 4)

    all_results = np.zeros(
        (n_repeat, len(metrics_of_interest), len(ood_proportions))
    )

    for repeat in range(n_repeat):
        rng = np.random.RandomState(repeat)

        real_data_repeat, synthetic_data_repeat = split_data_randomly(
            real_data, repeat=repeat
        )
        n_samples = real_data_repeat.shape[0]

        for j, ood_proportion in tqdm(
            enumerate(ood_proportions),
            desc=f"Repetition {repeat+1}/{n_repeat}",
            leave=(repeat == n_repeat - 1),
        ):
            real_data_iter = np.copy(real_data_repeat)
            gen_data_iter = np.copy(synthetic_data_repeat)
            n_oods = int(n_samples * ood_proportion)

            if n_oods > 0:
                ood_indices = rng.choice(n_samples, size=n_oods, replace=False)

                if oods is None:
                    ood_coeff = (10 + rng.standard_normal(n_oods)).reshape(
                        -1, 1
                    )
                    ood_coeff = np.maximum(2.0, ood_coeff)

                if where == "synthetic":
                    if oods is None:
                        gen_data_iter[ood_indices] *= ood_coeff
                    else:
                        gen_data_iter[ood_indices] = oods[:n_oods]
                elif where == "both":
                    if oods is None:
                        real_data_iter[ood_indices] *= ood_coeff
                        ood_coeff_synthetic = np.maximum(
                            2.0,
                            (10 + rng.standard_normal(n_oods)).reshape(-1, 1),
                        )
                        gen_data_iter[ood_indices] *= ood_coeff_synthetic
                    else:
                        real_data_iter[ood_indices] = oods[:n_oods]
                        gen_data_iter[ood_indices] = oods[n_oods : 2 * n_oods]

            res_proportion = call_evaluate_all(
                real_data=real_data_iter,
                synthetic_data=gen_data_iter,
                metrics_of_interest=metrics_of_interest,
                nearest_k=nearest_k,
                n_jobs=n_jobs,
            )

            for i, metric in enumerate(metrics_of_interest):
                all_results[repeat, i, j] = res_proportion[metric]

    if save_path:
        np.savez(
            f"{save_path}ood_proportion_test_data_{where}.npz",
            all_results=all_results,
            metrics_of_interest=np.array(metrics_of_interest, dtype=object),
            ood_proportions=ood_proportions,
            categories=(
                np.array(categories, dtype=object)
                if categories is not None
                else None
            ),
        )

        plot_ood_proportion_test(where=where, save_path=save_path)

    else:
        plot_ood_proportion_test(
            where=where,
            all_results=all_results,
            metrics_of_interest=metrics_of_interest,
            ood_proportions=ood_proportions,
            categories=categories,
        )


def plot_ood_proportion_test(
    where="synthetic",
    save_path=None,
    all_results=None,
    metrics_of_interest=None,
    ood_proportions=None,
    categories=None,
):
    """Either provide the results or the directory they are saved in."""
    if save_path is not None:
        save_path_file = f"{save_path}ood_proportion_test_data_{where}.npz"

        loaded_data = np.load(save_path_file, allow_pickle=True)
        all_results = loaded_data["all_results"]
        metrics_of_interest = loaded_data["metrics_of_interest"].tolist()
        ood_proportions = loaded_data["ood_proportions"]
        categories = (
            loaded_data["categories"].tolist()
            if "categories" in loaded_data
            and loaded_data["categories"] is not None
            else None
        )

    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    for i, metric in enumerate(metrics_of_interest):
        ideal_x = ood_proportions
        ideal_y = None
        ylim = None

        reference_value = mean_results[i, 0]
        if where == "synthetic":
            ideal_y = reference_value * (1 - ood_proportions)
        elif where == "both":
            ideal_y = [reference_value] * len(ood_proportions)
            min_val = np.min(mean_results[i] - std_results[i])
            max_val = np.max(mean_results[i] + std_results[i])
            ylim = (
                min(
                    min_val - 0.01,
                    reference_value - 0.01 * ood_proportions[-1] * 100,
                ),
                max(
                    max_val + 0.01,
                    reference_value + 0.01 * ood_proportions[-1] * 100,
                ),
            )

        tick_indices = np.linspace(
            0, len(ood_proportions) - 1, min(6, len(ood_proportions))
        ).astype(int)
        plot_ticks = ood_proportions[tick_indices]

        kind_of_bad_samples = "synthetic " if where == "synthetic" else ""
        xlabel = f"Proportion of bad {kind_of_bad_samples}samples"

        plot_single_metric(
            x_values=ood_proportions,
            mean_values=mean_results[i],
            std_values=std_results[i],
            metric_name=metric,
            xlabel=xlabel,
            ylabel="Score",
            save_path=(
                f"{save_path}ood_proportion_test_{metric}_{where}.pdf"
                if save_path
                else None
            ),
            ideal_x=ideal_x,
            ideal_y=ideal_y,
            xticks=plot_ticks,
            ylim=ylim,
        )

    if categories is not None:
        reference_values_norm = mean_results[:, 0]
        reference_values_norm[reference_values_norm == 0] = 1e-9
        normalized_mean = mean_results / reference_values_norm[:, np.newaxis]
        normalized_std = std_results / reference_values_norm[:, np.newaxis]

        ideal_x_norm = ood_proportions
        ideal_y_norm = None
        ref_value = 1.0
        if where == "synthetic":
            ideal_y_norm = ref_value * (1 - ood_proportions)
        elif where == "both":
            ideal_y_norm = [ref_value] * len(ood_proportions)

        for category in ["fidelity", "coverage"]:
            metrics_in_category = [
                m
                for i, m in enumerate(metrics_of_interest)
                if categories[i] == category
            ]

            category_indices = [
                i
                for i, _ in enumerate(metrics_of_interest)
                if categories[i] == category
            ]

            plot_multiple_normalized_metrics_with_legend(
                x_values=ood_proportions,
                normalized_means=normalized_mean[category_indices],
                normalized_stds=normalized_std[category_indices],
                metric_names=metrics_in_category,
                xlabel=xlabel,
                ylabel="Relative score",
                save_path=(
                    save_path
                    + f"ood_proportion_test_all_metrics_{category}_{where}"
                    if save_path
                    else None
                ),
                ideal_x=ideal_x_norm,
                ideal_y=ideal_y_norm,
                xticks=plot_ticks,
            )


def simultaneous_mode_dropping(
    real_data,
    metrics_of_interest,
    categories=None,
    save_path=None,
    nearest_k=5,
    n_jobs=8,
    n_repeat=3,
):
    n_modes = len(real_data)
    kept_ratios = np.linspace(1, 0, n_modes)
    # kept_ratios are the proportion of data to keep for each mode

    all_results = np.zeros((n_repeat, len(metrics_of_interest), n_modes))

    for repeat in range(n_repeat):
        real_data_repeat, synthetic_data_repeat = split_data_randomly(
            real_data, repeat=repeat, return_list=True
        )

        real_data_drop = np.concatenate(
            [
                real_data_repeat[i][: len(real_data_repeat[i]) // n_modes]
                for i in range(n_modes)
            ]
        )

        for j, kept_ratio in tqdm(
            enumerate(kept_ratios),
            desc=f"Repetition {repeat+1}/{n_repeat}",
            leave=(repeat == n_repeat - 1),
        ):
            synthetic_data_exp = np.vstack(
                [
                    synthetic_data_repeat[i][
                        : int(
                            kept_ratio
                            * (len(synthetic_data_repeat[i]) // n_modes)
                        )
                    ]
                    for i in range(1, n_modes)
                ]
            )

            # Fill the rest of the synthetic data with the first mode
            n_samples_left = (
                real_data_drop.shape[0] - synthetic_data_exp.shape[0]
            )
            if n_samples_left > 0:
                assert (
                    abs(
                        n_samples_left
                        - (real_data_drop.shape[0] // n_modes) * (j + 1)
                    )
                    <= n_modes
                )

                synthetic_data_exp = np.vstack(
                    [
                        synthetic_data_repeat[0][:n_samples_left],
                        synthetic_data_exp,
                    ]
                )
            assert synthetic_data_exp.shape[0] == real_data_drop.shape[0]

            res_mode = call_evaluate_all(
                real_data=real_data_drop,
                synthetic_data=synthetic_data_exp,
                metrics_of_interest=metrics_of_interest,
                nearest_k=nearest_k,
                n_jobs=n_jobs,
            )

            for i, metric in enumerate(metrics_of_interest):
                all_results[repeat, i, j] = res_mode[metric]

    if save_path:
        np.savez(
            f"{save_path}Simultaneous_mode_dropping_data.npz",
            all_results=all_results,
            metrics_of_interest=np.array(metrics_of_interest, dtype=object),
            kept_ratios=kept_ratios,
            categories=(
                np.array(categories, dtype=object)
                if categories is not None
                else None
            ),
            n_modes=n_modes,
        )

        plot_simultaneous_mode_dropping(save_path=save_path)
    else:
        plot_simultaneous_mode_dropping(
            all_results=all_results,
            metrics_of_interest=metrics_of_interest,
            kept_ratios=kept_ratios,
            categories=categories,
            n_modes=n_modes,
        )


def plot_simultaneous_mode_dropping(
    save_path=None,
    all_results=None,
    metrics_of_interest=None,
    kept_ratios=None,
    categories=None,
    n_modes=None,
):
    """Either provide the results or the directory they are saved in."""
    if save_path is not None and all_results is None:
        save_path_file = f"{save_path}Simultaneous_mode_dropping_data.npz"

        loaded_data = np.load(save_path_file, allow_pickle=True)
        all_results = loaded_data["all_results"]
        metrics_of_interest = loaded_data["metrics_of_interest"].tolist()
        kept_ratios = loaded_data["kept_ratios"]
        n_modes = loaded_data["n_modes"].item()
        categories = (
            loaded_data["categories"].tolist()
            if "categories" in loaded_data
            and loaded_data["categories"] is not None
            else None
        )

    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    reference_idx = 0

    for i, metric in enumerate(metrics_of_interest):
        ideal_x, ideal_y = None, None
        ylim = None
        if categories is not None:
            reference_value = mean_results[i, reference_idx]
            ideal_x = kept_ratios
            if categories[i] == "fidelity":
                ideal_y = [reference_value] * n_modes
                ylim = (0, np.max(mean_results[i] + std_results[i]) * 1.05)
            elif categories[i] == "coverage":
                ideal_y = (
                    (kept_ratios * (n_modes - 1) + 1)
                    / n_modes
                    * reference_value
                )

        plot_single_metric(
            x_values=kept_ratios,
            mean_values=mean_results[i],
            std_values=std_results[i],
            metric_name=metric,
            xlabel="Proportion of data kept per mode",
            ylabel=None,
            save_path=(
                f"{save_path}Simultaneous_mode_dropping_{metric}.pdf"
                if save_path
                else None
            ),
            ideal_x=ideal_x,
            ideal_y=ideal_y,
            ylim=ylim,
        )

    reference_values = mean_results[:, 0]
    reference_values[reference_values == 0] = 1e-9
    normalized_mean = mean_results / reference_values[:, np.newaxis]
    normalized_std = std_results / reference_values[:, np.newaxis]

    ideal_x_norm, ideal_y_norm = None, None

    if categories is not None:
        reference_value_norm = 1
        ideal_x_norm = kept_ratios

        for category in ["fidelity", "coverage"]:
            metrics_in_category = [
                m
                for i, m in enumerate(metrics_of_interest)
                if categories[i] == category
            ]

            category_indices = [
                i
                for i, _ in enumerate(metrics_of_interest)
                if categories[i] == category
            ]

            if category == "fidelity":
                ideal_y_norm = [reference_value_norm] * n_modes
            elif category == "coverage":
                ideal_y_norm = (
                    (kept_ratios * (n_modes - 1) + 1)
                    / n_modes
                    * reference_value_norm
                )

            plot_multiple_normalized_metrics_with_legend(
                x_values=ideal_x_norm,
                normalized_means=normalized_mean[category_indices],
                normalized_stds=normalized_std[category_indices],
                metric_names=metrics_in_category,
                xlabel="Proportion of data kept per mode",
                ylabel="Relative score",
                save_path=(
                    save_path
                    + f"Simultaneous_mode_dropping_all_metrics_{category}"
                    if save_path
                    else None
                ),
                ideal_x=ideal_x_norm,
                ideal_y=ideal_y_norm,
            )


def translation_test(
    real_data,
    metrics_of_interest,
    categories=None,
    n_translations=21,
    nearest_k=5,
    n_jobs=8,
    save_path=None,
    n_repeat=3,
):
    mus = np.linspace(-1, 1, n_translations)

    all_results = np.zeros((n_repeat, len(metrics_of_interest), len(mus)))

    for repeat in range(n_repeat):
        real_data_repeat, synthetic_data_repeat = split_data_randomly(
            real_data,
            repeat=repeat,
        )

        real_data_repeat[0] = 3 * np.ones_like(real_data_repeat[0])
        for j, mu in tqdm(
            enumerate(mus),
            desc=f"Repetition {repeat+1}/{n_repeat}",
            leave=(repeat == n_repeat - 1),
        ):
            translated_data = synthetic_data_repeat + mu
            translated_data[0] = -3 * np.ones_like(translated_data[0])

            res_mu = call_evaluate_all(
                real_data=real_data_repeat,
                synthetic_data=translated_data,
                metrics_of_interest=metrics_of_interest,
                nearest_k=nearest_k,
                n_jobs=n_jobs,
            )

            for i, metric in enumerate(metrics_of_interest):
                all_results[repeat, i, j] = res_mu[metric]

    if save_path:
        np.savez(
            f"{save_path}translation_test_data.npz",
            all_results=all_results,
            metrics_of_interest=np.array(metrics_of_interest, dtype=object),
            mus=mus,
            categories=(
                np.array(categories, dtype=object)
                if categories is not None
                else None
            ),
        )

        plot_translation_test(save_path=save_path)
    else:
        plot_translation_test(
            all_results=all_results,
            metrics_of_interest=metrics_of_interest,
            mus=mus,
            categories=categories,
        )


def plot_translation_test(
    save_path=None,
    all_results=None,
    metrics_of_interest=None,
    mus=None,
    categories=None,
):
    """Either provide the results or the directory they are saved in."""
    if save_path is not None and all_results is None:
        save_path_file = f"{save_path}translation_test_data.npz"

        loaded_data = np.load(save_path_file, allow_pickle=True)
        all_results = loaded_data["all_results"]
        metrics_of_interest = loaded_data["metrics_of_interest"].tolist()
        mus = loaded_data["mus"]
        categories = (
            loaded_data["categories"].tolist()
            if "categories" in loaded_data
            and loaded_data["categories"] is not None
            else None
        )

    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    # Find the middle index (where mu=0)
    mid_idx = len(mus) // 2

    for i, metric in enumerate(metrics_of_interest):
        plot_single_metric(
            x_values=mus,
            mean_values=mean_results[i],
            std_values=std_results[i],
            metric_name=metric,
            xlabel="μ",
            ylabel="Score",
            save_path=(
                f"{save_path}Translation_test_{metric}.pdf"
                if save_path
                else None
            ),
        )

    if categories is not None:
        reference_values = mean_results[:, mid_idx]
        reference_values[reference_values == 0] = 1e-9
        normalized_mean = mean_results / reference_values[:, np.newaxis]
        normalized_std = std_results / reference_values[:, np.newaxis]

        for category in ["fidelity", "coverage"]:
            metrics_in_category = [
                m
                for i, m in enumerate(metrics_of_interest)
                if categories[i] == category
            ]

            category_indices = [
                i
                for i, _ in enumerate(metrics_of_interest)
                if categories[i] == category
            ]

            if category_indices:
                plot_multiple_normalized_metrics_with_legend(
                    x_values=mus,
                    normalized_means=normalized_mean[category_indices],
                    normalized_stds=normalized_std[category_indices],
                    metric_names=metrics_in_category,
                    xlabel="μ",
                    ylabel="Relative score",
                    save_path=(
                        save_path + f"Translation_test_all_metrics_{category}"
                        if save_path
                        else None
                    ),
                )
