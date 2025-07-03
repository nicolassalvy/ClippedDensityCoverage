# Enhanced Generative Model Evaluation with Clipped Density and Coverage

This repository is the official implementation of [*Enhanced Generative Model Evaluation with Clipped Density and Coverage*](https://arxiv.org/abs/2507.01761).

## Setup

To install, navigate to the root directory of this repository. We recommend using Conda to create a dedicated environment:

```bash
conda create -n CDC python==3.10
conda activate CDC
pip install .
```

## Computing Clipped Density and Clipped Coverage

```python
from ClippedDensityCoverage.metrics.Clipped_Density_Coverage import ClippedDensityCoverage

CDC = ClippedDensityCoverage(
    real_data=real_data,
    K=5,
    n_jobs=n_jobs,
)

ClippedDensity = CDC.ClippedDensity(synthetic_data=synthetic_data)
ClippedCoverage = CDC.ClippedCoverage(synthetic_data=synthetic_data)
```

## Reproducing Results from the Paper

First, configure the following global parameters, either in `config.yaml` or by appending them as command-line arguments to any command below (`parameter_name=value`):

*   **`n_jobs`**: Number of CPU cores for parallel processing.
*   **`save_dir`**: Directory for saving results (default: `res/` in the current directory).
*   **`data_dir`**: Directory for storing data (default: `data/` in the current directory).

### Preprocessing

Prepare datasets using the following steps:

1.  **For Figure 6: Download and organize datasets**
    *   Download PNG datasets (see Appendix C).
    *   Store real datasets at: `data_dir/{dataset_name}/png/`
    *   Store generated datasets at: `data_dir/gen_{dataset_name}/{model_name}/png/`
        *   **Note:** Ensure `{model_name}` matches Figure 6 legend entries for the category mapping. If not, update `model_to_category` in `ClippedDensityCoverage/figures/fig6_plot.py` before running Figure 6.

2.  **For Figures 1, 4, 5: Prepare CIFAR-10**
    *   Download and convert CIFAR-10 to PNG format:
        ```bash
        python ClippedDensityCoverage/preprocessing/get_CIFAR-10_png.py
        ```
    *   Then, process CIFAR-10 as described in step 3.

3.  **Process all datasets (real and generated)**
    For each dataset, perform these two steps:
    *   **Filter samples:** Select 50,000 samples (class-balanced where applicable).
        ```bash
        python ClippedDensityCoverage/preprocessing/filter50000samples.py dataset=<dataset_identifier>
        ```
    *   **Extract DinoV2 embeddings:**
        ```bash
        python ClippedDensityCoverage/preprocessing/extract_DinoV2_embed.py dataset=<dataset_identifier>
        ```

    **Dataset identifiers (`<dataset_identifier>`):**
    *   Real datasets: `dataset_name` (e.g., `CIFAR-10`)
    *   Generated datasets: `gen_dataset_name/model_name` (e.g., `gen_FFHQ/LDM`)

    **Example (Generated FFHQ-LDM dataset):**
    ```bash
    python ClippedDensityCoverage/preprocessing/filter50000samples.py dataset=gen_FFHQ/LDM
    python ClippedDensityCoverage/preprocessing/extract_DinoV2_embed.py dataset=gen_FFHQ/LDM
    ```

#### Prepare out-of-distribution samples
Out-of-distribution samples for real experiments are embeddings of Gaussian noise images. To prepare them, generate PNG images of Gaussian noise and extract their DinoV2 embeddings:
```bash
python ClippedDensityCoverage/preprocessing/get_noise_png.py dataset=noise
python ClippedDensityCoverage/preprocessing/extract_DinoV2_embed.py dataset=noise
```

### Figure 2

```bash
python ClippedDensityCoverage/figures/fig2.py
```

### Figure 3

```bash
python ClippedDensityCoverage/figures/fig3.py N=50000 n_repeat=5
```

### Figures 1, 4, 5, 9, 10, 12, 13, 14, 15

Use the base command:
```bash
python ClippedDensityCoverage/tests/launch_tests.py
```
Append parameters specific to each figure:
-   **Figures 1, 4a, 12:** `real_ood_synthetic_proportion_test=True`
-   **Figures 4b, 13:** `real_simultaneous_mode_dropping_test=True metrics_of_interest="['Precision', 'Density', 'symPrecision', 'P-precision', 'PrecisionCover', 'alpha-Precision', 'TopP', 'ClippedDensity']"`
-   **Figures 4c, 5a, 14:** `real_ood_both_proportion_test=True`
-   **Figures 4d, 5b, 15:** `toy_translation_test=True n_repeat=5`
-   **Figures 9a, 10a:** `toy_ood_synthetic_proportion_test=True`
-   **Figures 9b:** `toy_simultaneous_mode_dropping_test=True metrics_of_interest="['Precision', 'Density', 'symPrecision', 'P-precision', 'PrecisionCover', 'alpha-Precision', 'TopP', 'ClippedDensity']"`
-   **Figures 9c, 10b:** `toy_ood_both_proportion_test=True`

### Figures 6, 7, 8

Run the following commands for each dataset (e.g. `CIFAR-10` or `FFHQ` for `<dataset_name>`):
```bash
python ClippedDensityCoverage/figures/fig6_compute.py dataset=<dataset_name>
python ClippedDensityCoverage/figures/fig6_plot.py dataset=<dataset_name>
```

### Figure 11

```bash
python ClippedDensityCoverage/figures/fig11.py dataset=FFHQ model_name=LDM
```
