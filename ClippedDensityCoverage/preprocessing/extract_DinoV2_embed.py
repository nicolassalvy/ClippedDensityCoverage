import os
import sys
import hydra
from omegaconf import DictConfig

from dgm_eval.__main__ import main as dgm_eval_main


@hydra.main(version_base=None, config_path="../..", config_name="config")
def main(config: DictConfig):
    filtered_data_dir = config.filtered_data_dir
    embed_dir = config.data_dir_embeddings
    os.makedirs(embed_dir, exist_ok=True)

    # FD-DinoV2
    sys.argv = [
        "dgm-eval",
        filtered_data_dir,
        filtered_data_dir,  # Does not matter
        "--model",
        "dinov2",
        "--metrics",
        "fd",
        "--nsample",
        "50000",
        "--output_dir",
        embed_dir,
        "--save",
    ]
    dgm_eval_main()


if __name__ == "__main__":
    main()
