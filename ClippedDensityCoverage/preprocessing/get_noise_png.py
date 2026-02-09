import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
from PIL import Image


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    data_dir_dataset = cfg.data_dir_dataset
    filtered_data_dir = cfg.filtered_data_dir
    os.makedirs(data_dir_dataset, exist_ok=True)
    os.makedirs(filtered_data_dir, exist_ok=True)

    shape = (32, 32, 3)
    for i in tqdm(range(50000)):
        img = ((np.random.randn(*shape) + 0.5) * 5).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"{filtered_data_dir}noise_{i}.png")


if __name__ == "__main__":
    main()
