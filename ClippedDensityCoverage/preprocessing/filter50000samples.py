import os
import shutil
import hydra
from omegaconf import DictConfig
import numpy as np


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(config: DictConfig):
    data_dir_png = config.data_dir_png
    filtered_data_dir = config.filtered_data_dir
    if not os.path.exists(filtered_data_dir):
        os.makedirs(filtered_data_dir)

    items_in_dir = os.listdir(data_dir_png)
    n_items = len(items_in_dir)

    rng = np.random.default_rng(42)

    if n_items > 20000:  # unconditional, select 50000 randomly
        selected_items = rng.choice(items_in_dir, size=50000, replace=False)
        for item in selected_items:
            item_path = os.path.join(data_dir_png, item)
            if os.path.isfile(item_path):
                new_item_path = os.path.join(filtered_data_dir, item)
                shutil.copy2(item_path, new_item_path)
    else:  # conditional, select as many samples per class
        n_keep_per_class = 50000 // len(items_in_dir)
        for folder in items_in_dir:
            folder_path = os.path.join(data_dir_png, folder)
            if os.path.isdir(folder_path):  # check if it's a directory
                items_in_folder = os.listdir(folder_path)
                n_items_in_folder = len(items_in_folder)
                if n_items_in_folder >= n_keep_per_class:
                    selected_items = rng.choice(
                        items_in_folder, size=n_keep_per_class, replace=False
                    )
                    for item in selected_items:
                        item_path = os.path.join(folder_path, item)
                        new_item_path = os.path.join(
                            filtered_data_dir, folder, item
                        )
                        if not os.path.exists(os.path.dirname(new_item_path)):
                            os.makedirs(os.path.dirname(new_item_path))
                        shutil.copy2(item_path, new_item_path)


if __name__ == "__main__":
    main()
