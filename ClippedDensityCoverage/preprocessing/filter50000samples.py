import os
import shutil
import hydra
from omegaconf import DictConfig
import numpy as np
from pathlib import Path


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(config: DictConfig):
    data_dir_png = config.data_dir_png
    filtered_data_dir = config.filtered_data_dir
    os.makedirs(filtered_data_dir, exist_ok=True)

    items_in_dir = os.listdir(data_dir_png)
    n_items = len(items_in_dir)

    are_all_dirs = all(
        os.path.isdir(os.path.join(str(data_dir_png), p)) for p in items_in_dir
    )

    rng = np.random.default_rng(42)

    if (
        not are_all_dirs and n_items > 20000
    ):  # unconditional, select 50000 randomly
        selected_items = rng.choice(items_in_dir, size=50000, replace=False)
        for item in selected_items:
            item_path = os.path.join(data_dir_png, item)
            if os.path.isfile(item_path):
                new_item_path = os.path.join(filtered_data_dir, item)
                shutil.copy2(item_path, new_item_path)
    else:  # conditional, select as many samples per class
        # Quickly deal with the case where there are only 50000 files
        images_paths = [
            p for p in Path(data_dir_png).rglob("*.png") if p.is_file()
        ]
        if len(images_paths) == 50000:
            for dir in items_in_dir:
                shutil.copytree(
                    data_dir_png, filtered_data_dir, dirs_exist_ok=True
                )
        else:
            print("Found", len(items_in_dir))
            n_keep_per_class = 50000 // len(items_in_dir)
            print(
                "Keep", n_keep_per_class, "per class"
            )  # help debug hidden files

            for folder in items_in_dir:
                folder_path = os.path.join(data_dir_png, folder)
                if os.path.isdir(folder_path):  # confirm it's a directory
                    items_in_folder = os.listdir(folder_path)
                    n_items_in_folder = len(items_in_folder)
                    if n_items_in_folder >= n_keep_per_class:
                        selected_items = rng.choice(
                            items_in_folder,
                            size=n_keep_per_class,
                            replace=False,
                        )
                    else:
                        selected_items = items_in_folder
                    for item in selected_items:
                        item_path = os.path.join(folder_path, item)
                        new_item_path = os.path.join(
                            filtered_data_dir, folder, item
                        )
                        os.makedirs(
                            os.path.dirname(new_item_path), exist_ok=True
                        )
                        shutil.copy2(item_path, new_item_path)


if __name__ == "__main__":
    main()
