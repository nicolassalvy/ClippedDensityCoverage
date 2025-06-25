import os
import hydra
from omegaconf import DictConfig
from torchvision import datasets
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    data_dir_dataset = cfg.data_dir_dataset
    data_dir_png = cfg.data_dir_png
    os.makedirs(data_dir_dataset, exist_ok=True)
    os.makedirs(data_dir_png, exist_ok=True)

    # Only the train set is needed
    train_dataset = datasets.CIFAR10(
        root=data_dir_dataset, train=True, download=True
    )

    for i in range(10):
        os.makedirs(os.path.join(data_dir_png, str(i)), exist_ok=True)
    for idx, (image, label) in tqdm(enumerate(train_dataset)):
        image_path = os.path.join(data_dir_png, str(label), f"{idx}.png")
        image.save(image_path)


if __name__ == "__main__":
    main()
