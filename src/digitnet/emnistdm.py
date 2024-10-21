import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EMNISTDataModule(LightningDataModule):
    train_data_num: datasets.EMNIST
    train_data_alpha: datasets.EMNIST
    val_data_num: datasets.EMNIST
    val_data_alpha: datasets.EMNIST

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 42,
        degrees: tuple[float, float] = (-30, 30),
        translate: tuple[float, float] = (0.1, 0.2),
        scale: tuple[float, float] = (0.75, 1.25),
        shear: tuple[float, float] = (-10, 10),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def prepare_data(self):
        datasets.EMNIST(self.data_dir, train=True, split="mnist", download=True)
        datasets.EMNIST(self.data_dir, train=False, split="mnist", download=True)
        datasets.EMNIST(self.data_dir, train=True, split="letters", download=True)
        datasets.EMNIST(self.data_dir, train=False, split="letters", download=True)

    def setup(self, stage=None):
        transform = T.Compose(
            [
                T.Resize((32, 32), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomAffine(
                    degrees=(self.degrees[0], self.degrees[1]),
                    translate=(self.translate[0], self.translate[1]),
                    scale=(self.scale[0], self.scale[1]),
                    shear=(self.shear[0], self.shear[1]),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
        if stage == "fit" or stage is None:
            # Load data
            self.train_data_num = datasets.EMNIST(
                self.data_dir, train=True, transform=transform, split="mnist"
            )
            self.val_data_num = datasets.EMNIST(
                self.data_dir,
                train=False,
                transform=transform,
                split="mnist",
            )
            self.train_data_alpha = datasets.EMNIST(
                self.data_dir, train=True, transform=transform, split="letters"
            )
            self.val_data_alpha = datasets.EMNIST(
                self.data_dir, train=False, transform=transform, split="letters"
            )

            # Transpose the data
            print(self.train_data_num.data.shape)
            self.train_data_num.data = self.train_data_num.data.permute(0, 2, 1)
            self.val_data_num.data = self.val_data_num.data.permute(0, 2, 1)
            self.train_data_alpha.data = self.train_data_alpha.data.permute(0, 2, 1)
            self.val_data_alpha.data = self.val_data_alpha.data.permute(0, 2, 1)

            # Offset the labels ()
            self.train_data_alpha.targets += 9
            self.val_data_alpha.targets += 9

            print(self.train_data_alpha.targets.shape)

            # Combine the datasets
            self.train_data = torch.utils.data.ConcatDataset(
                [self.train_data_num, self.train_data_alpha]
            )
            self.val_data = torch.utils.data.ConcatDataset(
                [self.val_data_num, self.val_data_alpha]
            )

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            generator=generator,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            generator=generator,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    @property
    def labels(self) -> dict[str, int]:
        assert hasattr(
            self, "train_data_num"
        ), "Data not prepared, call prepare_data() and setup() first"
        assert hasattr(
            self, "train_data_alpha"
        ), "Data not prepared, call prepare_data() and setup() first"

        numbers = self.train_data_num.classes
        letters = self.train_data_alpha.classes

        if "N/A" in letters:
            letters.remove("N/A")  # idk why this is here, its not a valid class

        all = numbers + letters

        return {label: i for i, label in enumerate(all)}


if __name__ == "__main__":
    dm = EMNISTDataModule(num_workers=0)
    dm.prepare_data()
    dm.setup()
    print(dm.labels)
    print(len(dm.train_data))
    print(len(dm.val_data))
    print(next(iter(dm.train_dataloader()))[0].shape)
    print(next(iter(dm.val_dataloader()))[0].shape)

    from matplotlib import pyplot as plt
    from torchvision.utils import make_grid

    for i, (x, y) in enumerate(dm.train_dataloader()):
        print(x.shape, y.shape)
        plt.imshow(make_grid(x, nrow=16).permute(1, 2, 0))
        plt.show()
        if i > 0:
            break
