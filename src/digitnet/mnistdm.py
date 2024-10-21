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


class MNISTDataModule(LightningDataModule):
    train_data: datasets.MNIST
    val_data: datasets.MNIST

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
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

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
            self.train_data = datasets.MNIST(
                self.data_dir, train=True, transform=transform
            )
            self.val_data = datasets.MNIST(
                self.data_dir, train=False, transform=transform
            )

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            generator=generator,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            generator=generator,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    @property
    def labels(self) -> dict[str, int]:
        assert hasattr(self, "train_data"), "Data has not been loaded"
        return {str(i): i for i in range(10)}


if __name__ == "__main__":
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()
    print(dm.labels)
    print(len(dm.train_data))
    print(len(dm.val_data))
    print(next(iter(dm.train_dataloader())))
    print(next(iter(dm.val_dataloader())))

    from matplotlib import pyplot as plt
    from torchvision.utils import make_grid

    for i, (x, y) in enumerate(dm.train_dataloader()):
        plt.figure(figsize=(8, 6))
        plt.imshow(make_grid(x, nrow=16).permute(1, 2, 0), interpolation="nearest")
        plt.title("MNIST Examples", fontdict={"fontsize": 16})
        plt.axis("off")
        plt.show()

        if i == 0:
            break
