import argparse

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import measure_flops

from digitnet import EMNISTDataModule, MNISTDataModule, Model

seed_everything(42, workers=True, verbose=False)


def main(args: argparse.Namespace):
    # Load the data
    if args.dataset == "mnist":
        data = MNISTDataModule()
    elif args.dataset == "emnist":
        data = EMNISTDataModule()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data.prepare_data()
    data.setup()

    # Load the model
    model = Model(
        in_channels=1,
        width=4,
        num_classes=len(data.labels),
        labels=data.labels,
        layers_per_block=(1, 1, 6, 1),
        mult_per_layer=(1, 2, 4, 8),
    )

    # Create logger
    logger = WandbLogger(
        project="digitnet",
        save_dir="logs",
        tags=[args.dataset],
        save_code=True,
        offline=False,
        log_model=True,
    )
    logger.watch(model)
    logger.log_hyperparams(
        {
            "num_params": sum(p.numel() for p in model.parameters()),
            "fwd_flops": measure_flops(model, lambda: model(torch.randn(1, 1, 32, 32))),
            "fwd_and_bwd_flops": measure_flops(
                model,
                lambda: model(torch.randn(1, 1, 32, 32)),
                lambda y: torch.nn.functional.cross_entropy(
                    model(torch.randn(1, 1, 32, 32)), y
                ),
            ),
        }
    )

    # Train the model
    trainer = Trainer(
        max_epochs=10,
        deterministic=True,
        logger=logger,
        callbacks=[
            ModelSummary(max_depth=2),
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=True,
                log_weight_decay=True,
            ),
            StochasticWeightAveraging(
                swa_lrs=8e-4,
                swa_epoch_start=10,
                annealing_epochs=2,
                device=None,
            ),
        ],
        enable_model_summary=False,
    )
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "emnist"],
        help="select the dataset to use",
    )
    args = parser.parse_args()

    main(args)
