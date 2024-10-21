# https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/convnextv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)
from wandb import Table, plot


## Layers
class GRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-8)
        return self.gamma * (x * nx) + self.beta + x


class Depthwise(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        ff_mult: int = 4,
    ):
        super(Depthwise, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=in_channels,
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * ff_mult, 1)
        self.conv3 = nn.Conv2d(out_channels * ff_mult, out_channels, 1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.grn = GRN()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        x = self.grn(x)
        x = self.conv3(x)

        x += self.skip(residual)
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 5,
        stride: int = 1,
        ff_mult: int = 4,
    ):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            *[
                Depthwise(in_channels, out_channels, kernel_size, stride, ff_mult)
                if _ == 0
                else Depthwise(out_channels, out_channels, kernel_size, 1, ff_mult)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.layers(x)


class Model(LightningModule):
    logger: WandbLogger

    def __init__(
        self,
        in_channels: int,
        width: int,
        labels: dict[str, int] | None = None,
        num_classes: int = 10,
        layers_per_block: tuple[int, int, int, int] = (2, 2, 6, 2),
        mult_per_layer: tuple[int, int, int, int] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.normin = nn.GroupNorm(in_channels, in_channels)
        self.block1 = Block(in_channels, width * mult_per_layer[0], layers_per_block[0])
        self.block2 = Block(
            width * mult_per_layer[0],
            width * mult_per_layer[1],
            layers_per_block[1],
        )
        self.block3 = Block(
            width * mult_per_layer[1],
            width * mult_per_layer[2],
            layers_per_block[2],
        )
        self.block4 = Block(
            width * mult_per_layer[2],
            width * mult_per_layer[3],
            layers_per_block[3],
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(width * mult_per_layer[3] * 4, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

        self.save_hyperparameters(ignore=["labels"])

        if labels is None:
            labels = {str(i): i for i in range(num_classes)}

        self.labels = labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normin(x)
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = F.max_pool2d(x, 2)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    @torch.jit.export
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train/loss", loss)
        self.log(
            "train/top1",
            multiclass_accuracy(logits, y, len(self.labels), top_k=1),
        )
        self.log(
            "train/top2",
            multiclass_accuracy(logits, y, len(self.labels), top_k=2),
        )

        return loss

    @torch.jit.export
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val/loss", loss)
        self.log(
            "val/top1",
            multiclass_accuracy(logits, y, len(self.labels), top_k=1),
        )
        self.log(
            "val/top2",
            multiclass_accuracy(logits, y, len(self.labels), top_k=2),
        )

        return loss

    def on_test_start(self) -> None:
        self.test_preds = []
        self.test_targets = []

    @torch.jit.export
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("test/loss", loss, on_epoch=True)
        self.log(
            "test/top1",
            multiclass_accuracy(logits, y, len(self.labels), top_k=1),
        )
        self.log(
            "test/top2",
            multiclass_accuracy(logits, y, len(self.labels), top_k=2),
        )

        self.test_preds.append(logits)
        self.test_targets.append(y)

        return loss

    def on_test_epoch_end(self) -> None:
        preds = torch.cat(self.test_preds).cpu()
        targets = torch.cat(self.test_targets).cpu()

        cm = plot.confusion_matrix(
            preds=preds.argmax(dim=1).tolist(),
            y_true=targets.tolist(),
            class_names=list(self.labels.keys()),
            title="Confusion Matrix",
        )

        # zero out the diagonal
        cm_rows = cm._data.iterrows()
        rows = []

        for row in cm_rows:
            r = row[1]
            r = r if r[0] != r[1] else (*r[:-1], 0)

            rows.append(r)

        cm._data = Table(columns=cm._data.columns, data=rows)

        self.logger.experiment.log({"confusion_matrix": cm})

        table = Table(
            columns=["Label", "Top-1", "Top-2", "Precision", "Recall", "F1"],
            data=[
                [
                    k,
                    multiclass_accuracy(
                        preds[targets == v],
                        targets[targets == v],
                        len(self.labels),
                        top_k=1,
                    ),
                    multiclass_accuracy(
                        preds[targets == v],
                        targets[targets == v],
                        len(self.labels),
                        top_k=2,
                    ),
                    multiclass_precision(
                        targets[targets == v],
                        preds[targets == v].argmax(dim=1),
                        num_classes=len(self.labels),
                    ),
                    multiclass_recall(
                        targets[targets == v],
                        preds[targets == v].argmax(dim=1),
                        num_classes=len(self.labels),
                    ),
                    multiclass_f1_score(
                        targets[targets == v],
                        preds[targets == v].argmax(dim=1),
                        num_classes=len(self.labels),
                    ),
                ]
                for k, v in self.labels.items()
            ],
        )

        self.logger.experiment.log(
            {
                "charts/top1": plot.bar(
                    table,
                    title="Top-1 Accuracy",
                    label="Label",
                    value="Top-1",
                ),
                "charts/top2": plot.bar(
                    table,
                    title="Top-2 Accuracy",
                    label="Label",
                    value="Top-2",
                ),
                "charts/precision": plot.bar(
                    table,
                    title="Precision",
                    label="Label",
                    value="Precision",
                ),
                "charts/recall": plot.bar(
                    table,
                    title="Recall",
                    label="Label",
                    value="Recall",
                ),
                "charts/f1": plot.bar(
                    table,
                    title="F1 Score",
                    label="Label",
                    value="F1",
                ),
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=8e-4)


if __name__ == "__main__":
    from lightning.pytorch.utilities import measure_flops
    from torchinfo import summary

    with torch.device("meta"):
        model = Model(
            1,
            6,
            {str(k): v for k, v in enumerate(range(10))},
            10,
            layers_per_block=(2, 2, 6, 2),
            mult_per_layer=(1, 2, 4, 8),
        )
        x = torch.randn(1, 1, 32, 32)

    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)

    model_loss = lambda y: F.cross_entropy(model(x), y)
    fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)

    print(f"FLOPs (forward only): {fwd_flops:,}")
    print(f"FLOPs (forward and backward): {fwd_and_bwd_flops:,}")

    summary(
        Model(
            1,
            12,
            {str(k): v for k, v in enumerate(range(36))},
            36,
            layers_per_block=(1, 1, 3, 1),
            mult_per_layer=(1, 2, 4, 4),
        ),
        (1, 1, 32, 32),
    )
