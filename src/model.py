import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class ModelConfig(PretrainedConfig):
    def __init__(
        self,
        width=4,
        in_channels=1,
        num_classes=10,
        kernel_size=5,
        padding=2,
        m=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.padding = padding
        self.m = m


class Model(PreTrainedModel):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__(config)
        self.config = config
        in_channels = config.in_channels
        width = config.width
        num_classes = config.num_classes
        kernel_size = config.kernel_size
        padding = config.padding
        m = config.m

        self.conv1 = Depthwise(in_channels, width, kernel_size, padding, m)
        self.conv2 = Depthwise(width, width * 2, kernel_size, padding, m)
        self.conv3a = Depthwise(width * 2, width * 4, kernel_size, padding, m)
        self.conv3b = Depthwise(width * 4, width * 4, kernel_size, padding, m)
        self.conv3c = Depthwise(width * 4, width * 4, kernel_size, padding, m)
        self.conv4 = Depthwise(width * 4, width * 4, kernel_size, padding, m)

        # self.fc = nn.Linear((width * 4) * 16, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.input_norm = nn.GroupNorm(in_channels, in_channels)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.pool(x)
        x = self.conv4(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class ModelForImageClassification(PreTrainedModel):
    def __init__(self, config: ModelConfig):
        super(ModelForImageClassification, self).__init__(config)
        self.config = config
        in_channels = config.in_channels
        width = config.width
        num_classes = config.num_classes
        kernel_size = config.kernel_size
        padding = config.padding
        m = config.m

        self.conv1 = Depthwise(in_channels, width, kernel_size, padding, m)
        self.conv2 = Depthwise(width, width * 2, kernel_size, padding, m)
        self.conv3a = Depthwise(width * 2, width * 4, kernel_size, padding, m)
        self.conv3b = Depthwise(width * 4, width * 4, kernel_size, padding, m)
        self.conv3c = Depthwise(width * 4, width * 4, kernel_size, padding, m)
        self.conv4 = Depthwise(width * 4, width * 4, kernel_size, padding, m)

        self.fc = nn.Linear((width * 4) * 16, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.input_norm = nn.GroupNorm(in_channels, in_channels)

    def forward(self, x, labels=None):
        x = self.input_norm(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.pool(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if labels is not None:
            loss = nn.functional.cross_entropy(x, labels, label_smoothing=0.1)
            return {"logits": x, "loss": loss}
        return {"logits": x}


class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, m):
        super(Depthwise, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise1 = nn.Conv2d(in_channels, out_channels * m, 1)
        self.pointwise2 = nn.Conv2d(out_channels * m, out_channels, 1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm = nn.GroupNorm(1, in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)

        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)

        x += self.skip(residual)
        return x


def main():
    from torchinfo import summary

    config = ModelConfig()
    model = Model(config)
    summary(model, (1, 1, 32, 32), depth=1)


if __name__ == "__main__":
    main()
