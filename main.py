import os

import torch
import torchvision.transforms.v2 as T
from torchvision import datasets, utils
from tqdm import tqdm
from safetensors.torch import save_model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchinfo import summary

from src import ModelForImageClassification as Model, ModelConfig


device = torch.device("mps")


def main(width, name):
    os.makedirs(f"checkpoints/{name}", exist_ok=True)
    torch.manual_seed(42)
    torch.mps.manual_seed(42)

    ## Prepare the dataset
    # Define the transforms
    transforms = T.Compose(
        [
            T.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.75, 1.5),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.Resize((32, 32)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    # Load the dataset
    train_dataset = datasets.QMNIST(
        root="data",
        what="train",
        download=True,
        transform=transforms,
    )

    val_dataset = datasets.QMNIST(
        root="data",
        what="test10k",
        download=True,
        transform=transforms,
    )

    test_dataset = datasets.QMNIST(
        root="data",
        what="test",
        download=True,
        transform=transforms,
    )

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    x, y = next(iter(val_loader))
    utils.save_image(
        x[:64], "checkpoints/sample.png", nrow=8, normalize=False, pad_value=1
    )

    ## Create the model
    model = Model(ModelConfig(width=width))
    model.to(device)
    summary(model, (1, 1, 32, 32), depth=1, device=device)

    # Define the optimizer and the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=0.2
    )  # using label smoothing to regularize the model

    ## Train the model
    val_loss = float("inf")
    val_acc = 0

    for epoch in range(50):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred["logits"]
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": (y_pred.argmax(1) == y).float().mean().item(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                y_pred = y_pred["logits"]
                val_loss += criterion(y_pred, y).item()
                val_acc += (y_pred.argmax(1) == y).float().mean().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

    ## Evaluate the model
    test_loss = 0
    test_acc = 0
    test_outputs = []
    test_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred = y_pred["logits"]
            test_loss += criterion(y_pred, y).item()
            test_acc += (y_pred.argmax(1) == y).float().mean().item()
            test_outputs.append(y_pred)
            test_targets.append(y)
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # confusion matrix
    test_outputs = torch.cat(test_outputs)
    test_targets = torch.cat(test_targets)

    # zero out the diagonal
    zeros = torch.eye(10).bool().numpy()

    cm = confusion_matrix(test_targets.cpu(), test_outputs.argmax(1).cpu())
    cm[zeros] = 0
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot()
    disp.figure_.suptitle(f"Confusion Matrix (width={width})")
    disp.figure_.savefig(f"checkpoints/{name}/cm.png", dpi=300)

    ## Save the model and the optimizer state
    # save checkpoint
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        f"checkpoints/{name}/model.ckpt",
    )

    # save safetensors
    save_model(model, f"checkpoints/{name}/model.safetensors")

    # save onnx
    dummy_input = torch.randn(1, 1, 32, 32, device=device)
    torch.onnx.export(
        model, dummy_input, f"checkpoints/{name}/model.onnx", opset_version=12
    )


if __name__ == "__main__":
    print("Training new model: sm")
    main(2, "sm")  # sm
    print("\nTraining new model: md")
    main(4, "md")  # md
    print("\nTraining new model: lg")
    main(6, "lg")  # lg
    print("\nTraining new model: xl")
    main(8, "xl")  # xl
