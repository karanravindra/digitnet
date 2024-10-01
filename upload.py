import os
from safetensors.torch import load_model
from src import ModelForImageClassification, ModelConfig

widths = [2, 4, 6, 8]

for w, folder in zip(widths, os.listdir("checkpoints")):
    if not os.path.isdir(f"checkpoints/{folder}"):
        continue

    model = ModelForImageClassification(ModelConfig(width=w))
    load_model(model, f"checkpoints/{folder}/model.safetensors")

    model.push_to_hub(repo_id=f"karanravindra/digitnet-{folder}", use_temp_dir=True)
