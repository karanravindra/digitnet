---
license: mit
pipeline_tag: image-classification
datasets:
  - ylecun/mnist
  - karanravindra/qmnist
language:
  - en
tags:
  - legal
  - finance
---

# digitnet

The digitnet family of models are a series of models trained on the QMNIST datasets. The models are trained to classify images of handwritten digits into one of ten classes, corresponding to the digits 0-9. The models are trained using the PyTorch framework.

## Model Details

### Model Description

- **Developed by:** Karan Ravindra
- **Model type:** Convolutional Neural Network
- **License:** MIT

### Model Sources

- **Repository:** [Github](https://github.com/karanravindra/digitnet)
  <!-- - **Paper [optional]:** [More Information Needed] -->
  <!-- - **Demo [optional]:** [More Information Needed] -->

## Uses

### Direct Use

These models can be used to classify images of handwritten digits into one of ten classes, corresponding to the digits 0-9.

### Downstream Use

These models can be used as a component in a larger machine learning system, such as a system for recognizing handwritten digits in a larger document, or to train a custom LPIPS model.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
TODO: Add implementation code
```

## Training Details

All training and implementation details can be found in the [Github repository](https://github.com/karanravindra/digitnet).

### Results

TODO: Add confusion matrix and other results

#### small

digitnet on  main [!?] is 📦 v0.1.0 via 🐍 v3.12.5 (digitnet)
❯ /Users/karan/projects/digitnet/.venv/bin/python /Users/karan/projects/digitnet/main.py
Epoch 1: 100%|| 235/235 [00:08<00:00, 27.81it/s, loss=1.08, acc=0.781, val_loss=inf, val_acc=0]
Epoch 2: 100%|| 235/235 [00:07<00:00, 31.23it/s, loss=0.823, acc=0.896, val_loss=1.11, val_acc=0.766]
Epoch 3: 100%|| 235/235 [00:06<00:00, 34.84it/s, loss=0.87, acc=0.896, val_loss=0.926, val_acc=0.849]
Epoch 4: 100%|| 235/235 [00:06<00:00, 34.08it/s, loss=0.902, acc=0.844, val_loss=0.85, val_acc=0.885]
Epoch 5: 100%|| 235/235 [00:06<00:00, 33.60it/s, loss=0.735, acc=0.938, val_loss=0.817, val_acc=0.894]
Epoch 6: 100%|| 235/235 [00:06<00:00, 34.03it/s, loss=0.907, acc=0.865, val_loss=0.794, val_acc=0.905]
Epoch 7: 100%|| 235/235 [00:06<00:00, 34.31it/s, loss=0.849, acc=0.865, val_loss=0.777, val_acc=0.913]
Epoch 8: 100%|| 235/235 [00:07<00:00, 33.19it/s, loss=0.802, acc=0.896, val_loss=0.759, val_acc=0.919]
Epoch 9: 100%|| 235/235 [00:06<00:00, 33.76it/s, loss=0.818, acc=0.896, val_loss=0.752, val_acc=0.92]
Epoch 10: 100%|| 235/235 [00:06<00:00, 34.10it/s, loss=0.809, acc=0.854, val_loss=0.747, val_acc=0.925]
Test loss: 0.7475, Test accuracy: 0.9216

#### medium

digitnet on  main [!?] is 📦 v0.1.0 via 🐍 v3.12.5 (digitnet)
❯ /Users/karan/projects/digitnet/.venv/bin/python /Users/karan/projects/digitnet/main.py
Epoch 1: 100%|| 235/235 [00:16<00:00, 14.65it/s, loss=0.987, acc=0.833, val_loss=inf, val_acc=0]
Epoch 2: 100%|| 235/235 [00:07<00:00, 33.28it/s, loss=0.971, acc=0.833, val_loss=0.987, val_acc=0.825]
Epoch 3: 100%|| 235/235 [00:06<00:00, 34.02it/s, loss=0.779, acc=0.917, val_loss=0.827, val_acc=0.901]
Epoch 4: 100%|| 235/235 [00:06<00:00, 34.30it/s, loss=0.798, acc=0.896, val_loss=0.771, val_acc=0.917]
Epoch 5: 100%|| 235/235 [00:06<00:00, 34.10it/s, loss=0.754, acc=0.927, val_loss=0.738, val_acc=0.928]
Epoch 6: 100%|| 235/235 [00:07<00:00, 32.83it/s, loss=0.689, acc=0.938, val_loss=0.721, val_acc=0.934]
Epoch 7: 100%|| 235/235 [00:06<00:00, 34.18it/s, loss=0.706, acc=0.948, val_loss=0.704, val_acc=0.942]
Epoch 8: 100%|| 235/235 [00:07<00:00, 31.19it/s, loss=0.683, acc=0.938, val_loss=0.684, val_acc=0.95]
Epoch 9: 100%|| 235/235 [00:06<00:00, 35.26it/s, loss=0.691, acc=0.948, val_loss=0.678, val_acc=0.95]
Epoch 10: 100%|| 235/235 [00:06<00:00, 35.97it/s, loss=0.67, acc=0.958, val_loss=0.669, val_acc=0.954]
Test loss: 0.6668, Test accuracy: 0.9549

#### lg

digitnet on  main [!?] is 📦 v0.1.0 via 🐍 v3.12.5 (digitnet) took 1m23s
❯ /Users/karan/projects/digitnet/.venv/bin/python /Users/karan/projects/digitnet/main.py
Epoch 1: 100%|| 235/235 [00:09<00:00, 25.65it/s, loss=0.883, acc=0.875, val_loss=inf, val_acc=0]
Epoch 2: 100%|| 235/235 [00:06<00:00, 33.72it/s, loss=0.764, acc=0.917, val_loss=0.843, val_acc=0.889]
Epoch 3: 100%|| 235/235 [00:06<00:00, 33.74it/s, loss=0.768, acc=0.896, val_loss=0.731, val_acc=0.935]
Epoch 4: 100%|| 235/235 [00:07<00:00, 33.54it/s, loss=0.684, acc=0.948, val_loss=0.688, val_acc=0.949]
Epoch 5: 100%|| 235/235 [00:06<00:00, 33.76it/s, loss=0.679, acc=0.927, val_loss=0.663, val_acc=0.956]
Epoch 6: 100%|| 235/235 [00:06<00:00, 33.74it/s, loss=0.617, acc=0.979, val_loss=0.647, val_acc=0.963]
Epoch 7: 100%|| 235/235 [00:06<00:00, 33.63it/s, loss=0.618, acc=0.969, val_loss=0.635, val_acc=0.966]
Epoch 8: 100%|| 235/235 [00:07<00:00, 33.45it/s, loss=0.616, acc=0.969, val_loss=0.636, val_acc=0.965]
Epoch 9: 100%|| 235/235 [00:07<00:00, 33.45it/s, loss=0.628, acc=0.969, val_loss=0.621, val_acc=0.971]
Epoch 10: 100%|| 235/235 [00:07<00:00, 33.35it/s, loss=0.652, acc=0.958, val_loss=0.618, val_acc=0.969]
Test loss: 0.6148, Test accuracy: 0.9690

#### xl

Epoch 1: 100%|| 235/235 [00:10<00:00, 22.77it/s, loss=0.849, acc=0.885, val_loss=inf, val_acc=0]
Epoch 2: 100%|| 235/235 [00:07<00:00, 33.35it/s, loss=0.727, acc=0.938, val_loss=0.828, val_acc=0.898]
Epoch 3: 100%|| 235/235 [00:07<00:00, 33.52it/s, loss=0.625, acc=1, val_loss=0.733, val_acc=0.932]
Epoch 4: 100%|| 235/235 [00:06<00:00, 33.70it/s, loss=0.651, acc=0.948, val_loss=0.69, val_acc=0.95]
Epoch 5: 100%|| 235/235 [00:06<00:00, 33.62it/s, loss=0.704, acc=0.917, val_loss=0.657, val_acc=0.959]
Epoch 6: 100%|| 235/235 [00:07<00:00, 33.52it/s, loss=0.598, acc=0.99, val_loss=0.636, val_acc=0.966]
Epoch 7: 100%|| 235/235 [00:07<00:00, 33.54it/s, loss=0.631, acc=0.99, val_loss=0.627, val_acc=0.965]
Epoch 8: 100%|| 235/235 [00:07<00:00, 33.48it/s, loss=0.609, acc=0.969, val_loss=0.624, val_acc=0.97]
Epoch 9: 100%|| 235/235 [00:07<00:00, 31.57it/s, loss=0.613, acc=0.99, val_loss=0.617, val_acc=0.971]
Epoch 10: 100%|| 235/235 [00:07<00:00, 32.88it/s, loss=0.61, acc=0.979, val_loss=0.608, val_acc=0.973]
Test loss: 0.6075, Test accuracy: 0.9727

## Environmental Impact

- **Hardware Type:** Apple M3 Pro
- **Hours used:** 20 minutes
- **Cloud Provider:** Private Server
- **Carbon Emitted:** Very Low

## Technical Specifications

### Model Architecture and Objective

[More Information Needed]

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
