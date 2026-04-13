# Signature Origin Classifier — Human vs. Generative AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-80.21%25-brightgreen)
![Dataset Accuracy](https://img.shields.io/badge/Full%20Dataset%20Accuracy-85.70%25-brightgreen)

A PyTorch deep learning system for **4-class sequence classification** of handwritten signatures, distinguishing genuine human signatures from those synthesized by three generative architectures — GAN, SDT, and VAE. The model processes raw 2D pen-stroke coordinate sequences and addresses the growing challenge of **AI-generated biometric content attribution**.

**Achieves 80.21% test accuracy and 85.70% full-dataset accuracy** using a Bidirectional GRU trained on a stratified 70/15/15 split with early stopping and learning rate scheduling.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Classes & Label Encoding](#classes--label-encoding)
- [Model Architecture](#model-architecture)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Project Files](#project-files)
- [License](#license)

---

## Overview

Signature verification and origin attribution are critical tasks in biometric security. As generative models (GANs, VAEs, and kinematic synthesis models) become increasingly capable of producing realistic handwritten signatures, the ability to distinguish human-produced from machine-generated strokes becomes essential.

This project frames the problem as a **temporal sequence classification task**: each signature is represented as an ordered sequence of (X, Y) coordinate pairs captured during the signing process. A Bidirectional GRU is used to learn discriminative patterns across the full trajectory — both forward and backward — before classifying into one of four origin classes.

---

## Dataset Structure

```
signatures/
├── human/
│   ├── 001g01.csv
│   ├── 001g02.csv
│   └── ...
├── gan/
│   ├── 001g01.csv
│   ├── 001g02.csv
│   └── ...
├── sdt/
│   ├── 001g01.csv
│   ├── 001g02.csv
│   └── ...
└── vae/
    ├── 001g01.csv
    ├── 001g02.csv
    └── ...
```

Each `.csv` file contains space-separated `X` and `Y` columns representing the 2D coordinate sequence of a single signature stroke.

---

## Classes & Label Encoding

| Class   | Label | Description                          |
|---------|-------|--------------------------------------|
| `human` | 0     | Genuine human handwritten signature  |
| `gan`   | 1     | GAN-generated signature              |
| `sdt`   | 2     | SDT-generated signature              |
| `vae`   | 3     | VAE-generated signature              |

---

## Model Architecture

The classifier is a **Bidirectional GRU (BiGRU)** network designed to capture temporal dependencies in both directions across the pen-stroke sequence:

```
Input (X, Y) sequence — 150 timesteps × 2 features
        ↓
BiGRU — 2 stacked layers, 128 hidden units per direction
        ↓
Concatenate final forward + backward hidden states → 256-dim representation
        ↓
Dropout (p = 0.25)
        ↓
Linear classifier → 4 class logits
```

| Component     | Detail                                  |
|---------------|-----------------------------------------|
| RNN type      | Bidirectional GRU                       |
| Layers        | 2 stacked                               |
| Hidden units  | 128 per direction (256 combined)        |
| Dropout       | p = 0.25 (inter-layer + pre-classifier) |
| Output        | 4-class logits via Linear(256 → 4)      |

---

## Preprocessing Pipeline

Each signature CSV file is preprocessed as follows before being passed to the model:

1. **Load** space-separated X, Y columns; coerce non-numeric values and drop resulting `NaN` rows.
2. **Normalize** — apply per-signature Min-Max scaling independently to the X and Y axes (range [0, 1]).
3. **Pad or truncate** to a fixed length of 150 timesteps (zero-padding for shorter sequences).
4. **Degenerate sequences** — single-point sequences are zeroed out entirely as they carry no trajectory information.

> Per-signature normalization (rather than global) is used to preserve the relative shape of each trajectory while removing absolute position and scale biases.

---

## Setup & Installation

**1. Clone the repository:**

```bash
git clone https://github.com/emaadkalantarii/Signature-origin-classifier.git
cd Signature-origin-classifier
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv signature_env
source signature_env/bin/activate       # Linux / macOS
# signature_env\Scripts\activate        # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch>=2.4.0 numpy>=2.2.0 scikit-learn>=1.6 pandas>=2.2.3
```

---

## Usage

### Training

Place your dataset in `./signatures/` (following the directory structure above), or update `BASE_DATA_DIR` in `train.py`, then run:

```bash
python train.py
```

This will:
- Split the dataset (70% train / 15% validation / 15% test), stratified by class.
- Train the BiGRU model for up to 60 epochs with early stopping (patience = 15).
- Save the best checkpoint — lowest validation loss — to `model.pth`.
- Report final loss and accuracy on the held-out test set.

### Evaluation

```bash
python eval.py
```

When run directly, `eval.py` evaluates all `.csv` files found under `./signatures/` using the `model.pth` checkpoint and reports per-file predictions and overall accuracy.

To use custom paths, update the variables at the bottom of `eval.py`:

```python
eval_directory = "./signatures"
model_filepath  = "model.pth"
```

### Programmatic Use

The evaluation module exposes a clean API for integration:

```python
from eval import load_and_predict

predictions = load_and_predict("./signatures", "model.pth")
# Returns: { './signatures/human/001g01.csv': 0,
#             './signatures/gan/001g02.csv':   1, ... }
```

Label encoding: `human → 0`, `gan → 1`, `sdt → 2`, `vae → 3`

---

## Hyperparameters

| Parameter        | Value             | Description                                    |
|------------------|-------------------|------------------------------------------------|
| `MAX_SEQ_LENGTH` | 150               | Fixed sequence length (pad / truncate)         |
| `HIDDEN_SIZE`    | 128               | GRU hidden units per direction                 |
| `NUM_RNN_LAYERS` | 2                 | Number of stacked GRU layers                   |
| `DROPOUT_PROB`   | 0.25              | Dropout rate (inter-layer + pre-classifier)    |
| `LEARNING_RATE`  | 0.001             | Initial Adam learning rate                     |
| `WEIGHT_DECAY`   | 1e-4              | L2 regularization strength                     |
| `BATCH_SIZE`     | 32                | Training batch size                            |
| `NUM_EPOCHS`     | 60                | Maximum training epochs                        |
| `GRADIENT_CLIP`  | 1.0               | Gradient clipping threshold                    |
| `LR_SCHEDULER`   | ReduceLROnPlateau | Factor 0.5, patience 7 epochs                  |
| `EARLY_STOPPING` | 15                | Patience (epochs without val loss improvement) |

### Sequence Length Search

`MAX_SEQ_LENGTH = 150` was selected after evaluating four candidate values:

| MAX_SEQ_LENGTH | Val Loss | Val Accuracy | Note                              |
|----------------|----------|--------------|-----------------------------------|
| **150**        | **0.3348** | **80.21%** | ✓ Best loss · most efficient      |
| 200            | 0.4132   | 78.74%       | Worse on both metrics             |
| 400            | 0.3812   | 80.00%       | Comparable accuracy, higher cost  |
| 570            | 0.3520   | 81.89%       | Marginal gain, significantly higher cost |

---

## Results

| Metric                    | Value              |
|---------------------------|--------------------|
| Test Set Loss             | 0.3348             |
| **Test Set Accuracy**     | **80.21%**         |
| **Full Dataset Accuracy** | **85.70%**         |
| Best Validation Loss      | 0.3291 (Epoch 57)  |
| Best Validation Accuracy  | 82.24%             |
| Training Duration         | 60 epochs (full)   |

The learning rate was automatically reduced from 0.001 → 0.0005 at epoch 52 via `ReduceLROnPlateau`. The model continued to improve after the reduction and reached its best checkpoint at epoch 57.

---

## Project Files

| File              | Description                                                                   |
|-------------------|-------------------------------------------------------------------------------|
| `signatures/`     | Dataset directory: contains 4 subdirectories (human, gan, sdt, vae) with CSV files of signature coordinate sequences |
| `train.py`        | Training script: data loading, model definition, training loop, checkpointing |
| `eval.py`         | Evaluation script: loads a trained `.pth` checkpoint and predicts labels      |
| `requirements.txt`| Python dependencies                                                           |
| `README.md`       | This file                                                                     |

> **Note:** `model.pth` is not included in the repository. It is generated by running `train.py` on your dataset. The `signatures/` dataset directory is included.

---

## License

This project is licensed under the [MIT License](LICENSE).
