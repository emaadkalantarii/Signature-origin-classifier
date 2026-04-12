import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration — must match train.py exactly
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 150
INPUT_FEATURES = 2
HIDDEN_SIZE = 128
NUM_RNN_LAYERS = 2
NUM_CLASSES = 4
DROPOUT_PROB = 0.25

CLASS_SUBDIRECTORIES_MAP = {"human": 0, "gan": 1, "sdt": 2, "vae": 3}

# ---------------------------------------------------------------------------
# Model — must be identical to the definition used in train.py
# ---------------------------------------------------------------------------

class SignatureModel(nn.Module):
    """
    Bidirectional 2-layer GRU classifier for signature sequences.

    Architecture:
        - BiGRU: 2 stacked layers, hidden_size units per direction
        - Dropout before the classification head
        - Linear head mapping to num_classes
    """

    def __init__(self, input_size: int, hidden_size: int, num_rnn_layers: int,
                 num_classes: int, dropout_prob: float):
        super().__init__()
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout_prob if num_rnn_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        out = torch.cat((h_forward, h_backward), dim=1)
        out = self.dropout(out)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return a CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        print("CUDA GPU detected — using GPU for inference.")
        return torch.device("cuda")
    print("No GPU found — using CPU.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint utility
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a model state dict from a .pth file, with compatibility fallback."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        print("Note: 'weights_only=True' not supported on this PyTorch version; falling back.")
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_signature(csv_path: str, max_seq_len: int, input_features: int) -> torch.Tensor:
    """
    Load a single signature CSV and return a preprocessed tensor.

    Steps:
        1. Read space-separated X, Y columns.
        2. Drop rows with non-numeric values.
        3. Normalize each axis independently with Min-Max scaling.
        4. Pad with zeros or truncate to max_seq_len.

    Returns:
        Tensor of shape (max_seq_len, input_features).
    """
    result = np.zeros((max_seq_len, input_features), dtype=np.float32)
    try:
        df = pd.read_csv(csv_path, sep=" ", header=0, names=["X", "Y"], engine="python")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df.dropna(inplace=True)

        if df.empty:
            return torch.tensor(result, dtype=torch.float32)

        coords = df[["X", "Y"]].values.astype(np.float32)

        if coords.shape[0] > 1:
            coords[:, 0] = MinMaxScaler().fit_transform(coords[:, [0]]).flatten()
            coords[:, 1] = MinMaxScaler().fit_transform(coords[:, [1]]).flatten()
        else:
            coords[:, :] = 0.0

        copy_len = min(len(coords), max_seq_len)
        result[:copy_len, :] = coords[:copy_len, :]

    except Exception as e:
        print(f"Warning: Could not process '{os.path.basename(csv_path)}': {e}. Using zero sequence.")

    return torch.tensor(result, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def load_and_predict(directory: str, model_file: str) -> dict:
    """
    Run inference on all CSV files found in a directory tree.

    The directory is expected to follow the same structure as the training dataset:
        /path/to/signatures/
            human/  001g01.csv  001g02.csv  ...
            gan/    001g01.csv  001g02.csv  ...
            sdt/    001g01.csv  001g02.csv  ...
            vae/    001g01.csv  001g02.csv  ...

    Args:
        directory:  Root path to search for .csv files (searched recursively).
        model_file: Path to a trained model checkpoint (.pth file).

    Returns:
        A dict mapping absolute CSV file paths to predicted integer labels:
            { '/abs/path/to/human/001g01.csv': 0,
              '/abs/path/to/gan/001g01.csv':   1, ... }

        Label encoding: human → 0, gan → 1, sdt → 2, vae → 3
    """
    device = get_device()

    # --- Load model ---
    model = SignatureModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_RNN_LAYERS, NUM_CLASSES, DROPOUT_PROB)
    try:
        state_dict = load_checkpoint(model_file, device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model file '{model_file}' not found.")
        return {}
    except Exception as e:
        print(f"Error loading model checkpoint '{model_file}': {e}")
        return {}

    model = model.to(device)
    model.eval()

    # --- Collect CSV files ---
    csv_files = sorted(set(
        os.path.abspath(p)
        for p in glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    ))

    if not csv_files:
        print(f"Warning: No .csv files found in '{directory}' or its subdirectories.")
        return {}

    # --- Batch inference ---
    EVAL_BATCH_SIZE = 32
    labels_dict = {}

    with torch.no_grad():
        for batch_start in range(0, len(csv_files), EVAL_BATCH_SIZE):
            batch_paths = csv_files[batch_start: batch_start + EVAL_BATCH_SIZE]

            tensors, valid_paths = [], []
            for path in batch_paths:
                try:
                    tensors.append(preprocess_signature(path, MAX_SEQ_LENGTH, INPUT_FEATURES))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Warning: Skipping '{os.path.basename(path)}': {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(device)
            logits = model(batch_tensor)
            predictions = logits.argmax(dim=1)

            for path, label in zip(valid_paths, predictions.tolist()):
                labels_dict[path] = label

    return labels_dict


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eval_directory = "./signatures"
    model_filepath = "model.pth"

    predictions = load_and_predict(eval_directory, model_filepath)

    if not predictions:
        print("No predictions were made.")
    else:
        print(f"\nPredictions ({len(predictions)} files):")
        for file_path, predicted_label in predictions.items():
            try:
                display_path = os.path.relpath(file_path)
            except ValueError:
                display_path = file_path
            print(f"  '{display_path}': {predicted_label}")

        # --- Accuracy against ground-truth labels inferred from directory names ---
        correct, total = 0, 0
        for file_path, predicted_label in predictions.items():
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if parent_dir in CLASS_SUBDIRECTORIES_MAP:
                true_label = CLASS_SUBDIRECTORIES_MAP[parent_dir]
                if predicted_label == true_label:
                    correct += 1
                total += 1

        if total > 0:
            print(f"\nTotal evaluated: {total}")
            print(f"Correct:         {correct}")
            print(f"Accuracy:        {correct / total:.4f} ({100.0 * correct / total:.2f}%)")
        else:
            print("\nNo files with recognizable class subdirectory names found for accuracy calculation.")
