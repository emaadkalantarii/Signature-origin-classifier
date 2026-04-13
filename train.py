import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DATA_DIR = "./signatures"
MODEL_SAVE_PATH = "model.pth"

LABELS_MAP = {"human": 0, "gan": 1, "sdt": 2, "vae": 3}
CLASS_NAMES = {v: k for k, v in LABELS_MAP.items()}

# Sequence & model hyperparameters
MAX_SEQ_LENGTH = 150   # Best found after tuning (see README for details)
INPUT_FEATURES = 2     # X and Y coordinates
HIDDEN_SIZE = 128      # GRU hidden units per direction
NUM_RNN_LAYERS = 2     # Number of stacked GRU layers
NUM_CLASSES = 4
DROPOUT_PROB = 0.25

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_VALUE = 1.0

# Data split ratios: 70% train / 15% val / 15% test
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

NUM_DATALOADER_WORKERS = 2  # Used when CUDA is available
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SignatureDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing signature CSV files.

    Each CSV file contains space-separated X and Y coordinate columns.
    Sequences are normalized per-signature with Min-Max scaling, then
    padded with zeros or truncated to MAX_SEQ_LENGTH.
    """

    def __init__(self, file_paths: list, labels: list, max_seq_len: int, input_features: int):
        self.file_paths = file_paths
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.input_features = input_features

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        sequence = self._load_and_preprocess(self.file_paths[idx])
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _load_and_preprocess(self, csv_path: str) -> np.ndarray:
        """Load a CSV file, normalize coordinates, and pad/truncate to fixed length."""
        result = np.zeros((self.max_seq_len, self.input_features), dtype=np.float32)
        try:
            df = pd.read_csv(csv_path, sep=" ", header=0, names=["X", "Y"], engine="python")
            df["X"] = pd.to_numeric(df["X"], errors="coerce")
            df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
            df.dropna(inplace=True)

            if df.empty:
                return result

            coords = df[["X", "Y"]].values.astype(np.float32)

            if coords.shape[0] > 1:
                coords[:, 0] = MinMaxScaler().fit_transform(coords[:, [0]]).flatten()
                coords[:, 1] = MinMaxScaler().fit_transform(coords[:, [1]]).flatten()
            else:
                # Single-point sequences are uninformative; zero them out
                coords[:, :] = 0.0

            copy_len = min(len(coords), self.max_seq_len)
            result[:copy_len, :] = coords[:copy_len, :]

        except Exception as e:
            print(f"Warning: Could not load '{csv_path}': {e}. Using zero sequence.")

        return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str, labels_map: dict) -> tuple:
    """
    Scan subdirectories for CSV files and collect file paths with integer labels.

    Returns:
        file_paths: list of absolute paths to CSV files
        labels: list of corresponding integer class labels
    """
    file_paths, labels = [], []
    print(f"Loading data from: {data_dir}")

    for class_name, label_idx in labels_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: Directory not found for class '{class_name}': {class_dir}")
            continue
        class_files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.endswith(".csv")
        ]
        file_paths.extend(class_files)
        labels.extend([label_idx] * len(class_files))

    if not file_paths:
        raise FileNotFoundError(f"No .csv files found in subdirectories of '{data_dir}'.")

    print(f"Total files found: {len(file_paths)}")
    for label_idx, class_name in CLASS_NAMES.items():
        print(f"  Class '{class_name}' (label {label_idx}): {labels.count(label_idx)} files")

    return file_paths, labels


# ---------------------------------------------------------------------------
# Model
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
        # h_n shape: (num_layers * 2, batch, hidden_size)
        _, h_n = self.rnn(x)
        # Concatenate last forward and last backward hidden states
        h_forward = h_n[-2]   # shape: (batch, hidden_size)
        h_backward = h_n[-1]  # shape: (batch, hidden_size)
        out = torch.cat((h_forward, h_backward), dim=1)
        out = self.dropout(out)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return a CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA GPU detected: {gpu_name} ({gpu_mem:.1f} GB) — using GPU.")
        return device
    print("No CUDA GPU found — using CPU.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a model state dict from a .pth file, with compatibility fallback."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        print("Note: 'weights_only=True' not supported on this PyTorch version; falling back.")
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> tuple:
    """Evaluate the model and return (average_loss, accuracy_percent)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset) if loader.dataset else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = get_device()
    print(f"Using device: {device}\n")

    # --- Data loading & splitting ---
    file_paths, labels = load_data(BASE_DATA_DIR, LABELS_MAP)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    relative_val_size = VALIDATION_RATIO / (1.0 - TRAIN_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    print(f"\nSplit sizes — Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}\n")

    # --- DataLoaders ---
    workers = NUM_DATALOADER_WORKERS if device.type == "cuda" else 0
    pin = device.type == "cuda"

    def make_loader(paths, lbls, shuffle):
        ds = SignatureDataset(paths, lbls, MAX_SEQ_LENGTH, INPUT_FEATURES)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=workers, pin_memory=pin)

    train_loader = make_loader(train_paths, train_labels, shuffle=True)
    val_loader = make_loader(val_paths, val_labels, shuffle=False)
    test_loader = make_loader(test_paths, test_labels, shuffle=False)

    # --- Model, loss, optimizer, scheduler ---
    model = SignatureModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_RNN_LAYERS, NUM_CLASSES, DROPOUT_PROB).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    # --- Training loop with early stopping ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stopping_patience = 15

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr < prev_lr:
            print(f"  → Learning rate reduced: {prev_lr:.6f} → {curr_lr:.6f}")

        print(
            f"Epoch {epoch:>3}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {curr_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping: no improvement for {early_stopping_patience} consecutive epochs.")
                break

    print("\nTraining complete.")

    # --- Final evaluation on test set ---
    try:
        model.load_state_dict(load_checkpoint(MODEL_SAVE_PATH, device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint '{MODEL_SAVE_PATH}' not found. Cannot evaluate.")
        return

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Set Results — Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
