"""
Crop Classification Project - Step 4: PyTorch Deep Learning Models
===================================================================
Trains 4 deep learning architectures using PyTorch:
  1. SpectralMLP      - Deep Multi-Layer Perceptron
  2. SpectralCNN1D    - 1D Convolutional Network
  3. SpectralHybrid   - CNN + MLP Combined
  4. SpectralAttention - Self-Attention / Transformer-style

Uses FocalLoss, AdamW, ReduceLROnPlateau, and early stopping.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import joblib
import math
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, cohen_kappa_score,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\Deep_Learning2026-20260129T154240Z-3-001\Deep_Learning2026")
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = DATA_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {0: 'Cotton', 1: 'Wheat', 2: 'Fallow', 3: 'Grass', 4: 'Water'}
CLASS_COLORS = {0: '#FF8C00', 1: '#FFD700', 2: '#8B4513', 3: '#32CD32', 4: '#0000FF'}
NUM_CLASSES = 5

BATCH_SIZE = 256
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("CROP CLASSIFICATION - PYTORCH DEEP LEARNING MODELS")
print("=" * 70)
print(f"    Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv(DATA_DIR / "crop_training_data_5classes_2020.csv")
print(f"    Loaded {len(df):,} samples")

feature_cols = [col for col in df.columns if col not in ['class', 'classname', 'system:index', '.geo']]
print(f"    Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.int64)

NUM_FEATURES = X.shape[1]
print(f"    X shape: {X.shape}, y shape: {y.shape}")

# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================
print("\n[2] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Training set: {X_train.shape[0]:,} samples")
print(f"    Test set: {X_test.shape[0]:,} samples")

for cls in range(NUM_CLASSES):
    train_count = np.sum(y_train == cls)
    test_count = np.sum(y_test == cls)
    print(f"    {CLASS_NAMES[cls]:<10}: Train={train_count:>5,} | Test={test_count:>4,}")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================
print("\n[3] Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / 'pytorch_scaler.joblib')
joblib.dump(feature_cols, MODEL_DIR / 'pytorch_feature_cols.joblib')
print(f"    Scaler saved to: {MODEL_DIR / 'pytorch_scaler.joblib'}")

# ============================================================================
# 4. CLASS WEIGHTS & SMOTE
# ============================================================================
print("\n[4] Computing class weights and applying SMOTE...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

print("    Class weights:")
for cls, w in enumerate(class_weights):
    print(f"    {CLASS_NAMES[cls]:<10}: {w:.4f}")

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"\n    Original training samples: {len(y_train):,}")
print(f"    After SMOTE: {len(y_train_smote):,}")

# ============================================================================
# 5. PYTORCH DATASETS & DATALOADERS
# ============================================================================
print("\n[5] Creating PyTorch DataLoaders...")

X_train_tensor = torch.FloatTensor(X_train_smote)
y_train_tensor = torch.LongTensor(y_train_smote)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Split training into train/val (85/15)
n_val = int(len(X_train_tensor) * 0.15)
n_train = len(X_train_tensor) - n_val
indices = torch.randperm(len(X_train_tensor), generator=torch.Generator().manual_seed(42))

train_idx = indices[:n_train]
val_idx = indices[n_train:]

train_dataset = TensorDataset(X_train_tensor[train_idx], y_train_tensor[train_idx])
val_dataset = TensorDataset(X_train_tensor[val_idx], y_train_tensor[val_idx])
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"    Train: {n_train:,} | Val: {n_val:,} | Test: {len(X_test_tensor):,}")
print(f"    Batch size: {BATCH_SIZE}")

# ============================================================================
# 6. FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance. Reduces loss for well-classified
    examples, focusing training on hard examples."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weight tensor

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# 7. MODEL ARCHITECTURES
# ============================================================================

class SpectralMLP(nn.Module):
    """Deep Multi-Layer Perceptron for spectral classification.
    Architecture: 24 -> 128 -> 256 -> 128 -> 64 -> 5"""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class SpectralCNN1D(nn.Module):
    """1D Convolutional Network treating spectral bands as a 1D signal.
    Conv1D layers: 1->32->64->128 with adaptive pooling."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


class SpectralHybrid(nn.Module):
    """Hybrid CNN + MLP model. CNN branch extracts local spectral patterns,
    MLP branch processes global features. Both are fused for classification."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        # CNN branch
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # MLP branch
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn_branch(x.unsqueeze(1)).squeeze(-1)  # (batch, 64)
        mlp_out = self.mlp_branch(x)  # (batch, 64)
        combined = torch.cat([cnn_out, mlp_out], dim=1)  # (batch, 128)
        return self.fusion(combined)


class SpectralAttention(nn.Module):
    """Self-Attention / Transformer-style model for spectral data.
    Uses learnable positional encoding, CLS token, and 2-layer
    Transformer encoder with 4 attention heads."""

    def __init__(self, n_features, n_classes, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model

        # Project each band to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)

        # Learnable positional encoding for n_features bands
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head from CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # x: (batch, n_features) -> (batch, n_features, 1) -> (batch, n_features, d_model)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+n_features, d_model)

        # Transformer encoder
        x = self.transformer(x)

        # Use CLS token output for classification
        cls_output = x[:, 0]
        return self.classifier(cls_output)


# ============================================================================
# 8. TRAINING FUNCTION
# ============================================================================
def train_model(model, model_name, train_loader, val_loader, class_weight_tensor):
    """Train a PyTorch model with FocalLoss, AdamW, scheduler, and early stopping."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {n_params:,}")

    criterion = FocalLoss(alpha=class_weight_tensor, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=False)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_state = None

    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.1e} | "
                  f"{'* best' if patience_counter == 0 else ''}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
            break

    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.1f}s | Best epoch: {best_epoch+1} | Best val loss: {best_val_loss:.4f}")

    # Load best weights
    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    return model, train_losses, val_losses


# ============================================================================
# 9. EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, test_loader, model_name):
    """Evaluate a PyTorch model on the test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_w = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n    {'='*50}")
    print(f"    RESULTS: {model_name}")
    print(f"    {'='*50}")
    print(f"    Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precision: {prec:.4f} (macro)")
    print(f"    Recall:    {rec:.4f} (macro)")
    print(f"    F1-Score:  {f1:.4f} (macro) | {f1_w:.4f} (weighted)")
    print(f"    Kappa:     {kappa:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]))

    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
        'f1_weighted': f1_w,
        'kappa': kappa,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


# ============================================================================
# 10. TRAIN ALL 4 MODELS
# ============================================================================
print("\n" + "=" * 70)
print("[10] TRAINING 4 DEEP LEARNING MODELS")
print("=" * 70)

architectures = {
    'SpectralMLP': SpectralMLP(NUM_FEATURES, NUM_CLASSES),
    'SpectralCNN1D': SpectralCNN1D(NUM_FEATURES, NUM_CLASSES),
    'SpectralHybrid': SpectralHybrid(NUM_FEATURES, NUM_CLASSES),
    'SpectralAttention': SpectralAttention(NUM_FEATURES, NUM_CLASSES),
}

trained_models = {}
all_results = {}
all_train_losses = {}
all_val_losses = {}

for name, model in architectures.items():
    model, train_losses, val_losses = train_model(
        model, name, train_loader, val_loader, class_weight_tensor
    )
    trained_models[name] = model
    all_train_losses[name] = train_losses
    all_val_losses[name] = val_losses

    result = evaluate_model(model, test_loader, name)
    all_results[name] = result

    # Save checkpoint
    checkpoint_path = MODEL_DIR / f'pytorch_{name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': name,
        'n_features': NUM_FEATURES,
        'n_classes': NUM_CLASSES,
        'feature_cols': feature_cols,
    }, checkpoint_path)
    print(f"    Saved checkpoint: {checkpoint_path.name}")

# ============================================================================
# 11. MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("[11] MODEL COMPARISON")
print("=" * 70)

comparison_data = []
for name, r in all_results.items():
    comparison_data.append({
        'Model': r['model_name'],
        'Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision_macro']:.4f}",
        'Recall': f"{r['recall_macro']:.4f}",
        'F1 (Macro)': f"{r['f1_macro']:.4f}",
        'F1 (Weighted)': f"{r['f1_weighted']:.4f}",
        'Kappa': f"{r['kappa']:.4f}",
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

best_model_key = max(all_results, key=lambda k: all_results[k]['f1_macro'])
print(f"\n    Best model (by Macro F1): {best_model_key}")

comparison_df.to_csv(OUTPUT_DIR / 'pytorch_model_comparison.csv', index=False)

# ============================================================================
# 12. PLOT TRAINING CURVES (all 4 models)
# ============================================================================
print("\n[12] Plotting training/validation loss curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, _) in enumerate(architectures.items()):
    ax = axes[idx]
    epochs_range = range(1, len(all_train_losses[name]) + 1)
    ax.plot(epochs_range, all_train_losses[name], label='Train Loss', linewidth=2)
    ax.plot(epochs_range, all_val_losses[name], label='Val Loss', linewidth=2, linestyle='--')
    ax.set_title(f'{name}\nF1={all_results[name]["f1_macro"]:.4f} | Acc={all_results[name]["accuracy"]:.2%}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('PyTorch Models - Training & Validation Loss', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pytorch_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'pytorch_training_curves.png'}")

# ============================================================================
# 13. PLOT CONFUSION MATRICES (all 4 models)
# ============================================================================
print("\n[13] Plotting confusion matrices...")

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for idx, (name, result) in enumerate(all_results.items()):
    cm = result['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = axes[idx].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text = f"{cm_norm[i, j]:.1%}\n({cm[i, j]})"
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            axes[idx].text(j, i, text, ha='center', va='center', fontsize=7, color=color)

    axes[idx].set_xticks(range(NUM_CLASSES))
    axes[idx].set_yticks(range(NUM_CLASSES))
    axes[idx].set_xticklabels([CLASS_NAMES[i] for i in range(NUM_CLASSES)], rotation=45, ha='right')
    axes[idx].set_yticklabels([CLASS_NAMES[i] for i in range(NUM_CLASSES)])
    axes[idx].set_title(f'{name}\nAcc: {result["accuracy"]:.2%} | K: {result["kappa"]:.3f}',
                        fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    if idx == 0:
        axes[idx].set_ylabel('Actual')

plt.suptitle('PyTorch Models - Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pytorch_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'pytorch_confusion_matrices.png'}")

# ============================================================================
# 14. SAVE BEST MODEL INFO
# ============================================================================
print("\n[14] Saving best model info...")

best_result = all_results[best_model_key]
pytorch_model_info = {
    'best_model': best_model_key,
    'n_features': NUM_FEATURES,
    'n_classes': NUM_CLASSES,
    'feature_cols': feature_cols,
    'class_names': CLASS_NAMES,
    'class_colors': CLASS_COLORS,
    'best_f1': best_result['f1_macro'],
    'best_accuracy': best_result['accuracy'],
}
joblib.dump(pytorch_model_info, MODEL_DIR / 'pytorch_model_info.joblib')
print(f"    Saved: pytorch_model_info.joblib")

# ============================================================================
# 15. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PYTORCH TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
    Device: {DEVICE}
    Models trained:
    ---------------""")

for name, r in all_results.items():
    marker = " <-- BEST" if name == best_model_key else ""
    print(f"    {name:<20} Acc: {r['accuracy']:.2%} | F1: {r['f1_macro']:.4f} | K: {r['kappa']:.4f}{marker}")

print(f"""
    Best model: {best_model_key} (F1={best_result['f1_macro']:.4f})

    Output files:
    -------------
    Checkpoints:  models/pytorch_SpectralMLP.pth
                  models/pytorch_SpectralCNN1D.pth
                  models/pytorch_SpectralHybrid.pth
                  models/pytorch_SpectralAttention.pth
    Scaler:       models/pytorch_scaler.joblib
    Features:     models/pytorch_feature_cols.joblib
    Info:         models/pytorch_model_info.joblib
    Plots:        outputs/pytorch_training_curves.png
                  outputs/pytorch_confusion_matrices.png
    CSV:          outputs/pytorch_model_comparison.csv

    Next step:
    ----------
    Run 05_apply_pytorch_to_image.py to classify the full satellite image
""")

print("=" * 70)
