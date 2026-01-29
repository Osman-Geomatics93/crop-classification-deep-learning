"""
Crop Classification Project - Step 2: Preprocessing and Model Training
=======================================================================
Uses scikit-learn MLPClassifier (neural network) and XGBoost.
All packages already available in the geodl environment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import joblib
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, cohen_kappa_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# SMOTE for oversampling
from imblearn.over_sampling import SMOTE

# XGBoost
from xgboost import XGBClassifier

# For reproducibility
np.random.seed(42)

# Set style
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

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("=" * 70)
print("CROP CLASSIFICATION - PREPROCESSING AND MODEL TRAINING")
print("=" * 70)

print("\n[1] Loading data...")
df = pd.read_csv(DATA_DIR / "crop_training_data_5classes_2020.csv")
print(f"    Loaded {len(df):,} samples")

feature_cols = [col for col in df.columns if col not in ['class', 'classname', 'system:index', '.geo']]
print(f"    Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.int64)

print(f"    X shape: {X.shape}")
print(f"    y shape: {y.shape}")

# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================
print("\n[2] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set: {X_train.shape[0]:,} samples")
print(f"    Test set: {X_test.shape[0]:,} samples")

print("\n    Class distribution in splits:")
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

joblib.dump(scaler, MODEL_DIR / 'feature_scaler.joblib')
joblib.dump(feature_cols, MODEL_DIR / 'feature_cols.joblib')
print(f"    Scaler saved to: {MODEL_DIR / 'feature_scaler.joblib'}")

# ============================================================================
# 4. CLASS WEIGHTS
# ============================================================================
print("\n[4] Computing class weights...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print("    Class weights:")
for cls, weight in class_weight_dict.items():
    print(f"    {CLASS_NAMES[cls]:<10}: {weight:.4f}")

# ============================================================================
# 5. SMOTE OVERSAMPLING
# ============================================================================
print("\n[5] Applying SMOTE oversampling...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"    Original training samples: {len(y_train):,}")
print(f"    After SMOTE: {len(y_train_smote):,}")
print("\n    Class distribution after SMOTE:")
for cls in range(NUM_CLASSES):
    count = np.sum(y_train_smote == cls)
    print(f"    {CLASS_NAMES[cls]:<10}: {count:,}")

# ============================================================================
# 6. EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print('='*60)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"\n    Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Macro F1-Score:   {f1_macro:.4f}")
    print(f"    Weighted F1-Score:{f1_weighted:.4f}")
    print(f"    Cohen's Kappa:    {kappa:.4f}")

    print("\n    Classification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred,
                                   target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)])
    print(report)

    cm = confusion_matrix(y_test, y_pred)

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

# ============================================================================
# 7. MODEL 1: MLP NEURAL NETWORK (with class weights)
# ============================================================================
print("\n" + "=" * 70)
print("[7] TRAINING MLP NEURAL NETWORK (with class weights)")
print("=" * 70)

# Compute sample weights from class weights for MLP
sample_weights_train = np.array([class_weight_dict[c] for c in y_train])

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,           # L2 regularization
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

print("\n    Architecture: 24 -> 256 -> 128 -> 64 -> 32 -> 5")
print("    Training with class-weighted samples...")
start_time = time.time()
mlp_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
mlp_time = time.time() - start_time
print(f"    Training completed in {mlp_time:.1f}s")
print(f"    Epochs trained: {mlp_model.n_iter_}")

results = {}
results['MLP'] = evaluate_model(mlp_model, X_test_scaled, y_test, "MLP Neural Network")

# ============================================================================
# 8. MODEL 2: MLP WITH SMOTE DATA (deeper network)
# ============================================================================
print("\n" + "=" * 70)
print("[8] TRAINING MLP WITH SMOTE DATA (deeper network)")
print("=" * 70)

mlp_smote_model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

print("\n    Architecture: 24 -> 512 -> 256 -> 128 -> 64 -> 32 -> 5")
print("    Training with SMOTE-balanced data...")
start_time = time.time()
mlp_smote_model.fit(X_train_smote, y_train_smote)
mlp_smote_time = time.time() - start_time
print(f"    Training completed in {mlp_smote_time:.1f}s")
print(f"    Epochs trained: {mlp_smote_model.n_iter_}")

results['MLP_SMOTE'] = evaluate_model(mlp_smote_model, X_test_scaled, y_test, "MLP + SMOTE")

# ============================================================================
# 9. MODEL 3: XGBOOST (with class weights)
# ============================================================================
print("\n" + "=" * 70)
print("[9] TRAINING XGBOOST (with class weights)")
print("=" * 70)

# Convert class weights to sample weights for XGBoost
sample_weights_xgb = np.array([class_weight_dict[c] for c in y_train])

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("\n    Training XGBoost with 500 trees...")
start_time = time.time()

# Split a validation set for early stopping
X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb, sw_tr, sw_val = train_test_split(
    X_train_scaled, y_train, sample_weights_xgb,
    test_size=0.15, random_state=42, stratify=y_train
)

xgb_model.fit(
    X_tr_xgb, y_tr_xgb,
    sample_weight=sw_tr,
    eval_set=[(X_val_xgb, y_val_xgb)],
    verbose=50
)

xgb_time = time.time() - start_time
print(f"    Training completed in {xgb_time:.1f}s")

results['XGBoost'] = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")

# ============================================================================
# 10. MODEL 4: XGBOOST WITH SMOTE
# ============================================================================
print("\n" + "=" * 70)
print("[10] TRAINING XGBOOST WITH SMOTE DATA")
print("=" * 70)

xgb_smote_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("\n    Training XGBoost with SMOTE data...")
start_time = time.time()

X_tr_xgb_s, X_val_xgb_s, y_tr_xgb_s, y_val_xgb_s = train_test_split(
    X_train_smote, y_train_smote,
    test_size=0.15, random_state=42, stratify=y_train_smote
)

xgb_smote_model.fit(
    X_tr_xgb_s, y_tr_xgb_s,
    eval_set=[(X_val_xgb_s, y_val_xgb_s)],
    verbose=50
)

xgb_smote_time = time.time() - start_time
print(f"    Training completed in {xgb_smote_time:.1f}s")

results['XGB_SMOTE'] = evaluate_model(xgb_smote_model, X_test_scaled, y_test, "XGBoost + SMOTE")

# ============================================================================
# 11. MODEL 5: RANDOM FOREST (baseline comparison)
# ============================================================================
print("\n" + "=" * 70)
print("[11] TRAINING RANDOM FOREST (baseline)")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n    Training Random Forest with 500 trees...")
start_time = time.time()
rf_model.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time
print(f"    Training completed in {rf_time:.1f}s")

results['RF'] = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# ============================================================================
# 12. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("[12] MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': [r['model_name'] for r in results.values()],
    'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
    'Macro F1': [f"{r['f1_macro']:.4f}" for r in results.values()],
    'Weighted F1': [f"{r['f1_weighted']:.4f}" for r in results.values()],
    'Kappa': [f"{r['kappa']:.4f}" for r in results.values()]
})

print("\n", comparison_df.to_string(index=False))

best_model_key = max(results, key=lambda k: results[k]['f1_macro'])
print(f"\n    Best model (by Macro F1): {results[best_model_key]['model_name']}")

# ============================================================================
# 13. PLOT MLP TRAINING CURVES
# ============================================================================
print("\n[13] Plotting MLP training curves...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MLP with class weights - loss curve
axes[0].plot(mlp_model.loss_curve_, label='Train', linewidth=2)
if hasattr(mlp_model, 'validation_scores_'):
    # Validation scores are accuracy, not loss - plot on secondary axis
    ax2 = axes[0].twinx()
    ax2.plot(mlp_model.validation_scores_, color='orange', label='Val Accuracy', linewidth=2)
    ax2.set_ylabel('Validation Accuracy', color='orange')
    ax2.legend(loc='center right')
axes[0].set_title('MLP (Class Weights) - Training Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# MLP with SMOTE - loss curve
axes[1].plot(mlp_smote_model.loss_curve_, label='Train', linewidth=2)
if hasattr(mlp_smote_model, 'validation_scores_'):
    ax2 = axes[1].twinx()
    ax2.plot(mlp_smote_model.validation_scores_, color='orange', label='Val Accuracy', linewidth=2)
    ax2.set_ylabel('Validation Accuracy', color='orange')
    ax2.legend(loc='center right')
axes[1].set_title('MLP (SMOTE) - Training Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'training_history.png'}")

# ============================================================================
# 14. PLOT CONFUSION MATRICES (all models)
# ============================================================================
print("\n[14] Plotting confusion matrices...")

n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

for idx, (name, result) in enumerate(results.items()):
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
    axes[idx].set_title(f'{result["model_name"]}\nAcc: {result["accuracy"]:.2%} | K: {result["kappa"]:.3f}',
                        fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    if idx == 0:
        axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")

# ============================================================================
# 15. FEATURE IMPORTANCE (XGBoost)
# ============================================================================
print("\n[15] Plotting XGBoost feature importance...")

fig, ax = plt.subplots(figsize=(10, 8))

# Get feature importance from best XGBoost model
best_xgb = xgb_model if results['XGBoost']['f1_macro'] >= results['XGB_SMOTE']['f1_macro'] else xgb_smote_model
importances = best_xgb.feature_importances_
indices = np.argsort(importances)

ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_cols[i] for i in indices])
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'feature_importance.png'}")

# ============================================================================
# 16. PER-CLASS ACCURACY COMPARISON
# ============================================================================
print("\n[16] Per-class accuracy comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(NUM_CLASSES)
width = 0.15
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

for idx, (name, result) in enumerate(results.items()):
    cm = result['confusion_matrix']
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bars = ax.bar(x + idx * width, per_class_acc, width, label=result['model_name'],
                  color=colors[idx], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Crop Class', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels([CLASS_NAMES[i] for i in range(NUM_CLASSES)])
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% target')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'per_class_accuracy.png'}")

# ============================================================================
# 17. SAVE MODELS
# ============================================================================
print("\n[17] Saving all models...")

joblib.dump(mlp_model, MODEL_DIR / 'crop_classifier_mlp.joblib')
joblib.dump(mlp_smote_model, MODEL_DIR / 'crop_classifier_mlp_smote.joblib')
joblib.dump(xgb_model, MODEL_DIR / 'crop_classifier_xgb.joblib')
joblib.dump(xgb_smote_model, MODEL_DIR / 'crop_classifier_xgb_smote.joblib')
joblib.dump(rf_model, MODEL_DIR / 'crop_classifier_rf.joblib')

# Save model info
model_info = {
    'input_dim': X_train.shape[1],
    'num_classes': NUM_CLASSES,
    'feature_cols': feature_cols,
    'class_names': CLASS_NAMES,
    'class_colors': CLASS_COLORS,
    'best_model': results[best_model_key]['model_name'],
    'best_model_key': best_model_key,
}
joblib.dump(model_info, MODEL_DIR / 'model_info.joblib')

comparison_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)

print(f"    Saved: crop_classifier_mlp.joblib")
print(f"    Saved: crop_classifier_mlp_smote.joblib")
print(f"    Saved: crop_classifier_xgb.joblib")
print(f"    Saved: crop_classifier_xgb_smote.joblib")
print(f"    Saved: crop_classifier_rf.joblib")
print(f"    Saved: model_info.joblib")
print(f"    Saved: model_comparison.csv")

# ============================================================================
# 18. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
    Models trained and saved:
    -------------------------
    1. MLP Neural Network (class weights)  -> Acc: {results['MLP']['accuracy']:.2%} | F1: {results['MLP']['f1_macro']:.4f}
    2. MLP Neural Network (SMOTE)          -> Acc: {results['MLP_SMOTE']['accuracy']:.2%} | F1: {results['MLP_SMOTE']['f1_macro']:.4f}
    3. XGBoost (class weights)             -> Acc: {results['XGBoost']['accuracy']:.2%} | F1: {results['XGBoost']['f1_macro']:.4f}
    4. XGBoost (SMOTE)                     -> Acc: {results['XGB_SMOTE']['accuracy']:.2%} | F1: {results['XGB_SMOTE']['f1_macro']:.4f}
    5. Random Forest (balanced)            -> Acc: {results['RF']['accuracy']:.2%} | F1: {results['RF']['f1_macro']:.4f}

    Best model (Macro F1): {results[best_model_key]['model_name']}

    Output files:
    -------------
    Models:  models/*.joblib
    Scaler:  models/feature_scaler.joblib
    Plots:   outputs/training_history.png
             outputs/confusion_matrices.png
             outputs/feature_importance.png
             outputs/per_class_accuracy.png
    CSV:     outputs/model_comparison.csv

    Next step:
    ----------
    Run 03_apply_to_image.py to classify the full satellite image
""")

print("=" * 70)
