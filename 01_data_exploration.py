"""
Crop Classification Project - Step 1: Data Loading and Exploration
==================================================================
This script loads and explores the training data for crop classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 60)
print("CROP CLASSIFICATION - DATA EXPLORATION")
print("=" * 60)

# Define paths
DATA_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\Deep_Learning2026-20260129T154240Z-3-001\Deep_Learning2026")
csv_path = DATA_DIR / "crop_training_data_5classes_2020.csv"

# Load the data
print("\n[1] Loading data...")
df = pd.read_csv(csv_path)
print(f"    Data loaded successfully!")

# ============================================================================
# 2. BASIC DATA INFO
# ============================================================================
print("\n" + "=" * 60)
print("[2] BASIC DATA INFORMATION")
print("=" * 60)

print(f"\n    Shape: {df.shape[0]:,} samples, {df.shape[1]} columns")
print(f"\n    Column names:")
for i, col in enumerate(df.columns):
    print(f"        {i+1:2d}. {col}")

print(f"\n    Data types:")
print(df.dtypes.value_counts())

print(f"\n    Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 3. CLASS DISTRIBUTION
# ============================================================================
print("\n" + "=" * 60)
print("[3] CLASS DISTRIBUTION")
print("=" * 60)

# Define class names and colors
class_names = {0: 'Cotton', 1: 'Wheat', 2: 'Fallow', 3: 'Grass', 4: 'Water'}
class_colors = {0: '#FF8C00', 1: '#FFD700', 2: '#8B4513', 3: '#32CD32', 4: '#0000FF'}

# Calculate class distribution
class_counts = df['class'].value_counts().sort_index()
class_percentages = (class_counts / len(df) * 100).round(2)

print("\n    Class Distribution:")
print("    " + "-" * 45)
print(f"    {'Class':<8} {'Name':<10} {'Samples':>10} {'Percent':>10}")
print("    " + "-" * 45)
for cls in sorted(class_counts.index):
    name = class_names.get(cls, f'Class_{cls}')
    count = class_counts[cls]
    pct = class_percentages[cls]
    print(f"    {cls:<8} {name:<10} {count:>10,} {pct:>9.1f}%")
print("    " + "-" * 45)
print(f"    {'Total':<8} {'':<10} {len(df):>10,} {'100.0':>9}%")

# Calculate imbalance ratio
max_class = class_counts.max()
min_class = class_counts.min()
print(f"\n    Imbalance ratio (max/min): {max_class/min_class:.1f}:1")

# ============================================================================
# 4. VISUALIZE CLASS DISTRIBUTION
# ============================================================================
print("\n[4] Creating class distribution visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = [class_colors[i] for i in sorted(class_counts.index)]
bars = axes[0].bar(range(len(class_counts)), class_counts.values, color=colors, edgecolor='black')
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Number of Samples', fontsize=12)
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(class_counts)))
axes[0].set_xticklabels([f"{i}: {class_names[i]}" for i in sorted(class_counts.index)], rotation=45, ha='right')

# Add value labels on bars
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{count:,}', ha='center', va='bottom', fontsize=10)

# Pie chart
axes[1].pie(class_counts.values, labels=[class_names[i] for i in sorted(class_counts.index)],
            colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.05]*5)
axes[1].set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(DATA_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: class_distribution.png")

# ============================================================================
# 5. FEATURE STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("[5] FEATURE STATISTICS")
print("=" * 60)

# Get feature columns (exclude class and classname)
feature_cols = [col for col in df.columns if col not in ['class', 'classname', 'system:index', '.geo']]
print(f"\n    Number of features: {len(feature_cols)}")
print(f"\n    Features: {feature_cols}")

# Basic statistics
print("\n    Feature Statistics (all classes):")
stats_df = df[feature_cols].describe().T
stats_df['range'] = stats_df['max'] - stats_df['min']
print(stats_df[['mean', 'std', 'min', 'max', 'range']].round(4).to_string())

# Check for missing values
print("\n    Missing Values:")
missing = df[feature_cols].isnull().sum()
if missing.sum() == 0:
    print("    No missing values found!")
else:
    print(missing[missing > 0])

# Check for infinite values
print("\n    Infinite Values:")
inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
if inf_count == 0:
    print("    No infinite values found!")
else:
    print(f"    Found {inf_count} infinite values")

# ============================================================================
# 6. FEATURE STATISTICS BY CLASS
# ============================================================================
print("\n" + "=" * 60)
print("[6] KEY FEATURES BY CLASS")
print("=" * 60)

# Key vegetation indices to analyze
key_features = ['NDVI', 'EVI', 'SAVI', 'NDRE', 'BSI', 'MNDWI']
key_features = [f for f in key_features if f in feature_cols]

print("\n    Mean values of key features by class:")
print("    " + "-" * 70)

for feature in key_features:
    print(f"\n    {feature}:")
    for cls in sorted(df['class'].unique()):
        mean_val = df[df['class'] == cls][feature].mean()
        std_val = df[df['class'] == cls][feature].std()
        print(f"        {class_names[cls]:<10}: {mean_val:>8.4f} ± {std_val:.4f}")

# ============================================================================
# 7. BOX PLOTS OF KEY FEATURES
# ============================================================================
print("\n[7] Creating box plots of key features...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features[:6]):
    data_to_plot = [df[df['class'] == cls][feature].values for cls in sorted(df['class'].unique())]
    bp = axes[idx].boxplot(data_to_plot, labels=[class_names[i] for i in sorted(df['class'].unique())],
                           patch_artist=True)

    # Color the boxes
    for patch, cls in zip(bp['boxes'], sorted(df['class'].unique())):
        patch.set_facecolor(class_colors[cls])
        patch.set_alpha(0.7)

    axes[idx].set_title(f'{feature} by Crop Class', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(feature)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(DATA_DIR / 'feature_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: feature_boxplots.png")

# ============================================================================
# 8. CORRELATION MATRIX
# ============================================================================
print("\n[8] Creating correlation matrix...")

# Calculate correlation matrix
corr_matrix = df[feature_cols].corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(DATA_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: correlation_matrix.png")

# Identify highly correlated features
print("\n    Highly correlated feature pairs (|r| > 0.9):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr:
    for f1, f2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"        {f1} <-> {f2}: {corr:.3f}")
else:
    print("        None found")

# ============================================================================
# 9. FEATURE DISTRIBUTIONS
# ============================================================================
print("\n[9] Creating feature distribution plots...")

fig, axes = plt.subplots(4, 6, figsize=(20, 16))
axes = axes.flatten()

for idx, feature in enumerate(feature_cols[:24]):
    for cls in sorted(df['class'].unique()):
        data = df[df['class'] == cls][feature]
        axes[idx].hist(data, bins=30, alpha=0.5, label=class_names[cls],
                       color=class_colors[cls], density=True)
    axes[idx].set_title(feature, fontsize=10)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(labelsize=8)

# Add legend to the last subplot
axes[0].legend(loc='upper right', fontsize=8)
plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(DATA_DIR / 'feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: feature_distributions.png")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("[10] SUMMARY")
print("=" * 60)

print(f"""
    Dataset Summary:
    ----------------
    - Total samples: {len(df):,}
    - Number of features: {len(feature_cols)}
    - Number of classes: {df['class'].nunique()}

    Class Imbalance Issues:
    -----------------------
    - Majority class (Fallow): {class_counts[2]:,} samples ({class_percentages[2]:.1f}%)
    - Minority classes:
      * Cotton: {class_counts[0]:,} samples ({class_percentages[0]:.1f}%)
      * Water: {class_counts[4]:,} samples ({class_percentages[4]:.1f}%)
    - Imbalance ratio: {max_class/min_class:.1f}:1

    Recommendations:
    ----------------
    1. Use class weights during training
    2. Apply SMOTE for minority class oversampling
    3. Consider focal loss for handling imbalance
    4. Monitor per-class metrics (especially Cotton and Water)
    5. Use stratified sampling for train/test split

    Output files created:
    ---------------------
    - class_distribution.png
    - feature_boxplots.png
    - correlation_matrix.png
    - feature_distributions.png
""")

print("=" * 60)
print("Data exploration complete!")
print("=" * 60)
