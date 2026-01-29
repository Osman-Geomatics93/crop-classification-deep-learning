"""
Crop Classification Project - Step 5: Apply Best PyTorch Model to GeoTIFF
==========================================================================
Loads the best-performing PyTorch model and classifies the full satellite image.
Handles band order mismatch between CSV features and GeoTIFF bands.
Uses chunk-based processing and GPU inference for efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path
import warnings
import time
import joblib
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from osgeo import gdal

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\Deep_Learning2026-20260129T154240Z-3-001\Deep_Learning2026")
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = DATA_DIR / "models"

IMAGE_PATH = DATA_DIR / "S2_composite_24bands_2020_Q1.tif"

CLASS_NAMES = {0: 'Cotton', 1: 'Wheat', 2: 'Fallow', 3: 'Grass', 4: 'Water'}
CLASS_COLORS_HEX = {0: '#FF8C00', 1: '#FFD700', 2: '#8B4513', 3: '#32CD32', 4: '#0000FF'}
NUM_CLASSES = 5

CLASS_COLORS_RGB = {
    0: (255, 140, 0),    # Cotton - Orange
    1: (255, 215, 0),    # Wheat - Gold
    2: (139, 69, 19),    # Fallow - Brown
    3: (50, 205, 50),    # Grass - Green
    4: (0, 0, 255),      # Water - Blue
    255: (0, 0, 0)       # NoData - Black
}

CHUNK_ROWS = 500
BATCH_SIZE = 4096  # larger batches for GPU inference

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("CROP CLASSIFICATION - APPLY PYTORCH MODEL TO SATELLITE IMAGE")
print("=" * 70)
print(f"    Device: {DEVICE}")

# ============================================================================
# MODEL ARCHITECTURE DEFINITIONS (must match training)
# ============================================================================

class SpectralMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)


class SpectralCNN1D(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class SpectralHybrid(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        cnn_out = self.cnn_branch(x.unsqueeze(1)).squeeze(-1)
        mlp_out = self.mlp_branch(x)
        return self.fusion(torch.cat([cnn_out, mlp_out], dim=1))


class SpectralAttention(nn.Module):
    def __init__(self, n_features, n_classes, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x.unsqueeze(-1))
        x = x + self.pos_encoding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0])


# Map model class name to constructor
MODEL_CLASSES = {
    'SpectralMLP': SpectralMLP,
    'SpectralCNN1D': SpectralCNN1D,
    'SpectralHybrid': SpectralHybrid,
    'SpectralAttention': SpectralAttention,
}

# ============================================================================
# 1. LOAD TRAINED MODEL AND SCALER
# ============================================================================
print("\n[1] Loading trained model and scaler...")

scaler = joblib.load(MODEL_DIR / 'pytorch_scaler.joblib')
feature_cols = joblib.load(MODEL_DIR / 'pytorch_feature_cols.joblib')
model_info = joblib.load(MODEL_DIR / 'pytorch_model_info.joblib')

best_model_name = model_info['best_model']
n_features = model_info['n_features']
n_classes = model_info['n_classes']

print(f"    Best model: {best_model_name}")
print(f"    Features ({n_features}): {feature_cols}")

# Load the best model checkpoint
checkpoint_path = MODEL_DIR / f'pytorch_{best_model_name}.pth'
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

ModelClass = MODEL_CLASSES[best_model_name]
model = ModelClass(n_features, n_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f"    Loaded checkpoint: {checkpoint_path.name}")

# ============================================================================
# 2. LOAD GEOTIFF AND BUILD BAND MAPPING
# ============================================================================
print("\n[2] Loading satellite image and mapping bands...")

ds = gdal.Open(str(IMAGE_PATH), gdal.GA_ReadOnly)
if ds is None:
    raise FileNotFoundError(f"Cannot open: {IMAGE_PATH}")

n_bands = ds.RasterCount
height = ds.RasterYSize
width = ds.RasterXSize
geotransform = ds.GetGeoTransform()
projection = ds.GetProjection()

pixel_size_x = abs(geotransform[1])
pixel_size_y = abs(geotransform[5])
pixel_area_m2 = pixel_size_x * pixel_size_y
pixel_area_ha = pixel_area_m2 / 10000

print(f"    Image: {width} x {height} pixels, {n_bands} bands")
print(f"    Pixel size: {pixel_size_x:.1f} x {pixel_size_y:.1f} m")

# Read GeoTIFF band descriptions
tif_band_names = {}
for b in range(1, n_bands + 1):
    desc = ds.GetRasterBand(b).GetDescription().strip()
    if desc:
        tif_band_names[desc] = b

print(f"\n    GeoTIFF band names: {list(tif_band_names.keys())}")

# Build reorder mapping: for each CSV feature, find the matching GeoTIFF band
band_read_order = []
matched = 0

print(f"\n    Band mapping (CSV feature -> GeoTIFF band):")
for i, feat_name in enumerate(feature_cols):
    if feat_name in tif_band_names:
        tif_band_idx = tif_band_names[feat_name]
        band_read_order.append(tif_band_idx)
        matched += 1
        print(f"    Feature[{i:2d}] {feat_name:<14} -> TIF Band {tif_band_idx:2d}")
    else:
        print(f"    Feature[{i:2d}] {feat_name:<14} -> NOT FOUND!")
        band_read_order.append(None)

print(f"\n    Matched: {matched}/{len(feature_cols)} features")

if matched != len(feature_cols):
    print("    WARNING: Not all features matched! Results may be inaccurate.")
    for i in range(len(band_read_order)):
        if band_read_order[i] is None:
            band_read_order[i] = i + 1

is_same_order = all(band_read_order[i] == i + 1 for i in range(len(band_read_order)))
if is_same_order:
    print("    Band order matches - no reordering needed.")
else:
    print("    Band order DIFFERS - reordering will be applied!")
    print(f"    Read order: {band_read_order}")

# ============================================================================
# 3. CLASSIFY IMAGE IN CHUNKS (GPU inference)
# ============================================================================
print(f"\n[3] Classifying image in chunks of {CHUNK_ROWS} rows...")

classification = np.full((height, width), 255, dtype=np.uint8)

start_time = time.time()
total_pixels = 0
valid_pixels = 0
n_chunks = (height + CHUNK_ROWS - 1) // CHUNK_ROWS

for chunk_idx in range(n_chunks):
    row_start = chunk_idx * CHUNK_ROWS
    row_end = min(row_start + CHUNK_ROWS, height)
    chunk_height = row_end - row_start

    # Read bands in the CORRECT ORDER matching CSV features
    chunk_data = np.zeros((n_features, chunk_height, width), dtype=np.float32)
    for feat_idx, tif_band in enumerate(band_read_order):
        band = ds.GetRasterBand(tif_band)
        chunk_data[feat_idx] = band.ReadAsArray(0, row_start, width, chunk_height).astype(np.float32)

    # Reshape to (pixels, features)
    chunk_2d = chunk_data.reshape(n_features, -1).T
    n_pixels = chunk_2d.shape[0]
    total_pixels += n_pixels

    # Valid pixel mask
    valid_mask = np.all(np.isfinite(chunk_2d), axis=1) & np.any(chunk_2d != 0, axis=1)
    n_valid = valid_mask.sum()
    valid_pixels += n_valid

    if n_valid > 0:
        # Scale features
        chunk_scaled = scaler.transform(chunk_2d[valid_mask])

        # GPU batch inference
        predictions = np.zeros(n_valid, dtype=np.int64)
        for batch_start in range(0, n_valid, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_valid)
            batch_tensor = torch.FloatTensor(chunk_scaled[batch_start:batch_end]).to(DEVICE)
            with torch.no_grad():
                outputs = model(batch_tensor)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions[batch_start:batch_end] = batch_preds

        chunk_result = np.full(n_pixels, 255, dtype=np.uint8)
        chunk_result[valid_mask] = predictions.astype(np.uint8)
        classification[row_start:row_end, :] = chunk_result.reshape(chunk_height, width)

    # Progress
    progress = (chunk_idx + 1) / n_chunks * 100
    if (chunk_idx + 1) % max(1, n_chunks // 10) == 0 or chunk_idx == n_chunks - 1:
        elapsed = time.time() - start_time
        print(f"    Chunk {chunk_idx+1:4d}/{n_chunks} ({progress:5.1f}%) | "
              f"Valid: {valid_pixels:,} | Time: {elapsed:.1f}s")

elapsed = time.time() - start_time
print(f"\n    Classification complete in {elapsed:.1f}s")
print(f"    Total pixels: {total_pixels:,}")
print(f"    Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")

# ============================================================================
# 4. AREA STATISTICS
# ============================================================================
print("\n[4] Computing area statistics...")

print(f"\n    {'Class':<8} {'Name':<10} {'Pixels':>12} {'Area (ha)':>12} {'Area (km²)':>12} {'%':>8}")
print("    " + "-" * 65)

total_valid = 0
class_stats = {}

for cls in range(NUM_CLASSES):
    pixel_count = int(np.sum(classification == cls))
    area_ha = pixel_count * pixel_area_ha
    area_km2 = area_ha / 100
    total_valid += pixel_count
    class_stats[cls] = {'pixels': pixel_count, 'area_ha': area_ha, 'area_km2': area_km2}

for cls in range(NUM_CLASSES):
    s = class_stats[cls]
    pct = s['pixels'] / total_valid * 100 if total_valid > 0 else 0
    print(f"    {cls:<8} {CLASS_NAMES[cls]:<10} {s['pixels']:>12,} {s['area_ha']:>12.2f} {s['area_km2']:>12.2f} {pct:>7.1f}%")

nodata_count = int(np.sum(classification == 255))
print("    " + "-" * 65)
print(f"    {'Total':<8} {'Valid':<10} {total_valid:>12,} {total_valid*pixel_area_ha:>12.2f} {total_valid*pixel_area_ha/100:>12.2f} {'100.0':>7}%")
print(f"    {'255':<8} {'NoData':<10} {nodata_count:>12,}")

# ============================================================================
# 5. SAVE CLASSIFICATION GEOTIFF (single band with color table)
# ============================================================================
print("\n[5] Saving classification GeoTIFF...")

output_path = OUTPUT_DIR / 'pytorch_classification_map.tif'

driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(str(output_path), width, height, 1, gdal.GDT_Byte,
                       options=['COMPRESS=LZW', 'TILED=YES'])
out_ds.SetGeoTransform(geotransform)
out_ds.SetProjection(projection)

out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(classification)
out_band.SetNoDataValue(255)

ct = gdal.ColorTable()
for cls, (r, g, b) in CLASS_COLORS_RGB.items():
    if cls < 256:
        ct.SetColorEntry(cls, (r, g, b, 255))
out_band.SetColorTable(ct)
out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
out_band.FlushCache()
out_ds = None

print(f"    Saved: {output_path}")

# ============================================================================
# 6. CREATE VISUALIZATION
# ============================================================================
print("\n[6] Creating classification visualization...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Classification map
colors = [CLASS_COLORS_HEX[i] for i in range(NUM_CLASSES)]
cmap = ListedColormap(colors)
bounds = list(range(NUM_CLASSES + 1))
norm = BoundaryNorm(bounds, cmap.N)

display_map = np.ma.masked_where(classification == 255, classification)

axes[0].imshow(display_map, cmap=cmap, norm=norm, interpolation='nearest')
axes[0].set_title(f'PyTorch Classification Map\n{best_model_name} | Elgabel Region, Sudan',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')

legend_patches = [Patch(facecolor=CLASS_COLORS_HEX[i], edgecolor='black', label=CLASS_NAMES[i])
                  for i in range(NUM_CLASSES)]
axes[0].legend(handles=legend_patches, loc='lower right', fontsize=10,
               title='Crop Class', title_fontsize=11, fancybox=True)

# Area pie chart
labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
sizes = [class_stats[i]['area_ha'] for i in range(NUM_CLASSES)]
colors_pie = [CLASS_COLORS_HEX[i] for i in range(NUM_CLASSES)]

nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
if nonzero:
    nz_labels, nz_sizes, nz_colors = zip(*nonzero)
    wedges, texts, autotexts = axes[1].pie(
        nz_sizes, labels=nz_labels, colors=nz_colors, autopct='%1.1f%%',
        startangle=90, explode=[0.03] * len(nz_sizes), pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)

axes[1].set_title('Classified Area Distribution\n(hectares)', fontsize=13, fontweight='bold')

area_text = "\n".join([f"{CLASS_NAMES[i]}: {class_stats[i]['area_ha']:.1f} ha ({class_stats[i]['area_km2']:.2f} km²)"
                       for i in range(NUM_CLASSES)])
axes[1].text(0, -1.3, area_text, ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pytorch_classification_visualization.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"    Saved: {OUTPUT_DIR / 'pytorch_classification_visualization.png'}")

# ============================================================================
# 7. TRUE COLOR vs CLASSIFICATION COMPARISON
# ============================================================================
print("\n[7] Creating true color vs classification comparison...")

try:
    b4_band = tif_band_names.get('B4')
    b3_band = tif_band_names.get('B3')
    b2_band = tif_band_names.get('B2')

    if b4_band and b3_band and b2_band:
        red = ds.GetRasterBand(b4_band).ReadAsArray().astype(np.float32) if ds else None

        # Reopen if closed
        if ds is None:
            ds = gdal.Open(str(IMAGE_PATH), gdal.GA_ReadOnly)

        red = ds.GetRasterBand(b4_band).ReadAsArray().astype(np.float32)
        green = ds.GetRasterBand(b3_band).ReadAsArray().astype(np.float32)
        blue = ds.GetRasterBand(b2_band).ReadAsArray().astype(np.float32)

        rgb_tc = np.dstack([red, green, blue])
        for i in range(3):
            p2, p98 = np.nanpercentile(rgb_tc[:, :, i][rgb_tc[:, :, i] > 0], [2, 98])
            rgb_tc[:, :, i] = np.clip((rgb_tc[:, :, i] - p2) / (p98 - p2 + 1e-10), 0, 1)
        rgb_tc = np.nan_to_num(rgb_tc, nan=0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        axes[0].imshow(rgb_tc)
        axes[0].set_title('True Color Composite (B4-B3-B2)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')

        axes[1].imshow(display_map, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1].set_title(f'PyTorch Classification ({best_model_name})', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        axes[1].legend(handles=legend_patches, loc='lower right', fontsize=9,
                       title='Crop Class', fancybox=True)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'pytorch_truecolor_vs_classification.png', dpi=200, bbox_inches='tight')
        plt.show()
        print(f"    Saved: {OUTPUT_DIR / 'pytorch_truecolor_vs_classification.png'}")
    else:
        print("    Could not find B2/B3/B4 bands for true color composite.")
except Exception as e:
    print(f"    Error creating true color composite: {e}")

# Close source dataset
ds = None

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PYTORCH IMAGE CLASSIFICATION COMPLETE")
print("=" * 70)

print(f"""
    Input: {IMAGE_PATH.name} ({width}x{height}, {n_bands} bands)
    Model: {best_model_name} (PyTorch)
    Device: {DEVICE}
    Band reordering: {'Applied' if not is_same_order else 'Not needed'}

    Classification Results:
    -----------------------""")

for cls in range(NUM_CLASSES):
    s = class_stats[cls]
    pct = s['pixels'] / total_valid * 100 if total_valid > 0 else 0
    print(f"    {CLASS_NAMES[cls]:<10}: {s['area_ha']:>10.1f} ha  ({s['area_km2']:>7.2f} km²)  [{pct:.1f}%]")

print(f"""
    Output Files:
    -------------
    - outputs/pytorch_classification_map.tif              (GeoTIFF with colors)
    - outputs/pytorch_classification_visualization.png
    - outputs/pytorch_truecolor_vs_classification.png
""")

print("=" * 70)
