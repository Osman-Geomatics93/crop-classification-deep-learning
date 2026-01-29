"""
Diagnose band order mismatch between CSV training data and GeoTIFF image.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from osgeo import gdal

DATA_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\Deep_Learning2026-20260129T154240Z-3-001\Deep_Learning2026")

# ============================================================================
# 1. CSV FEATURE ORDER (what the model was trained on)
# ============================================================================
print("=" * 70)
print("BAND ORDER DIAGNOSIS")
print("=" * 70)

df = pd.read_csv(DATA_DIR / "crop_training_data_5classes_2020.csv", nrows=5)
feature_cols = [col for col in df.columns if col not in ['class', 'classname', 'system:index', '.geo']]

print("\n[1] CSV feature column order (model training order):")
for i, col in enumerate(feature_cols):
    print(f"    Feature {i:2d}: {col}")

# ============================================================================
# 2. GEOTIFF BAND ORDER
# ============================================================================
print("\n[2] GeoTIFF band information:")

ds = gdal.Open(str(DATA_DIR / "S2_composite_24bands_2020_Q1.tif"), gdal.GA_ReadOnly)
n_bands = ds.RasterCount
print(f"    Total bands: {n_bands}")

tif_band_names = []
for b in range(1, n_bands + 1):
    band = ds.GetRasterBand(b)
    desc = band.GetDescription()
    # Also check metadata
    meta = band.GetMetadata()
    tif_band_names.append(desc if desc else f"Band_{b}")
    print(f"    Band {b:2d}: description='{desc}' | metadata={meta}")

# ============================================================================
# 3. CHECK VALUE RANGES
# ============================================================================
print("\n[3] Comparing value ranges:")

print("\n    CSV sample values (first pixel, Wheat class):")
wheat_row = df[df['class'] == 1].iloc[0]
for col in feature_cols:
    print(f"    {col:<12}: {wheat_row[col]:.6f}")

print("\n    GeoTIFF sample values (center pixel):")
cy, cx = ds.RasterYSize // 2, ds.RasterXSize // 2
# Find a non-zero pixel near center
for offset in range(0, 100):
    test_vals = []
    for b in range(1, n_bands + 1):
        val = ds.GetRasterBand(b).ReadAsArray(cx + offset, cy, 1, 1)[0, 0]
        test_vals.append(val)
    if any(v != 0 for v in test_vals):
        break

for b in range(n_bands):
    bname = tif_band_names[b] if tif_band_names[b] != f"Band_{b+1}" else f"Band {b+1}"
    print(f"    Band {b+1:2d} ({bname:<12}): {test_vals[b]:.6f}")

# ============================================================================
# 4. DETECT BAND MAPPING
# ============================================================================
print("\n[4] Band mapping analysis:")

# Check if band descriptions match feature names
print("\n    Trying to match GeoTIFF bands to CSV features...")

band_mapping = {}  # csv_feature_index -> geotiff_band_index (0-based)

if all(name.startswith("Band_") for name in tif_band_names):
    print("\n    GeoTIFF has NO band descriptions!")
    print("    We need to figure out the correct order.")
    print("\n    Likely GeoTIFF band order (standard Sentinel-2 export):")

    # Standard GEE export order for Sentinel-2 bands
    likely_order = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
                    'NDVI', 'EVI', 'SAVI', 'GNDVI', 'NDRE', 'NDRE2',
                    'NDWI', 'MNDWI', 'BSI', 'NDTI',
                    'CIgreen', 'CIrededge', 'MSAVI', 'GCVI']

    print(f"    Expected {len(likely_order)} bands: {likely_order}")

    # Try to verify by comparing value ranges
    print("\n    Verifying by comparing value ranges:")
    print(f"    {'CSV Feature':<14} {'CSV Range':<24} {'Likely TIF Band':<16} {'TIF Value':<12}")
    print("    " + "-" * 70)

    # Load full CSV stats
    df_full = pd.read_csv(DATA_DIR / "crop_training_data_5classes_2020.csv")

    for i, feat in enumerate(feature_cols):
        csv_min = df_full[feat].min()
        csv_max = df_full[feat].max()

        # Find this feature in the likely order
        if feat in likely_order:
            tif_idx = likely_order.index(feat)
            tif_val = test_vals[tif_idx] if tif_idx < len(test_vals) else "N/A"
            band_mapping[i] = tif_idx
            print(f"    {feat:<14} [{csv_min:>8.4f}, {csv_max:>8.4f}]   TIF Band {tif_idx+1:<8}  {tif_val}")
        else:
            print(f"    {feat:<14} [{csv_min:>8.4f}, {csv_max:>8.4f}]   NOT FOUND")
else:
    print("\n    GeoTIFF HAS band descriptions. Matching...")
    for i, feat in enumerate(feature_cols):
        for j, tif_name in enumerate(tif_band_names):
            if tif_name.strip().lower() == feat.strip().lower():
                band_mapping[i] = j
                break
        if i in band_mapping:
            print(f"    CSV[{i:2d}] {feat:<14} -> TIF Band {band_mapping[i]+1}")
        else:
            print(f"    CSV[{i:2d}] {feat:<14} -> NOT MATCHED")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("[5] SUMMARY")
print("=" * 70)

print(f"\n    CSV features: {len(feature_cols)}")
print(f"    TIF bands: {n_bands}")
print(f"    Matched: {len(band_mapping)}")

if band_mapping:
    # Check if mapping is identity (same order)
    is_identity = all(band_mapping[i] == i for i in band_mapping)
    print(f"    Same order: {'YES' if is_identity else 'NO - REORDERING NEEDED!'}")

    if not is_identity:
        print("\n    Required reorder mapping (CSV index -> TIF band):")
        for csv_idx in sorted(band_mapping):
            tif_idx = band_mapping[csv_idx]
            print(f"    Feature[{csv_idx:2d}] {feature_cols[csv_idx]:<14} = TIF Band {tif_idx + 1}")

ds = None
print("\n" + "=" * 70)
