/*
 * Crop Classification - Google Earth Engine Data Acquisition
 * ===========================================================
 * Study Area: Elgabel Region, Sudan
 * Sensor: Sentinel-2 Surface Reflectance (Level-2A)
 * Period: Q1 2020 (January - March)
 *
 * This script:
 *   1. Creates a cloud-free Sentinel-2 median composite
 *   2. Computes 14 spectral indices
 *   3. Exports a 24-band GeoTIFF and training data CSV
 *
 * Usage:
 *   - Open in Google Earth Engine Code Editor (code.earthengine.google.com)
 *   - Define your study area geometry and training polygons
 *   - Run the script and start the export tasks in the Tasks tab
 */

// ============================================================================
// 1. STUDY AREA
// ============================================================================
// Replace with your own geometry (Feature or FeatureCollection)
// Example: use the GEE drawing tools to draw a rectangle over Elgabel Region
var studyArea = ee.Geometry.Rectangle([33.0, 13.5, 34.5, 14.5]);

Map.centerObject(studyArea, 10);
Map.addLayer(studyArea, {color: 'red'}, 'Study Area');

// ============================================================================
// 2. SENTINEL-2 COMPOSITE
// ============================================================================
// Cloud masking function using the SCL band (Scene Classification Layer)
function maskS2Clouds(image) {
  var scl = image.select('SCL');
  // Keep vegetation, bare soil, water; mask clouds and shadows
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7)).or(scl.eq(11));
  return image.updateMask(mask)
              .select('B.*')
              .divide(10000)  // Scale to reflectance [0-1]
              .copyProperties(image, ['system:time_start']);
}

// Filter Sentinel-2 SR collection
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(studyArea)
    .filterDate('2020-01-01', '2020-03-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2Clouds);

print('Number of images:', s2.size());

// Create median composite with 10 spectral bands
var composite = s2.median().clip(studyArea);

var spectralBands = composite.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']);

// ============================================================================
// 3. SPECTRAL INDICES (14 indices)
// ============================================================================
var NDVI = spectralBands.normalizedDifference(['B8', 'B4']).rename('NDVI');
var EVI = spectralBands.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {NIR: spectralBands.select('B8'), RED: spectralBands.select('B4'), BLUE: spectralBands.select('B2')}
).rename('EVI');
var SAVI = spectralBands.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
    {NIR: spectralBands.select('B8'), RED: spectralBands.select('B4')}
).rename('SAVI');
var NDRE = spectralBands.normalizedDifference(['B8', 'B5']).rename('NDRE');
var GNDVI = spectralBands.normalizedDifference(['B8', 'B3']).rename('GNDVI');
var NDMI = spectralBands.normalizedDifference(['B8', 'B11']).rename('NDMI');
var BSI = spectralBands.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    {SWIR: spectralBands.select('B11'), RED: spectralBands.select('B4'),
     NIR: spectralBands.select('B8'), BLUE: spectralBands.select('B2')}
).rename('BSI');
var MNDWI = spectralBands.normalizedDifference(['B3', 'B11']).rename('MNDWI');
var LSWI = spectralBands.normalizedDifference(['B8', 'B12']).rename('LSWI');
var GCVI = spectralBands.expression(
    '(NIR / GREEN) - 1',
    {NIR: spectralBands.select('B8'), GREEN: spectralBands.select('B3')}
).rename('GCVI');
var WDRVI = spectralBands.expression(
    '(0.2 * NIR - RED) / (0.2 * NIR + RED)',
    {NIR: spectralBands.select('B8'), RED: spectralBands.select('B4')}
).rename('WDRVI');
var CIgreen = spectralBands.expression(
    '(NIR / GREEN) - 1',
    {NIR: spectralBands.select('B8'), GREEN: spectralBands.select('B3')}
).rename('CIgreen');
var CIrededge = spectralBands.expression(
    '(NIR / REDEDGE) - 1',
    {NIR: spectralBands.select('B8'), REDEDGE: spectralBands.select('B5')}
).rename('CIrededge');
var MSAVI = spectralBands.expression(
    '(2 * NIR + 1 - sqrt(pow(2 * NIR + 1, 2) - 8 * (NIR - RED))) / 2',
    {NIR: spectralBands.select('B8'), RED: spectralBands.select('B4')}
).rename('MSAVI');

// Stack all 24 bands
var fullStack = spectralBands
    .addBands(NDVI).addBands(EVI).addBands(SAVI).addBands(NDRE)
    .addBands(GNDVI).addBands(NDMI).addBands(BSI).addBands(MNDWI)
    .addBands(LSWI).addBands(GCVI).addBands(WDRVI)
    .addBands(CIgreen).addBands(CIrededge).addBands(MSAVI);

print('Full stack bands:', fullStack.bandNames());

// Visualization
var visParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3};
Map.addLayer(composite, visParams, 'True Color Composite');
Map.addLayer(NDVI, {min: 0, max: 0.8, palette: ['brown', 'yellow', 'green']}, 'NDVI');

// ============================================================================
// 4. TRAINING DATA
// ============================================================================
// Define training classes as FeatureCollections with a 'class' property:
//   0: Cotton, 1: Wheat, 2: Fallow, 3: Grass, 4: Water
//
// Example (replace with your own polygons):
// var cotton = ee.FeatureCollection('users/YOUR_USERNAME/cotton_polygons');
// var wheat  = ee.FeatureCollection('users/YOUR_USERNAME/wheat_polygons');
// var fallow = ee.FeatureCollection('users/YOUR_USERNAME/fallow_polygons');
// var grass  = ee.FeatureCollection('users/YOUR_USERNAME/grass_polygons');
// var water  = ee.FeatureCollection('users/YOUR_USERNAME/water_polygons');
//
// Merge and assign class labels:
// var trainingPolygons = cotton.map(function(f) { return f.set('class', 0).set('classname', 'Cotton'); })
//   .merge(wheat.map(function(f) { return f.set('class', 1).set('classname', 'Wheat'); }))
//   .merge(fallow.map(function(f) { return f.set('class', 2).set('classname', 'Fallow'); }))
//   .merge(grass.map(function(f) { return f.set('class', 3).set('classname', 'Grass'); }))
//   .merge(water.map(function(f) { return f.set('class', 4).set('classname', 'Water'); }));

// Sample training points from polygons
// var trainingData = fullStack.sampleRegions({
//   collection: trainingPolygons,
//   properties: ['class', 'classname'],
//   scale: 10,
//   geometries: true
// });
// print('Training samples:', trainingData.size());

// ============================================================================
// 5. EXPORTS
// ============================================================================
// Export 24-band composite
Export.image.toDrive({
  image: fullStack,
  description: 'S2_composite_24bands_2020_Q1',
  folder: 'CropClassification',
  region: studyArea,
  scale: 10,
  maxPixels: 1e13,
  crs: 'EPSG:32636',
  fileFormat: 'GeoTIFF'
});

// Export key indices only (smaller file for quick visualization)
var keyIndices = NDVI.addBands(EVI).addBands(SAVI).addBands(NDRE)
    .addBands(GNDVI).addBands(NDMI).addBands(BSI).addBands(MNDWI);

Export.image.toDrive({
  image: keyIndices,
  description: 'S2_key_indices_2020_Q1',
  folder: 'CropClassification',
  region: studyArea,
  scale: 10,
  maxPixels: 1e13,
  crs: 'EPSG:32636',
  fileFormat: 'GeoTIFF'
});

// Export training data as CSV
// Export.table.toDrive({
//   collection: trainingData,
//   description: 'crop_training_data_5classes_2020',
//   folder: 'CropClassification',
//   fileFormat: 'CSV'
// });

// Export training data as GeoJSON
// Export.table.toDrive({
//   collection: trainingData,
//   description: 'crop_training_data_5classes_2020_geo',
//   folder: 'CropClassification',
//   fileFormat: 'GeoJSON'
// });

print('=== Exports ready. Go to Tasks tab to start them. ===');
