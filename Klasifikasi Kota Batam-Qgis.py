import ee
from ee_plugin import Map  # Bagian dari Earth Engine Plugin di QGIS



# ROI Kota Batam
roi = ee.FeatureCollection("FAO/GAUL/2015/level2") \
  .filter(ee.Filter.eq('ADM2_NAME', 'Kota Batam')) \
  .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))

# Koleksi citra Sentinel-2 SR (4 band: B2, B3, B4, B8)
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
  .filterBounds(roi) \
  .filterDate("2024-01-01", "2024-12-31") \
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
  .map(lambda img: img.select(['B2', 'B3', 'B4', 'B8']).divide(10000).copyProperties(img, img.propertyNames()))

# Ambil median & crop ke ROI
median = collection.median().clip(roi)

# Polygon training manual
areaterbangun = ee.Geometry.Polygon([
    [[104.02539263565725,1.1400776826761825],
     [104.03054247696585,1.136988387915249],
     [104.03346072037405,1.1491739201887468],
     [104.0248776515264,1.1481441589012948],
     [104.02539263565725,1.1400776826761825]]
])
vegetasi = ee.Geometry.Polygon([
    [[103.97358614707248,1.0838786625085348],
     [103.98036677146213,1.0734091740699037],
     [103.98663241172092,1.080274416457961],
     [103.97736269736545,1.0873112738069546],
     [103.97358614707248,1.0838786625085348]]
])
lahanterbuka = ee.Geometry.Polygon([
    [[104.1054278782567,1.1480566869205289],
     [104.12576975142565,1.1458255361241416],
     [104.12036241805163,1.153205489799842],
     [104.10680116927233,1.153205489799842],
     [104.1054278782567,1.1480566869205289]]
])

# Buat fitur dengan label kelas
builtupFeature = ee.Feature(areaterbangun, {'class': 0})
vegetationFeature = ee.Feature(vegetasi, {'class': 1})
barelandFeature = ee.Feature(lahanterbuka, {'class': 2})
trainingPolygons = ee.FeatureCollection([builtupFeature, vegetationFeature, barelandFeature])

# Sampling citra berdasarkan polygon latih
training = median.sampleRegions(
    collection=trainingPolygons,
    properties=['class'],
    scale=10
)

# Split data training/testing
withRandom = training.randomColumn('random')
split = 0.7
trainingSet = withRandom.filter(ee.Filter.lt('random', split))
testingSet = withRandom.filter(ee.Filter.gte('random', split))

# Train klasifier SVM
classifier = ee.Classifier.libsvm().train(
    features=trainingSet,
    classProperty='class',
    inputProperties=['B2', 'B3', 'B4', 'B8']
)

# Klasifikasi citra
classified = median.classify(classifier)

# Visualisasi RGB dan hasil klasifikasi
rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}
class_vis = {'min': 0, 'max': 2, 'palette': ['black', 'green', 'white']}

# Tambahkan layer ke QGIS
Map.addLayer(median, rgb_vis, 'Citra Median RGB')
Map.addLayer(classified, class_vis, 'Hasil Klasifikasi SVM')
Map.centerObject(roi, 10)
