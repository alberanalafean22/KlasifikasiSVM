//Inisialisasi ROI untuk Kota Batam
var roi = ee.FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.eq('ADM2_NAME', 'Kota Batam'))
  .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'));
Map.centerObject(roi, 10);
Map.addLayer(roi, {}, "Kota Batam");

//Ambil koleksi citra Sentinel-2 SR Harmonized
var collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(roi)
  .filterDate("2024-01-01", "2024-12-31")
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(image) {
    return image.select(['B2', 'B3', 'B4', 'B8']) // Blue, Green, Red, NIR
                .divide(10000)
                .copyProperties(image, image.propertyNames());
  });

//Hitung median
var median = collection.median().clip(roi);

//Tampilkan citra RGB
Map.addLayer(median, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, "RGB");

//TRAINING DATA (tentukan polygon secara manual)
var areaterbangun = ee.Geometry.Polygon(
        [[[104.02539263565725,1.1400776826761825], [104.03054247696585,1.136988387915249], [104.03346072037405,1.1491739201887468], [104.0248776515264,1.1481441589012948],[104.02539263565725,1.1400776826761825] ]]);
var vegetasi = ee.Geometry.Polygon(
        [[[103.97358614707248,1.0838786625085348], [103.98036677146213,1.0734091740699037], [103.98663241172092,1.080274416457961], [103.97736269736545,1.0873112738069546],[103.97358614707248,1.0838786625085348] ]]);
var lahanterbuka = ee.Geometry.Polygon(
        [[[104.1054278782567,1.1480566869205289], [104.12576975142565,1.1458255361241416], [104.12036241805163,1.153205489799842], [104.10680116927233,1.153205489799842],[104.1054278782567,1.1480566869205289] ]]);

var builtupFeature = ee.Feature(areaterbangun, {class: 0});
var vegetationFeature = ee.Feature(vegetasi, {class: 1});
var barelandFeature = ee.Feature(lahanterbuka, {class: 2});

var trainingPolygons = ee.FeatureCollection([builtupFeature, vegetationFeature, barelandFeature]);
Map.addLayer(trainingPolygons, {}, 'Training Areas');

//Sampling nilai spektral
var training = median.sampleRegions({
  collection: trainingPolygons,
  properties: ['class'],
  scale: 10
});

//Split data menjadi train (70%) dan test (30%)
var withRandom = training.randomColumn('random');
var split = 0.7;
var trainingSet = withRandom.filter(ee.Filter.lt('random', split));
var testingSet = withRandom.filter(ee.Filter.gte('random', split));


// KLASIFIKASI DENGAN SVM
var trainedClassifier = ee.Classifier.libsvm().train({
  features: trainingSet,
  classProperty: 'class',
  inputProperties: ['B2', 'B3', 'B4', 'B8']
});
var classified = median.classify(trainedClassifier);


//VISUALISASI
//(Kelas: 0-Area terbangun[black], 1-Vegetasi[green], 2-Lahanterbuka[white])
var palette = ['black', 'green', 'white'];
Map.addLayer(classified, {min: 0, max: 2, palette: palette}, 'Klasifikasi SVM');

//Evaluasi akurasi
var validated = testingSet.classify(trainedClassifier);
var testAccuracy = validated.errorMatrix('class', 'classification');
print('Confusion Matrix (SVM):', testAccuracy);
print('Overall Accuracy (SVM):', testAccuracy.accuracy());

//Ekspor hasil
Export.image.toDrive({
  image: classified,
  description: 'Klasifikasi_SVM_Kota_Batam_2024',
  folder: 'GEE',
  fileNamePrefix: 'Klasifikasi_SVM_Batam_2024',
  region: roi.geometry(),
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
