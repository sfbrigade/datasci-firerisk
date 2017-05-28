var fs = require('fs');
var readline = require('readline');
var async = require('async');
var svm = require('node-svm');

var dataTransformByField = {
  'Closed Roll Assessed Improvement Value': 'identity',
  'Closed Roll Assessed Land Value': 'identity',
  //'End Street Number': 'class',
  'Had Fire Incident': 'label',
  //'Latitude': 'identity',
  //'Longitude': 'identity',
  'Lot Area': 'identity',
  'Neighborhood Code Definition': 'class',
  'Number of Rooms': 'identity',
  'Number of Stories': 'identity',
  'Number of Units': 'identity',
  'Property Area in Square Feet': 'identity',
  'Property Class Code Definition': 'class',
  //'Start Street Number': 'class',
  //'Street Name': 'class',
  //'Street Suffix': 'class',
  'Year Property Built': 'identity',
  'Zipcode of Parcel': 'class'
};

var svmOptions = {
  probability: true,
  c: [0.1, 0.3, 1, 3, 10],
  gamma: [0.1, 0.3, 1, 3, 10]
};

function applyRiskScore(dataset) {
  var valuesByField = buildValuesByField(dataset);
  var indexByValueByField = buildIndexByValueByField(valuesByField);

  var positiveExamples = dataset.filter(function(example) {
    return example['Had Fire Incident'] === true;
  });

  var negativeExamples = dataset.filter(function(example) {
    return example['Had Fire Incident'] === false;
  }).filter(function(example, index) {
    return index % 100 === 0;
  });

  var trainingSet = [].concat(positiveExamples, negativeExamples);
  var modelInput = buildModelInput(trainingSet, indexByValueByField);

  var model = new svm.SVM(svmOptions);
  model.train(modelInput).progress(function(rate) {
    readline.cursorTo(process.stderr, 0);
    process.stderr.write('Training SVM model: '
     + (rate * 100).toFixed(1) + '% complete');
  }).then(function(report) {
    console.error();
    console.error('c: %s, gamma: %s',
      report[0].params.c,
      report[0].params.gamma);
    console.error('fscore: %s, precision: %s, recall: %s, accuracy: %s',
      report[1].fscore.toFixed(3),
      report[1].precision.toFixed(3),
      report[1].recall.toFixed(3),
      report[1].accuracy.toFixed(3));

    var testSetInput = buildModelInput(dataset, indexByValueByField);
    testSetInput.forEach(function(example, index) {
      var prediction = model.predictSync(example[0]);
      var positiveProbability = model.predictProbabilitiesSync(example[0])[1];
      var datum = dataset[index];
      var values = [
        datum['Start Street Number'],
        datum['End Street Number'],
        datum['Street Name'],
        datum['Street Suffix'],
        datum['Latitude'],
        datum['Longitude'],
        datum['Had Fire Incident'] === true ? 1 : 0,
        prediction,
        positiveProbability.toFixed(4)
      ];
      console.log(values.join('\t'));
    });
  });
}

function buildValuesByField(dataset) {
  return Object.keys(dataTransformByField).filter(function(fieldName) {
    return dataTransformByField[fieldName] === 'class';
  }).reduce(function(valuesByField, fieldName) {
    var valueSet = dataset.reduce(function(valueSet, example) {
      var value = example[fieldName];
      valueSet[value] = true;
      return valueSet;
    }, {});
    valuesByField[fieldName] = Object.keys(valueSet).sort();
    return valuesByField;
  }, {});
}

function buildIndexByValueByField(valuesByField) {
  var fieldNames = Object.keys(valuesByField);
  return fieldNames.reduce(function(indexByValueByField, fieldName) {
    var values = valuesByField[fieldName];
    var indexByValue = values.reduce(function(indexByValue, value, index) {
      indexByValue[value] = index;
      return indexByValue;
    }, {});
    indexByValueByField[fieldName] = indexByValue;
    return indexByValueByField;
  }, {});
}

function buildModelInput(dataset, indexByValueByField) {
  var fieldNames = Object.keys(dataTransformByField).sort();
  return dataset.map(function(example) {
    var features = [];
    var label = 0;
    fieldNames.forEach(function(fieldName) {
      switch(dataTransformByField[fieldName]) {
        case 'label': label = example[fieldName] ? 1 : 0;
          break;
        case 'identity': features.push(Number(example[fieldName]));
          break;
        case 'class':
          var value = example[fieldName];
          var indexByValue = indexByValueByField[fieldName];
          features.push(indexByValue[value]);
          break;
        default:
      }
    });
    return [ features, label ];
  });
}

function buildFileReads(fileNames) {
  return fileNames.map(function(fileName) {
    return function(cb) {
      fs.readFile(fileName, 'utf8', cb);
    };
  });
}

function main() {
  if(process.argv.length !== 3) {
    console.error('%s %s <labeled dataset json>',
      process.argv[0], process.argv[1]);
    return;
  }

  var fileNames = [ process.argv[2] ];
  async.parallel(buildFileReads(fileNames), function(err, results) {
    if(err) {
      console.error('Error reading files', err);
      return;
    }

    var labeledProperties = applyRiskScore(JSON.parse(results[0]));
  });
}

main();
