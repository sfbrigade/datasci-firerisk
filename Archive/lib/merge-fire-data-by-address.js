var fs = require('fs');
var async = require('async');

var propertyStreetSuffixByFireStreetSuffix = {
  'St': 'ST',
  'Ave': 'AV',
  '': '  ',
  'Blvd': 'BL',
  'Dr': 'DR',
  'Way': 'WY',
  'Ct': 'CT',
  'Rd': 'RD',
  'Pl': 'PL',
  'Ln': 'LN',
  'Hwy': 'HW',
  'Ter': 'TE',
  'Plz': 'PZ',
  'Aly': 'AL',
  'Cir': 'CR',
  'Walk': 'WK',
  'Sq': 'SQ'
};

function merge(properties, fireIncidents) {
  var fireIncidentAddressSet = fireIncidents.reduce(function(addressSet, fireIncident) {
    var addressKey = [
      fireIncident.street_number,
      fireIncident.street_name.toUpperCase(),
      propertyStreetSuffixByFireStreetSuffix[fireIncident.street_suffix],
    ].join('|');

    addressSet[addressKey] = true;
    return addressSet;
  }, {});


  properties.forEach(function(property, index, allRecords) {
    var startStreetNumber = Number(property['Start Street Number']);
    var endStreetNumber = Number(property['End Street Number']);

    var allPropertyStreetNumbers = [ startStreetNumber ];

    if(!isNaN(endStreetNumber)) {
      var currentStreetNumber = startStreetNumber + 2;
      while(currentStreetNumber <= endStreetNumber) {
        allPropertyStreetNumbers.push(currentStreetNumber);
        currentStreetNumber += 2;
      }
    }

    var hadFireIncidentAtProperty = false;
    allPropertyStreetNumbers.forEach(function(propertyStreetNumber) {
      var addressKey = [
        propertyStreetNumber,
        property['Street Name'],
        property['Street Suffix']
      ].join('|');
      if(fireIncidentAddressSet[addressKey]) {
        fireIncidentAddressSet[addressKey] = false;
        hadFireIncidentAtProperty = true;
      }
    });

    property['Had Fire Incident'] = hadFireIncidentAtProperty;
  });

  var unmatchedFireIncidentAddresses = Object.keys(fireIncidentAddressSet).filter(function(addressKey) {
    return fireIncidentAddressSet[addressKey];
  });

  console.error('Unmatched fire incident address count: %d', unmatchedFireIncidentAddresses.length);

  return properties;
}

function buildFileReads(fileNames) {
  return fileNames.map(function(fileName) {
    return function(cb) {
      fs.readFile(fileName, 'utf8', cb);
    };
  });
}

function main() {
  if(process.argv.length !== 4) {
    console.error('%s %s <property tax file json> <fire incidents file json>',
      process.argv[0], process.argv[1]);
    return;
  }

  var fileNames = [ process.argv[2], process.argv[3] ];
  async.parallel(buildFileReads(fileNames), function(err, results) {
    if(err) {
      console.error('Error reading files', err);
      return;
    }

    var labeledProperties = merge(JSON.parse(results[0]), JSON.parse(results[1]));
    console.log(JSON.stringify(labeledProperties));
  });
}

main();
