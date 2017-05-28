var fs = require('fs');

var OPERATIONS_BY_FIELD_NAME = {
  'Start Street Number': 'GROUP BY',
  'End Street Number': 'GROUP BY',
  'Street Name': 'GROUP BY',
  'Street Suffix': 'GROUP BY',
  'Neighborhood Code Definition': 'GROUP BY',
  'Property Class Code Definition': 'GROUP BY',
  'Year Property Built': 'GROUP BY',
  'Number of Rooms': 'SUM',
  'Number of Stories': 'MAX',
  'Number of Units': 'SUM',
  'Property Area in Square Feet': 'SUM',
  'Lot Area': 'SUM',
  'Closed Roll Assessed Improvement Value': 'SUM',
  'Closed Roll Assessed Land Value': 'SUM',
  'Zipcode of Parcel': 'GROUP BY',
  'Latitude': 'AVG',
  'Longitude': 'AVG'
};

function buildAggregatedRecords(records) {
  var recordsByAddress = records.reduce(function(recordsByAddress, record) {
    var key = Object.keys(OPERATIONS_BY_FIELD_NAME).filter(function(fieldName) {
      return OPERATIONS_BY_FIELD_NAME[fieldName] === 'GROUP BY';
    }).map(function(fieldName) {
      return record[fieldName];
    }).join('|');
    recordsByAddress[key] = recordsByAddress[key] || [];
    recordsByAddress[key].push(record);
    return recordsByAddress;
  }, {});

  var aggregatedRecords = Object.keys(recordsByAddress).map(function(address) {
    var aggregatedRecord = recordsByAddress[address].reduce(function(aggregate, record, index, records) {
      return Object.keys(OPERATIONS_BY_FIELD_NAME).reduce(function(aggregate, fieldName) {
        var operation = OPERATIONS_BY_FIELD_NAME[fieldName];
        if(operation === 'GROUP BY') {
          aggregate[fieldName] = aggregate[fieldName] || record[fieldName].replace(/^[\s0]+|\s+$/g, '');
        } else {
          aggregate[fieldName] = aggregate[fieldName] || 0;
          var fieldValue = Number(record[fieldName]) || 0;
          switch(operation) {
            case 'SUM': aggregate[fieldName] += fieldValue; break;
            case 'AVG': aggregate[fieldName] += fieldValue / records.length; break;
            case 'MAX': aggregate[fieldName] = Math.max(fieldValue, aggregate[fieldName]); break;
            default:
          }
        }
        return aggregate;
      }, aggregate)
    }, {});
    return aggregatedRecord;
  });

  return aggregatedRecords;
}

function parseCoords(latLngString) {
    var pattern = /^\((.*),(.*)\)$/;
    var executionResult = pattern.exec(latLngString);
    return executionResult ? executionResult.slice(1, 3) : null;
}

function addAddressComponents(records) {
  return records.map(function(record) {
    record['Start Street Number'] = record['Property Location'].substring(5, 10);
    record['End Street Number'] = record['Property Location'].substring(0, 5);
    record['Street Name'] = record['Property Location'].substring(10, 30);
    record['Street Suffix'] = record['Property Location'].substring(30, 32);

    var latLng = parseCoords(record['Location']);
    record['Latitude'] = latLng ? latLng[0] : null;
    record['Longitude'] = latLng ? latLng[1] : null;

    return record;
  });
}

function parseTsv(tsvString) {
  var matrix = tsvString.split('\n').filter(function(line) {
    return line;
  }).map(function(line) {
    return line.split('\t');
  });

  var fields = matrix[0];
  var rows = matrix.slice(1);
  return rows.map(function(row) {
    var record = {};
    fields.forEach(function(field, index) {
      record[field] = row[index];
    });

    return record;
  });
}

function main() {
  if(process.argv.length !== 3) {
    console.error('%s %s <property tax file>',
    process.argv[0], process.argv[1]);
    return;
  }

  var fileName = process.argv[2];
  fs.readFile(fileName, 'utf8', function(err, result) {
    if(err) {
      console.error('Error reading files', err);
    }

    var tsv = parseTsv(result);
    var aggregatedRecords = buildAggregatedRecords(addAddressComponents(tsv));
    console.log(JSON.stringify(aggregatedRecords));
  });
}

main();
