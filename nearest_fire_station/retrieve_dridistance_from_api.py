#!/usr/bin/python
from ast import literal_eval
import pandas as pd
import os
import pickle
import requests, json, pprint
import cPickle as pickle

import sys

DATA_URL = '../data/' 
NEW_FILE_NAME = 'masterdf_201710230_andirs.csv'

with open('private_keys/google_api_key', 'rb') as handle:
    GOOGLE_API_KEY = handle.readline()

def retrieve_driving_distance(data):
    """
    Limited to 50 requests per second. 
    """
    data = data.copy()
    data = data.drop_duplicates("EAS")
    result_store = {}
    for idx, value in data[["EAS", "Location_y", "next_fire_dpt_latlong"]].iterrows():
        value[1] = literal_eval(value[1]) # lat/long point 1
        value[2] = literal_eval(value[2]) # lat/long point 2
        origin = str(value[1][0]) + "," + str(value[1][1])
        destination = str(value[2][0]) + "," + str(value[2][1])
        GOOGLE_CALL_STRING = "https://maps.googleapis.com/maps/api/directions/json?origin={}&destination={}&key={}".format(
            origin, destination, GOOGLE_API_KEY)
        print "Working on {}".format(GOOGLE_CALL_STRING)
        r = requests.get(GOOGLE_CALL_STRING)
        if r.status_code == 200: # everything worked out fine
            d = r.json()
            if d["status"] == 'OVER_QUERY_LIMIT':
                print "EAS {} couldn't be loaded. (Code {})".format(
                    value[0], 'OVER_QUERY_LIMIT')
                break
            else:
                file_name = 'json_dumps/' + str(value[0])
                with open(file_name, 'wb') as handle:
                    pickle.dump(d, handle)
        else:
            print "EAS {} couldn't be loaded. (Code {})".format(
                value[0], r.status_code)

def count_files(path):
    counter = 0
    for filename in os.listdir(path):
        if '.DS' not in filename:
            counter += 1
    return counter

def main():
    current_index = count_files('json_dumps') if len(sys.argv) < 2 else int(sys.argv[1])
    step_size = 2500 if len(sys.argv) < 3 else int(sys.argv[2])
    print "Starting at index {} to retrieve the next {:,} entries.".format(
        current_index, step_size)

    print "Loading data and removing EAS duplicates."
    df = pd.read_csv(os.path.join(DATA_URL, NEW_FILE_NAME), low_memory=False)
    df = df.drop_duplicates("EAS")
    
    print "Retrieving data {} -> {}".format(current_index, current_index+step_size)
    retrieve_driving_distance(df[current_index:current_index+step_size])


if __name__ == '__main__':
    main()