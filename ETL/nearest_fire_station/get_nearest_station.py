from ast import literal_eval
from bs4 import BeautifulSoup
from os.path import exists
from datetime import datetime
import time
import random
import requests
import numpy as np
import pandas as pd
import os
import pickle
import urllib2
from math import pi,sqrt,sin,cos,atan2
fire_station_address_dict = {}

def get_fire_station_addresses(URL):
    page = urllib2.urlopen(URL)
    soup = BeautifulSoup(page, "lxml")
    result_set = soup.find_all('div', attrs={"class" : "view-opensf-layout"})
    
    # list to store addresses in
    fire_station_adds = []
    for links in soup.find_all('a'):
        try:
            if 'propertymap' in links.get('href'):
                fire_station_adds.append(links.get('href').split("=")[2] + ", San Francisco")
        except TypeError: #omit empty results
            continue
    return fire_station_adds

def fetch_address_info(address_list, service='google', verbose=True, max_iter=5, timer=False):
    """
    Uses geopy iteratively until all addresses are stored.
    """
    def _fetch_geopy(address):
        try:
            tmp_result = geolocator.geocode(address)
        except Exception:
            tmp_result = []
        return tmp_result
    
    def _fetch_google(address):
        URL = "https://maps.googleapis.com/maps/api/geocode/json?address=" + address
        response = requests.get(URL)
        resp_json_payload = response.json()
        return resp_json_payload['results']
    
    address_dict = {}
    non_succ_set = list(address_list)
    iterations = 1
    len_counter = 1
    len_val = -1
    
    from geopy.geocoders import Nominatim
    geolocator = Nominatim()

    while non_succ_set:
        if len_val == len(non_succ_set):
            len_counter += 1
        len_val = len(non_succ_set)
        print "{} addresses in the queue (Iteration {})".format(len_val, iterations)
        for address in non_succ_set:
            fetch_verbose_string = "Fetching data for: " + address
            if service == 'google':
                address_dict[address] = _fetch_google(address)
            elif service == 'geopy':
                address_dict[address] = _fetch_geopy(address)
            else:
                raise AttributeError("You need to specify either 'google' or 'geopy' as service attribute.")
            if address_dict[address]:
                fetch_verbose_string += "\t\t\t ... successful"
                non_succ_set.remove(address)
            else:
                fetch_verbose_string += "\t\t\t ... not successful, queueing up again"
            if verbose:
                print fetch_verbose_string
            if timer:
                sleep_time = random.randint(2, 4) 
                time.sleep(sleep_time)
        iterations += 1
        if len_counter > max_iter-1:
            print "Termination: {} addresses couldn't be found".format(len_val)
            return address_dict
    return address_dict

def get_lat_long(address_dict):
    lat_lng_dict = {}
    for address in address_dict:
        lat_lng_dict[address] = [(address_dict[address][0]['geometry']['location']['lat'], 
                                 address_dict[address][0]['geometry']['location']['lng'])]
    return lat_lng_dict

def get_timestamp():
    local_time = datetime.now()
    return str(local_time.strftime("%Y%m%d"))

def get_nearest_fire_station(data):
    def _select_distance(row):
        return row[0]
    def _select_address(row):
        return row[1]
    def _select_latlong(row):
        return row[2]
    data = data.copy()
    
    data["next_fire_dpt_address"] = data["Location_y"].apply(hav_all)
    data["next_fire_dpt_distance"] = data["next_fire_dpt_address"].apply(_select_distance)
    data["next_fire_dpt_latlong"] = data["next_fire_dpt_address"].apply(_select_latlong)
    data["next_fire_dpt_address"] = data["next_fire_dpt_address"].apply(_select_address)
    
    return data

def haversine(pos1, pos2):
    pos2 = literal_eval(pos2)
    lat1 = float(pos1[0])
    long1 = float(pos1[1])
    lat2 = float(pos2[0])
    long2 = float(pos2[1])

    degree_to_rad = float(pi / 180.0)

    d_lat = (lat2 - lat1) * degree_to_rad
    d_long = (long2 - long1) * degree_to_rad

    a = pow(sin(d_lat / 2), 2) + cos(lat1 * degree_to_rad) * cos(lat2 * degree_to_rad) * pow(sin(d_long / 2), 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    km = 6367 * c

    return km

def hav_all(row):
    pos1 = literal_eval(row) # make sure the entry is a well formed tuple
    min_distance = 12742.0 # diameter of earth in km as maximum distance
    min_address = ""
    min_latlong = ""
    global fire_station_address_dict
    for fire_station in fire_station_address_dict:
        distance = haversine(pos1, fire_station_address_dict[fire_station])
        if distance < min_distance:
            min_address = fire_station
            min_distance = distance
            min_latlong = fire_station_address_dict[fire_station]
    return (min_distance, min_address, min_latlong)

def main():
    # I renamed the Google Drive Folder to /data/ in my repo - access will change once moving to database
    timestamp = get_timestamp()
    DATA_URL = '../data/' 
    FIRE_STATIONS_DATA = 'fire_stations_{}_andirs.csv'.format(timestamp)
    MASTER_FILE_NAME = 'masterdf_20170920.csv' # change this to recent
    NEW_FILE_NAME = 'nearest_fire_station_{}_andirs.csv'.format(timestamp)
    FIRE_STATION_URL = "http://sf-fire.org/fire-station-locations"
    recompute = False
    global fire_station_address_dict

    # Get fire stations
    if not exists(FIRE_STATIONS_DATA):
        fire_station_adds = get_fire_station_addresses(FIRE_STATION_URL)
        fire_station_address_dict = fetch_address_info(
            fire_station_adds, service='google', verbose=False, timer=True)
        fire_station_address_dict = get_lat_long(google_address_dict)
        fire_station_table = pd.DataFrame.from_dict(fire_station_address_dict, orient='index')
        fire_station_table.to_csv(FIRE_STATIONS_DATA, index=True)
    else:
        # load fire station data and turn into dict
        fire_station_address_dict = pd.DataFrame.to_dict(
            pd.read_csv(FIRE_STATIONS_DATA, index_col=0))['0']

    # Compute nearest fire station
    df = pd.read_csv(os.path.join(
        DATA_URL, MASTER_FILE_NAME), index_col=0, low_memory=False)
    df = get_nearest_fire_station(df)
    df = df[["EAS", "next_fire_dpt_address", "next_fire_dpt_distance", "next_fire_dpt_latlong"]]
    df.to_csv(NEW_FILE_NAME)

if __name__ == "__main__":
    main()