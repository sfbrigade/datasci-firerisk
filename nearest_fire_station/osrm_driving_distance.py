from ast import literal_eval
import pandas as pd
import requests, json, pprint
import sys
data_length = 0
counter = 0

def progress(progress, dlen):
    sys.stdout.write('\r%s / %s' % (progress, dlen))
    sys.stdout.flush()  # As suggested by Rom Ruben

def get_distance_osrm_eas(row):
    # sample request url
    # "http://127.0.0.1:5000/route/v1/driving/13.388860,52.517037;13.385983,52.496891?steps=false"
    global counter, data_length
    server_ip_and_port = "127.0.0.1:5000"
    counter += 1 # to determine progress
    eas = row[0]
    pos1 = literal_eval(row[1])
    pos1_lat = str(pos1[0])
    pos1_lng = str(pos1[1])
    
    pos2 = literal_eval(row[2])
    pos2_lat = str(pos2[0])
    pos2_lng = str(pos2[1])
    
    req_string = "http://" + server_ip_and_port+ "/route/v1/driving/"
    req_string += pos1_lng + "," + pos1_lat + ";" + pos2_lng + "," + pos2_lat + "?steps=false"
    r = requests.get(req_string)
    d = r.json()
    write_string = str(eas) + "," + str(d['routes'][0]['distance']) + "\n"
    with open("eas_driving_distance.csv", 'a+') as handle:
        handle.write(write_string)
    # show progress
    progress(counter, data_length)
    #return [eas, d['routes'][0]['distance']]
    #return pos1, pos2, row[0]

def main():
    global data_length, counter
    print "Loading data"
    df = pd.read_csv(
        "../data/masterdf_20170920.csv", low_memory=False, index_col=0)
    df = pd.merge(df, pd.read_csv(
        "eas_haversine_distance.csv", index_col=0), on="EAS")
    df = df[["EAS", "Location_y", "next_fire_dpt_latlong"]]
    df.drop_duplicates("EAS", inplace=True)
    data_length = len(df)
    print "Start retrieving distances"
    df.apply(get_distance_osrm_eas, axis=1)

if __name__ == "__main__":
    main()