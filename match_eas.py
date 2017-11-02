import sys
sys.setrecursionlimit(10000)
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import BallTree

pd.set_option("display.max_columns", 101)


def join_data_on_address_GPS(radius=40, df=None):
    # uses Addresses_-_Enterprise_Addressing_System.csv' as the reference address table
    # df is the table to be linked by closest lon/lat and address.
    # returns data frame of matched EAS, Address in EAS, and Match Score
    reference = pd.read_csv('./raw_csv/Addresses_-_Enterprise_Addressing_System.csv')
    #convert Lon/Lat to radians
    reference['LonRad'] = reference['Longitude'].apply(math.radians)
    reference['LatRad'] = reference['Latitude'].apply(math.radians)

    if 'Permit Address' in df.columns:
        df.rename(columns={'Permit Address': 'Address'}, inplace=True)

    class r_closest_EAS(BallTree):
        # given an address and lon/lat in the data, uses the Balltree to find the closest lon/lat locations in the EAS reference table.
        def __init__(self, reference=None, *args, **kwargs):
            #reference EAS table
            self.reference = reference
            # table of interest
            data = reference[['LonRad', 'LatRad']].values
            #initializes table as data
            super(r_closest_EAS, self).__init__(data=data, *args, **kwargs)

        def search_around(self, lon=None, lat=None, address=None, radius=None):
            if (lon or lat or address) == None:
                print('missing variables')
                return
            # given lon/lat as array and radius, return indices of addresses in EAS
            indices = self.query_radius(np.array([lon, lat]).reshape(1, -1), r=radius)
            indices = indices[0].tolist()

            #return rows in reference address that are found in the query_radius
            found_places = self.reference.iloc[indices]

            #get list of addresses from found_places
            found_addresses = found_places['Address'].values.tolist()
            if found_addresses == []:
                # print('no address found')
                return

            #using process function from fuzzy wuzzy (edit distance) to return closest text match in EAS to the address of interest.
            closest_address, score = process.extractOne(query=address, choices=found_addresses)
            # get index in list
            closest_index = found_addresses.index(closest_address)
            # get row of index in dataframe
            closest_place = found_places.iloc[closest_index]
            # return EAS ID of that row
            closest_eas = closest_place['EAS BaseID']

            if closest_eas is None:
                print('None found')
            return {'EAS': closest_eas, 'Address': closest_address, 'Score': score}

    df['LonRad'] = df['Longitude'].apply(math.radians)
    df['LatRad'] = df['Latitude'].apply(math.radians)
    r_radians = radius / 40075000 * 2 * math.pi * .7
    #instantiate BallTree object with reference EAS table.
    k = r_closest_EAS(reference=reference, metric='haversine')

    #perform columwise operation with k using columns as inputs from data.
    eas_match = df.apply(
        lambda cols: k.search_around(cols['LonRad'], cols['LatRad'], cols['Address'], radius=r_radians), axis=1)

    #return match dataframe
    return eas_match
