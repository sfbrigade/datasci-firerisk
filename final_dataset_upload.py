
# coding: utf-8

# ## MapBox API Upload

# *Program to transform most recent model output to GeoJSON format and upload new and edited rows to MapBox via API*

# #### Read in required packages

# In[2]:


import requests
from requests import Request, Session
import json
from uritemplate import URITemplate, expand
import pandas as pd
import numpy as np
import time
from datetime import date
import random
import mapbox
from mapbox import Datasets
import os


# #### Define function to initialize Datasets API

# In[3]:


def initialize_api():
    # initialize API and SDK
    try:
        username = 'fire-risk-sf'
        path = 'mapping_models/map_layer.csv'
        dataset_to_update = 'cjeg82ksu066z2yogjy65cbq0'
        print("succesfully authorized")
        return username, path, dataset_to_update
    except:
        print("please make sure you have initialized the Mapbox Python SDK according to the documentation")


# #### Define function to import deduplicated model output 
# *(this will be a call to CFA database down the line; for now just read in a csv)*

# In[4]:


def get_upload_data(path):
    try:
        map_data = pd.DataFrame(pd.read_csv(path))
        print("successfully accessed local model data")
        return map_data
    except:
        print("Error: could not access local model data")


# #### Define function to select rows to upload

# In[6]:


# get features of dataset, save to pandas dataframe
# outer join output and features
# return dataframe of non-overlapping features to upload

def select_upload_data(dataset_to_update, dataframe):

    try:
        # initialize an empty list to hold the features that we have already uploaded to mapbox
        existing = []

        # initialize API session
        datasets = Datasets()
        os.environ['MAPBOX_ACCESS_TOKEN'] = 'sk.eyJ1IjoiZmlyZS1yaXNrLXNmIiwiYSI6ImNqZGw1dTlwdDA1aXMzM3FrbDZpZnpmczMifQ.JbUPsvx8384WHOAfA9Vy9w'

        # get features of the mapbox dataset by calling the API
        features_collection = datasets.list_features(dataset_to_update).json()

        # loop through the features of the mapbox dataset and save them to the list "existing"
        for i in range(0,len(features_collection['features'])):
            existing.append([
                features_collection['features'][i]["id"],
                features_collection['features'][i]["properties"]["name"],
            ])

        # convert "existing" to a dataframe
        existing = pd.DataFrame(existing,columns=["id","address"])
        
        print("succesfully accessed the existing mapbox dataset")
        
    except:
        
        print("Error: could not access the mapbox dataset. Check that you are accessing your token correctly")

    try:
        # get rows in dataframe that are not yet uploaded
        new_rows = pd.merge(left = dataframe,
                            right = existing,
                            on = "address",
                            how = "left")
        new_rows = new_rows[new_rows.id.isnull() == True]
        new_rows = new_rows.set_index(np.arange(0,len(new_rows)))

        # create a "start point" for new row ids
        # note: this was not a very smart way to creat ids, but I'm stuck with it for now
        start = max(existing["id"].astype("int64"))

        # create new random ids for these new uploads
        new_ids = random.sample(range(start,start*5),len(new_rows))
        
        print("succesfully selected new upload data")
    
    except:
        
        print("there was a problem uploading the data")
        
    # make sure these new random ids are not already in the dataset
    
    try:
        [new_id not in existing["id"] for new_id in new_ids]
    except:
        "Error! Duplicate ids created"
        
    # return the new rows and new ids to upload
    return new_rows, new_ids
    


# #### Define function to convert upload data to geojson

# In[7]:


def convert_to_geojson(dataframe,id_list):
    
    try: 
        # initialize empty geojson collection
        output = {}
        output["type"] = "FeatureCollection"
        feature_collection = []

        #id_list = random.sample(range(1,len(dataframe)*10),len(dataframe))

        # iterate through model output and convert file types
        for i in range(0,len(dataframe)):
            layer = {}
            layer["id"] = str(id_list[i])
            layer["type"] = "Feature"
            layer["properties"] = {}
            layer["properties"]["name"] = dataframe.address.loc[i]
            layer["properties"]["fire_probability"] = dataframe.fire_probability.loc[i]
            layer["geometry"] = {}
            layer["geometry"]["type"] = "Point"
            layer["geometry"]["coordinates"] = [dataframe.longitude.loc[i],dataframe.latitude.loc[i]]
            feature_collection.append(layer)
        output["features"] = feature_collection

        # export new geojson data to a file with formate "map_layer_geojson_today's date.json
        filename = "map_layer_geojson_" + date.strftime(date.today(),format="%m%d%Y") + '.json'
        with open(filename,'w') as outfile:
            json.dump(output,outfile)

        # check that there are no duplicate id's
        ids = []
        for i in range(0,len(output['features'])):
            ids.append(output['features'][i]['id'])
        assert len(ids) == len(set(ids)), "duplicate ids"

        return output
    
    except:
        
        print("Error: could not convert the new data to GeoJSON. Are all required field included?")


# #### Define function to call API and upload dataset

# In[9]:


def upload_to_mapbox(upload, dataset_to_update):
    
    try:
        # initialize api
        datasets = Datasets()

        os.environ['MAPBOX_ACCESS_TOKEN'] = 'sk.eyJ1IjoiZmlyZS1yaXNrLXNmIiwiYSI6ImNqZGw1dTlwdDA1aXMzM3FrbDZpZnpmczMifQ.JbUPsvx8384WHOAfA9Vy9w'

        # upload features to new dataset
        for i in range(0,len(upload['features'])):
            feature = upload['features'][i]
            index = upload['features'][i]['id']
            resp = datasets.update_feature(dataset = dataset_to_update,
                                           fid = index,
                                           feature = feature)
    except:
        
        print("Error: could not upload the new data to Mapbox. Is there a problem with the access token?")


# #### Define function to check the success of upload

# In[10]:


def check_upload_result(upload, dataset_to_update):
    try:
        # check that the dataset upload was successful
        datasets = Datasets()
        os.environ['MAPBOX_ACCESS_TOKEN'] = 'sk.eyJ1IjoiZmlyZS1yaXNrLXNmIiwiYSI6ImNqZGw1dTlwdDA1aXMzM3FrbDZpZnpmczMifQ.JbUPsvx8384WHOAfA9Vy9w'
        resp = datasets.list_features(dataset_to_update).json()
        assert len(resp['features']) == len(upload), "Only " + str(len(resp['features'])) + " features are included in the dataset"    
    except:
        
        print("Error: unable to check the number of uploaded records. To manually check the execution: Mapbox > Studio > Datasets > Menu > View details")
        


# #### Define function to initiate upload

# In[11]:


def __init__():
    # initialize api and values
    username, path, dataset_to_update = initialize_api() 
    # get model output from local storage
    dataframe = get_upload_data(path)
    # select the data that still needs to be uploaded 
    new_rows, new_ids = select_upload_data(dataframe, dataset_to_update)
    # convert data that needs to be uploaded to geojson
    upload = convert_to_geojson(dataframe = new_rows, id_list = new_ids)
    # upload data
    upload_to_mapbox(upload, dataset_to_update)
    # check how many rows we successfully added
    check_upload_result(upload, dataset_to_update)


# #### Set parameter values and call API

# In[12]:


__init__()


# In[ ]:


username, path, dataset_to_update = initialize_api()
dataframe = get_upload_data(path)
new_rows, new_ids = select_upload_data(dataset_to_update, dataframe)
upload = convert_to_geojson(new_rows,new_ids)
upload_to_mapbox(upload, dataset_to_update)


# In[ ]:


check_upload_result(upload,dataset_to_update)


# In[16]:


len(new_ids)

