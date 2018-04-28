
# coding: utf-8

# In[633]:


import pandas as pd
import numpy as np
import os
import re


# #### Open and assess model output

# In[634]:


frame = pd.read_csv('model_output_120517.csv',header=0,encoding='latin1',low_memory=False)
frame = frame.drop(['Unnamed: 0','Unnamed: 0.1','Index'],axis=1)
frame.info()


# In[635]:


addresses = frame[frame.Address!='0 UNKNOWN']
addresses = frame.groupby('Address').count()


# *Ok, the issue here is that Model Output outputs a row for every* **incident** *not every address. But risk is predicted on the house level, right?*

# In[636]:


# Test: is there ever a case where the same building has different predicted probabilities?
multis = []

for i in addresses.index[addresses.EAS>1]:
    f = frame[frame.Address==i]
    if len(f['1'].unique()) > 1:
        multis.append(i)
        
print(multis)


# *Yes, risk is the same for each address, except for the unknowns, which we don't care about*

# *Just to check--are there any duplicate records of individual fire incidents?*

# In[637]:


multi_incidents = []

for i in addresses.index[addresses.EAS>1]:
    f = frame[frame.Address==i]
    if len(f['Incident Date'].unique()) < len(f):
          multi_incidents.append(i)
            
print(multi_incidents)


# In[638]:


# Let's check out some of these addresses
frame[frame.Address=='1995 OAKDALE AVE']
frame[frame.Address=='619 CLAYTON ST'].sort_values(by='Incident Date')
frame[frame.Address=='140 SOUTH VAN NESS AVE'].sort_values(by='Incident Date')
frame[frame.Address=='3151 FILLMORE ST'].sort_values(by='Incident Date')


# *Clearly there are cases where an identical incident (same date, same type) or very similar and possibly related incident (same date, different type) is recorded. It's difficult to know why these incidents are duplicated: data entry error? multiple related fires? Since this could bias our data, it would be a good thing to mention to the fire department in conversation at some point. In the mean time, I don't think we can accurately determine which duplicate incident records should be kept and which should be dropped, so we should leave them all in for modeling. Since all addresses have only 1 predicted risk value, we can drop these duplicate incidents for the purposes of mapping.*

# * Recommendation: for mapping, we just need each address and the average value of fire risk probability, the column labeled '1' in this dataset. No additional de-duping needed.

# #### Group data by address, keep first observation

# In[656]:


# Group data by Address; keep the first observation for each address; 
map_frame = frame[['Address','Neighborhood','x','y','1']].groupby('Address').first()
map_frame['Address'] = map_frame.index
map_frame = map_frame[map_frame.Address!='0 UNKNOWN']
map_frame = map_frame.rename(columns={'Address':'address','Neighborhood':'neighborhood','x':'longitude','y':'latitude','1':'fire_probability'})
map_frame = map_frame.set_index(np.arange(0,len(map_frame)))


# In[657]:


map_frame.info()
print("duplicates removed: ",len(frame)-len(map_frame))


# In[658]:


# check that there is one record per address
assert max(map_frame.address.value_counts()) == 1, "there is more than 1 record for some address(es)"


# #### Get rid of weird address formats

# In[659]:


# make copy of map_frame and apply function
map_frame_clean = map_frame.copy()


# In[660]:


# break address into a number column and a street column
map_frame_clean['number'] = map_frame_clean.address.str.extract(pat='(\d*\s*)')
map_frame_clean['street'] = map_frame_clean.address.str.extract(pat='\s+(\d*\D+)')


# In[661]:


to_remove = [u' ST', 
             u' AVE', 
             u' TER', 
             u' WAY', 
             u' CT', 
             u' DR', 
             u' RD', 
             u' WALK', 
             u' LN',
             u' BLVD',
             u' PL', 
             u' ROW', 
             u' CIR',
             u' ALY', 
             u' PARK', 
             u' STWY', 
             u' PLZ',
             u' HL', 
             u' HWY']

for i in range(0,len(to_remove)):
    map_frame_clean['street'] = map_frame_clean['street'].str.replace(pat=to_remove[i],repl='')


# In[667]:


map_frame_clean.head(20)


# In[646]:


assert len(map_frame_clean[map_frame_clean.number.isnull()]) == 0, "null values for number"
assert len(map_frame_clean[map_frame_clean.street.isnull()]) == 0, "null values for street"


# In[664]:


# define funtion:
    # find streets that include 0 before single-digit numerical street name (e.g. '06TH')
    # truncate the first 0
def fix_streets(street):
    street = str(street)
    pattern = re.search('(0\d*\D*)',street)
    if pattern is None:
        street = street  
    elif len(pattern.group(0)) > 0:
        street = pattern.group(0)[1:]
    else:
        "something is wrong with your string format"
    return street


# In[665]:


# replace rows containing '06TH' format in map_frame_clean with replacement
map_frame_clean['street'] = map_frame_clean.street.apply(lambda x:fix_streets(x))


# In[765]:


# delete records where number = '0'
map_frame_out = map_frame_clean[(map_frame_clean.number!='0 ')&(map_frame_clean.street!='TH')]


# #### Combine records in the same building into one record

# In[766]:


# combine number and street into one simplified address variable
# add ="" formatting to make sure excel doesn't read addresses as dates
map_frame_out['address'] = map_frame_out['number'].str.cat(map_frame_out['street'],sep=' ')
map_frame_out['address'] = map_frame_out['address'].apply(lambda x: '="' + str(x) + '"')

# I'm getting SettingWithCopy errors--but is this really sestting with copy? And it's sooooo slow with a loop


# In[767]:


# drop number and street columns
map_frame_out = map_frame_out.drop(['number','street'],axis=1)


# In[768]:


# group dataframe by simplified address, keeping the first observation for each
map_frame_out = map_frame_out.groupby('address').first()


# #### Checkout Output

# In[769]:


map_frame_out.tail()


# In[770]:


# check out my parents' house :p
map_frame_out.loc[u'="512  DEWEY"']


# In[771]:


map_frame_clean.info()


# In[772]:


print("unique addresses in original data: " + str(len(frame.Address.unique())))     
print("unique addresses in cleaned data: " + str(len(map_frame_out.index.unique())))
print("duplicates removed: " + str(len(frame.Address.unique())-len(map_frame_out.index.unique())))


# #### Write map_frame to csv for mapping

# In[773]:


if not os.path.exists('mapping_models'):
    os.mkdir('mapping_models')


# In[774]:


map_frame_out.to_csv(path_or_buf='mapping_models/map_layer.csv',sep=',',encoding='UTF-8')

