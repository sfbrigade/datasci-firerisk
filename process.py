import sys
from time import perf_counter
sys.setrecursionlimit(10000)
from fuzzywuzzy import process,fuzz
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
import numpy as np
import requests
import os
import math
from sklearn.neighbors import BallTree
pd.set_option("display.max_columns",101)
def download_file_or_not(url=None,local_file=None,overwrite=False):

    if not os.path.exists(local_file) or overwrite:
        """do download if local file doesn't exist or overwrite=True"""
        request=requests.get(url=url)
        with open(local_file,'w') as f:
            f.write(request.text)
    df=pd.read_csv(local_file)

    return df

def concatenate_address_fields_or_not(df):


    if not 'full_address' in df.columns:
        df['full_address'] = df.Address.apply(lambda x: x.strip()) + ' ' + df['Unit Number'].fillna('').apply(
            lambda x: str(x).strip()) + ' ' + df.Zipcode.apply(lambda x: str(x).strip())

        df['full_address'] = df['full_address'].apply(lambda x: x.replace('  ', ' ').strip())

    return df



def join_data(reference_path=None, target_path=None,save_path=None,radius=None,sample=0,head=False):


    reference=pd.read_csv(reference_path)
    target=pd.read_csv(target_path,low_memory=False)
    reference['LonRad'] = reference['Longitude'].apply(math.radians)
    reference['LatRad'] = reference['Latitude'].apply(math.radians)

    if 'Permit Address' in target.columns:
        target.rename(columns={'Permit Address':'Address'},inplace=True)


    if head:
        print(target.head(20))

    #target.dropna(subset=['Location'],inplace=True)
    #target.reset_index(drop=True,inplace=True)
    if 'Longitude' not in target.columns:

        target['Longitude']=target.Location.apply(lambda x: float(str(x[1:-1]).split(',')[1]) if not pd.isnull(x) else None)

        target['Latitude']=target.Location.apply(lambda x: float(str(x[1:-1]).split(',')[0]) if not pd.isnull(x) else None)

    target['LonRad']=target['Longitude'].apply(math.radians)
    target['LatRad']=target['Latitude'].apply(math.radians)

    start=perf_counter()

    k=BallTree(data=reference[['LonRad','LatRad']],metric='haversine')

    i=target#[target.index==108] #274171]

    if sample>0:
        i=i.sample(sample,replace=False)
    #[['Longitude','Latitude']].iloc[5]
    r_radians=radius/40075000*2*math.pi*.7
    # print(rows[['LonRad','LatRat']].values.reshape(1,-1).isfinite())

    # i['knn_indices_found']=i.apply(lambda rows:list(k.query_radius(rows[['LonRad','LatRad']].values.reshape(1,-1),r=r_radians)),axis=1)
    # i.assign(knn_indices_found=k.query_radius(i[['LonRad','LatRad']].values.reshape(1,-1),r=r_radians))

#    i.insert(len(i.columns),'knn_indices_found',k.query_radius(i[['LonRad','LatRad']].values,r=r_radians))

    i['knn_indices_found']=i.apply(lambda cols:list(k.query_radius(cols[['LonRad','LatRad']].values.reshape(1,2),r=r_radians)) if not pd.isnull(cols['LonRad']) else None,axis=1).apply(lambda x: x[0] if x is  not None else None)


    i['found_count']=i.knn_indices_found.apply(lambda x:len(x) if x is not None else None).values


    i['longitudes_found']=i.apply(lambda rows:list(reference['Longitude'].iloc[rows['knn_indices_found']]) if rows['knn_indices_found'] is not None else None,axis=1)
    i['latitudes_found']=i.apply(lambda rows:list(reference['Latitude'].iloc[rows['knn_indices_found']]) if rows['knn_indices_found'] is not None else None ,axis=1)



    i['addresses_found']=i.apply(lambda rows:list(reference['full_address'].iloc[rows['knn_indices_found']] ) if rows['knn_indices_found'] is not None else None,axis=1)

    # i.assign(most_likely=process.extractOne(i['Address'],choices=i['addresses_found']) if not i.Address.isnull().item() else None)
    #



    i['most_likely']=i.apply(lambda cols:process.extractOne(cols['Address'],choices=cols['addresses_found']
    if (not pd.isnull(cols['Address']) and cols['addresses_found'] is not None) else None),axis=1)


    i['most_likely_address']=i.apply(lambda cols: cols['most_likely'][0] if cols['most_likely'] is not None else None,axis=1)

    i['most_likely_index']=i.apply(lambda cols: cols['addresses_found'].index(cols['most_likely_address']) if (not pd.isnull(cols['Address']) and cols['most_likely_address'] is not None) else None,axis=1)

    i['most_likely_lon']=i.apply(lambda cols:cols['longitudes_found'][int(cols['most_likely_index'])] if not pd.isnull(cols['most_likely_index']) else None,axis=1)

    i['most_likely_lat'] = i.apply(lambda cols: cols['latitudes_found'][int(cols['most_likely_index'])] if not pd.isnull(cols['most_likely_index'])  else None,axis=1)



    i['most_likely_edit_ratio']=i['most_likely'].apply(lambda x: x[1] if x is not None else None)

    i['master_indices']=i.apply(lambda rows: int(rows['knn_indices_found'][rows['addresses_found'].index(rows['most_likely_address'])]) if rows['most_likely'] is not None else None,axis=1)
    i.to_csv(save_path)

    return (i,radius)

def plot_one(s):
    import math
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    i=s[0].sample(1)
    radius=s[1]
    print(i.columns)
    import seaborn as sns
    f,ax=plt.subplots()

    #ax.annotate(i['Address'].values[0],xy=(i['Longitude'],i['Latitude']),xytext=(i['Longitude'],i['Latitude']))
    up=1.000001
    down=.999999


    x=i['longitudes_found'].values[0]
    y=i['latitudes_found'].values[0]
    ax.scatter(x=x,y=y,marker='x')

    #ax.scatter(x=i['Longitude'],y=i['Latitude'],label=None)  #      ,label='Inspection Address:{}'.format(i['Address'].values[0]))
    circle=plt.Circle(xy=(i['Longitude'],i['Latitude']),radius=radius,color='r',fill=False)
    ax.add_artist(circle)
    ax.set_xlim(i['Longitude'].values+circle.radius,i['Longitude'].values-circle.radius)
    ax.set_ylim(i['Latitude'].values+circle.radius,i['Latitude'].values-circle.radius)

    best_index=int(i.most_likely_index)
    # print(i.addresses_found.iloc(best_index)))
    # [ax.annotate(address,xy=(x,y),xytext=(x,y),rotation=45) for address,x,y in zip(i.addresses_found.values[0],x,y)]
    ax.annotate('Finding: {}'.format(i['Address'].values[0]),xy=(x[best_index],y[best_index]),xycoords='data',
                xytext=(.5,1.05),textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05,width=1,connectionstyle='arc3,rad=.2'))

    ax.annotate('Found: {}'.format(i['most_likely_address'].values[0]),xy=(x[best_index],y[best_index]),xycoords='data',
                xytext=(x[best_index],y[best_index]),textcoords='data',
                )


    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
    plt.legend()
    plt.show()

def plot_goog(s,sample=1):
    import webbrowser
    """['Inspection Number', 'Inspection Type', 'Inspection Type Description',
       'Address', 'Inspection Address Zipcode', 'Battalion', 'Station Area',
       'Fire Prevention District', 'Billable Inspection',
       'Inspection Start Date', 'Inspection End Date', 'Inspection Status',
       'Return Date', 'Corrective Action Date', 'Referral Agency',
       'Complaint Number', 'Permit Number', 'Referral Number',
       'Violation Number', 'DBI Application Number', 'Invoice Date',
       'Second Notice Date', 'Final Notice Date', 'Lien Date',
       'Sent to Bureau of Delinquent Revenue', 'Invoice Amount', 'Fee',
       'Penalty Amount', 'Posting Fee', 'Interest Amount', 'Paid Amount',
       'Paid Date', 'Supervisor District', 'Neighborhood  District',
       'Location', 'Longitude', 'Latitude', 'knn_indices_found', 'found_count',
       'longitudes_found', 'latitudes_found', 'coordinates_found',
       'addresses_found', 'most_likely', 'most_likely_address',
       'most_likely_index', 'most_likely_edit_ratio', 'master_indices']"""
    from string import Template
    import math
    inspections=s[0]

    radius = s[1]

    with open('map_template.html','r') as f:
        template=Template(f.read())

    for plots in range(sample):
        i = inspections.sample(n=1,replace=False)

        o=.0003
        search_coord=('{{ lat:{},lng:{} }}'.format(i.Latitude.values[0],i.Longitude.values[0]))
        offset=('{{ lat:{},lng:{} }}'.format(i.Latitude.values[0]+o,i.Longitude.values[0]+o))
        target_address='"'+i['Address'].values[0]+'"'
        mle_address='"'+i['most_likely_address'].values[0]+'"'
        mle_lon=i['most_likely_lon'].values[0]
        mle_lat=i['most_likely_lat'].values[0]
        mle_coord = ('{{ lat:{},lng:{} }}'.format(mle_lat, mle_lon))

        x = i['longitudes_found'].values[0]
        y = i['latitudes_found'].values[0]

        lontext=['lng:']*len(x)
        lattext=['lat:']*len(y)

        coordinates='['+','.join(['{{{}{},{}{}}}'.format(lontext,x,lattext,y) for lontext,x,lattext,y in list(zip(lontext,x,lattext,y))])+']'

        labels = i['addresses_found'].values[0]
        l=i['addresses_found'].values
        title='Searching for {} \n -> Found {}'.format(target_address,mle_address).replace('"','')



        text=template.substitute(radius=radius,coordinates=coordinates,labels=labels,target_address=target_address,search_coord=search_coord,offset=offset,mle_address=mle_address,mle_coord=mle_coord,title=title)

        with open('map.html','w') as f:
            f.write(text)

        print(text)
        #webbrowser.open_new('map.html')

#plot_goog(join_data(radius=60))

def analyze(path=None):


    i=pd.read_csv(path)

    results=i[['Address','most_likely_address','most_likely_edit_ratio','master_indices','found_count','addresses_found']]

    print(results[results.isnull().any(axis=1)].count())


    end=perf_counter()

    #print('elapsed {} sec'.format((end-start)))
    print(results[['Address','most_likely_address','most_likely_edit_ratio']][(results.most_likely_edit_ratio<60) & (results.most_likely_edit_ratio<=100)].sort_values('most_likely_edit_ratio',ascending=False))

    import matplotlib.pyplot as plt
    import seaborn as sns
    f=plt.figure()

    sns.distplot(results['most_likely_edit_ratio'][results.notnull().all(axis=1)],kde=False,bins=20)
    plt.title='Histogram of edit ratios for {}'.format(target_file)
    plt.savefig('Histogram of edit ratios for {}.png'.format(target_file))

# i.to_csv(os.path.join(processed_data_directory,'Inspections_with_Master_Address.csv'))

if __name__ == '__main__':

    target_directory='../data/raw'
    processed_data_directory='../data/processed'

    reference='Addresses_-_Enterprise_Addressing_System_cleaned.csv'
    reference_path=os.path.join(processed_data_directory,reference )
    target_file='Police_Department_Incidents.csv'

    target_path=os.path.join(target_directory,target_file)

    save_path=os.path.join(processed_data_directory,'{}_with_matched.csv'.format(target_file.replace('.csv','')))

    def modify_business():
        df = pd.read_csv(os.path.join(data_directory, 'Registered_Business_Locations_-_San_Francisco.csv'))

        stuff=df['Business Location'].astype(str).apply(lambda x: x.split('\n'))
        df['Address']=stuff.apply(lambda x:x[0] if x is not None else None)
        df['Location']=stuff.apply(lambda x:x[2] if len(x)>2  else None)
        print(df.head(50))

        exit()
        print (df.Location.apply(lambda x:x is None).sum())
        exit()
        df = df.to_csv(os.path.join(data_directory, 'Registered_Business_Locations_-_San_Francisco.csv'))


    def modify_labeled():
        df=pd.read_csv(os.path.join(data_directory,'labeled-property-tax-addresses_060117.csv'))
        df['Zipcode of Parcel']=df.apply(lambda cols: int(cols['Zipcode of Parcel']) if not pd.isnull(cols['Zipcode of Parcel']) else '',axis=1)
        df['Address']=df.apply(lambda cols: '{} {} {} {}'.format(cols['Start Street Number'],cols['Street Name'],cols['Street Suffix'],cols['Zipcode of Parcel'] ),axis=1)

        df.to_csv(os.path.join(data_directory,'labeled-property-tax-addresses_060117.csv'))



    data=join_data(reference_path=reference_path,target_path=target_path,save_path=save_path,radius=40,sample=0,head=True)

    analyze(save_path)
    # plot_goog(data,2)

    exit()

































