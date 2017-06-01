from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import numpy as np
import requests
import os
fname='address.csv'
url='http://data.sfgov.org/api/views/dxjs-vqsy/rows.csv?accessType=DOWNLOAD&api_foundry=true'
def get_data(url,to_file,overwrite=False):
    """opens sfgov url and writes csv of san francisco addresses to local.
    Read csv to pandas dataframe and return"""

    if not os.path.exists(to_file) or overwrite:
        """do download if local file doesn't exist or overwrite=True"""
        request=requests.get(url)
        with open('address.csv','w') as f:
            f.write(request.text)
    df=pd.read_csv(to_file)
    df['full_address'] = df.Address.apply(lambda x: x.strip()) + ' ' + df['Unit Number'].fillna('').apply(
        lambda x: str(x).strip()) + ' ' + df.Zipcode.apply(lambda x: str(x).strip())

    df['full_address'] = df['full_address'].apply(lambda x: x.replace('  ', ' ').strip())

    print (df.full_address)
    return df

def model_corpus(load=False,docs=None):
    """creates a vector representation of each address as a subset of all words in the address list and saves model and corpus.  If load =True, returns corpus and model from existing local files"""
    model=CountVectorizer(analyzer='char_wb',ngram_range=(2,3),min_df=1)
    if load and os.path.exists('model'):
        with open('model', 'rb') as f:
            model = pickle.load(f)
        with open('array', 'rb') as f:
            corpus = pickle.load(f)
    else:
        corpus=model.fit_transform(raw_documents=docs)
        with open('model', 'wb') as f:
            pickle.dump(model, f)
        with open('array', 'wb') as f:
            pickle.dump(corpus, f)
    return model,corpus


def find_address(search=[],how_many=1):
    """Given list of addresses to search, matches the closest address(es) from the database"""
    for s in search:
        vec=model.transform([s]).toarray()
        #index=np.argmax(cosine_similarity(corpus,vec))


        indices=np.argsort(cosine_similarity(corpus,vec).flatten())[::-1][:how_many]

        print('searching for: {}\n'.format(s),[('index={}, found={}'.format(i,df.full_address.loc[i])) for i in indices])

if __name__=='__main__':

    df=get_data(url=url,to_file=fname,overwrite=False)
    model,corpus=model_corpus(load=True,docs=df['full_address'])

    print (model.vocabulary_)

    search=['2351 29th street 94116','1495 25th ave','50 california 94111','5 california st','1 market','one market','16th ave','ocean ave','santa rose','1431 grant ave']
    find_address(search)