import pandas as pd

def prepare_data(multiclass=False):
    #will process binary or multiclass

    k=pd.read_csv('kevin.csv')

    # set target to Fire Incident Type
    y=k.pop('Fire_Incident_Type')

    # assign classes
    # Nan becomes no incident
    # Values either become an incident or classes of incidents
    y=y.apply(lambda x:'0 No incident' if pd.isnull(x) else x if multiclass else '1 Incident')

    #store class labels
    unique=sorted(y.unique())

    #substitue class index number for class description
    y=y.apply(lambda x:unique.index(x))

    # set x to remaining data
    x=k
    #calculate property age
    x['age']=2016-x.Yr_Property_Built
    #create one-hot variables for property type and neighborhood
    x_dummies=pd.get_dummies(data=x[['Property_Code_Des','Neighborhood']],drop_first=True)

    # get quantitative features
    x_quantitative=x[['age','Num_Bathrooms', 'Num_Bedrooms',
           'Num_Rooms', 'Num_Stories', 'Num_Units', 'Land_Value',
           'Property_Area', 'Assessed_Improvement_Val', 'Tot_Rooms' ]]
    #normalize quantitative features
    x_scaled=(x_quantitative-x_quantitative.mean())/(x_quantitative.max()-x_quantitative.min())

    #combine x dummies and x scaled data
    x_all=pd.concat([x_dummies,x_scaled],axis=1)
    return x_all,y,unique


def classifier(train=True,x=None,y=None,target_names=None,class_weight=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier

    # use multiclass random forest classifier for both binary and multiclass
    rf_model=OneVsRestClassifier(RandomForestClassifier(verbose=1,class_weight=class_weight),n_jobs=3)

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.33)

    import pickle
    train=train

    if train: # run training and pickle model else just load model
        rf_model.fit(xtrain,ytrain)
        # output file name
        output=open('clf.pkl','wb')
        s = pickle.dump(rf_model,output)
    # load output file
    load=open('clf.pkl','rb')
    # return model
    rf_model = pickle.load(load)

    print('training accuracy {:.2f}'.format(rf_model.score(xtrain,ytrain)))

    print('testing accuracy {:.2f}'.format(rf_model.score(xtest,ytest)))

    ypred=rf_model.predict(xtest)
    ypred=pd.DataFrame(ypred)

    from sklearn.metrics import classification_report
    print('labels {}'.format(target_names))
    ytest=ytest.reset_index(drop=True)

    print(classification_report(ytest,ypred,target_names=target_names))



if __name__ == '__main__':

    x,y,target_names=prepare_data(multiclass=False)

    classifier(train=True,x=x,y=y,target_names=target_names, class_weight='balanced')
