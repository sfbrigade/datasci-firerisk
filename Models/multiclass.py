import pandas as pd


def XY_data(multiclass=False):
    #will process binary or multiclass

    k=pd.read_csv('masterdf_20170920.csv',low_memory=False,)


    # set target to Fire Incident Type
    y=k.pop('Incident_Cat')

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



    return x,y,unique

def Data_normalized(multiclass=False):

    x,y,unique=XY_data(multiclass=multiclass)


    x_dummies=pd.get_dummies(data=x[['Building_Cat','Neighborhood']],drop_first=True)




    # get quantitative features




    x_quantitative=x[['age','Num_Bathrooms', 'Num_Bedrooms',
           'Num_Rooms', 'Num_Stories', 'Num_Units', 'Land_Value',
           'Property_Area', 'Assessed_Improvement_Val', 'Tot_Rooms','Perc_Ownership' ,
                      'count potential fire control', 'count all complaints',
                      'count all complaints not corrected',
                      'count potential fire control not corrected',
                      'count fire emergency safety', 'count potential fire cause',
                      'count fire emergency safety not corrected',
                      'count potential fire cause not corrected'


                      ]]

    x_ids=x[['EAS','Address','Location_y']]
    #normalize quantitative features
    x_scaled=(x_quantitative-x_quantitative.mean())/(x_quantitative.max()-x_quantitative.min())


    #combine x dummies and x scaled data
    x_all=pd.concat([x_dummies,x_scaled],axis=1)



    return x_all,y,unique,x_ids


def classifier(train=True,x=None,y=None,target_names=None,class_weight=None,multiclass=False,plot=False,cross_val=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier

    # use multiclass random forest classifier for both binary and multiclass
    if multiclass:

        rf_model=OneVsRestClassifier(RandomForestClassifier(verbose=0,class_weight=class_weight),n_jobs=3)
    else:
        rf_model = RandomForestClassifier(verbose=0, class_weight=class_weight)

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
    from datetime import datetime

    print('model run time {}'.format(datetime.now()))
    print(classification_report(ytest,ypred,target_names=target_names))


    if multiclass == False:
        from sklearn.metrics import roc_curve

        fpr, tpr, thresh=roc_curve(ytest,rf_model.predict_proba(xtest)[:,1])

        import matplotlib.pyplot as plt
        import numpy as np
        plt.plot(fpr,tpr,linestyle='-')
        plt.plot([0,1],[0,1],linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Binary Class')
        if plot:
            plt.show()
        print([xtrain.columns[i] for i in np.argsort(rf_model.feature_importances_)[::-1]])

    from sklearn.model_selection import cross_val_score
    if cross_val:
        scores=cross_val_score(rf_model,X=x,y=y,cv=5)
        print('cross validation {}'.format(scores))
    return rf_model

def visualization_table(model=None,x=None,y=None,target_names=None):
    probs = pd.DataFrame(model.predict_proba(x), columns=target_names)
    predicts = pd.DataFrame(model.predict(x), columns=["prediction"])
    return pd.concat([x_ids, probs, predicts, y], axis=1)



if __name__ == '__main__':
    multiclass = False
    x,y,target_names,x_ids=Data_normalized(multiclass=multiclass)

    rf_model=classifier(train=True,x=x,y=y,target_names=target_names, class_weight=None,multiclass=multiclass,plot=False,cross_val=False)

    visualization_table=visualization_table(rf_model,x,y,target_names)
    visualization_table.to_csv("visualization_table.csv")