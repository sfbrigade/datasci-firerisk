import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

pd.options.display.max_columns = 200


def data_preprocessing(df, target_col='Incident_Cat', drop_cat=False):
    '''
    Create target from df. NaN becomes no incident

    INPUT: df
    OUTPUT: df, df features and target
    '''
    quant_cols =['Num_Bathrooms', 'Num_Bedrooms',
           'Num_Rooms', 'Num_Stories', 'Num_Units', 'Land_Value',
           'Property_Area', 'Assessed_Improvement_Val', 'Tot_Rooms' ,'age']

    cat_cols = ['Building_Cat','Neighborhood']

    # feature engineering
    df['age'] = 2016 - df['Yr_Property_Built']

    # dummification
    dummies = pd.get_dummies(df[cat_cols], drop_first=drop_cat)

    # target creation
    df[target_col] = df[target_col].notnull()
    y = df.pop(target_col)

    # final df
    df = df.loc[:, quant_cols]
    df_final = pd.concat([df, dummies], axis=1)

    return y, df_final

def add_scores(score_dict, model_key, y_true, y_pred, predicted_probs):
    l_loss = log_loss(y_true, predict_probs)
    brier_loss = brier_score_loss(y_true, predicted_probs)
    f1 = f1_score(y_true, y_pred)

    for key, val in [('log_loss', l_loss),
                     ('brier_loss', brier_loss),
                     ('f1', f1)]:

        score_dict[model_key][key].append(val)

    return score_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Please enter file location'

    df = pd.read_csv('data/masterdf_20170920.csv', low_memory=False)

    y, X = data_preprocessing(df)


    for col in X.columns:
        print col

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, test_size=0.33)
    X_train = X_train.reset_index().values
    X_test = X_test.reset_index().values
    y_train = y_train.values
    y_test = y_test.values

    # can change model to anything
    rf = RandomForestClassifier(n_estimators=10, n_jobs=3)
    logit = LogisticRegression()
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', rf)])

    kfold = KFold(n_splits=5)

    model_names = ['base', 'isotonic', 'sigmoidal']
    score_dict = {key : defaultdict(list) for key in model_names}

    ind = 0
    for train_index, test_index in kfold.split(X_train):
        ind += 1
        print 'Fold {}'.format(ind)
        X_training, X_val = X_train[train_index], X_train[test_index]
        y_training, y_val = y_train[train_index], y_train[test_index]

        print 'Fitting base model'
        pipe.fit(X_train, y_train)
        predicts = pipe.predict(X_val)
        predict_probs = pipe.predict_proba(X_val)[:, 1]
        print 'Updating score dictionary with base scores'
        score_dict = add_scores(score_dict, 'base', y_val, predicts, predict_probs)

        # isotonic
        print 'Fitting isotonic calibration'
        iso = CalibratedClassifierCV(pipe, method='isotonic', cv=3)
        iso.fit(X_train, y_train)
        iso_preds = iso.predict(X_val)
        iso_probs = iso.predict_proba(X_val)[:, 1]
        print 'updating score dict w/ isotonic scores'
        score_dict = add_scores(score_dict, 'isotonic', y_val, iso_preds, iso_probs)

        # sigmoid / Platt Scaling
        print 'Fitting sigmoidal calibration'
        sigmoid = CalibratedClassifierCV(pipe, method='sigmoid', cv=3)
        sigmoid.fit(X_train, y_train)
        sigmoid_preds = sigmoid.predict(X_val)
        sigmoid_probs = sigmoid.predict_proba(X_val)[:, 1]
        print 'updating score dict w/ sigmoid scores'
        score_dict = add_scores(score_dict, 'sigmoidal', y_val, sigmoid_preds, sigmoid_probs)

    print 'finished cross validation'

    pipe.fit(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print 'test_score: {}'.format(test_score)

    feat_imp = pipe.named_steps['model'].feature_importances_
    cols = X.columns

    col_fi = sorted(zip(cols, feat_imp), key = lambda x: x[1])

    for col, fi, in col_fi:
        print col, fi
    # import data
    # process data
    # cross validate
    # train model
    # train calibration
