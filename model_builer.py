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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import sklearn.calibration as calibration
pd.options.display.max_columns = 200


def data_preprocessing(df, target_col='Incident_Cat', drop_cat=False):
    '''
    Create target from df. NaN becomes no incident

    INPUTS:
    df - Pandas DataFrame, including target
    target_col - str, column name of target
    drop_cat - bool, False if keeping all categories in dummification

    OUTPUTS:
    y - Pandas DataFrame, target values only
    df_final, Pandas DataFrame, includes engineered features and dummified cols
    '''
    # quant_cols =['Num_Bathrooms', 'Num_Bedrooms',
    #        'Num_Rooms', 'Num_Stories', 'Num_Units', 'Land_Value',
    #        'Property_Area', 'Assessed_Improvement_Val', 'Tot_Rooms' ,'age']

    quant_cols =['Num_Stories', 'Num_Units', 'Land_Value',
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
    '''
    Update score dictionary w/ various metrics
    '''
    l_loss = log_loss(y_true, predicted_probs)
    brier_loss = brier_score_loss(y_true, predicted_probs)
    f1 = f1_score(y_true, y_pred)

    for key, val in [('log_loss', l_loss),
                     ('brier_loss', brier_loss),
                     ('f1', f1)]:
        print 'Adding {} for {}'.format(key, model_key)
        score_dict[model_key][key].append(val)

    return score_dict


def prob_calibration_cross_val(model, X_train, y_train, n_folds=5):
    '''
    cross validate, includes probaility calibration
    '''
    # Initialize dicttionary of scores and kfold obj
    model_names = ['base', 'isotonic', 'sigmoidal']
    score_dict = {key : defaultdict(list) for key in model_names}
    kfold = KFold(n_splits=n_folds)

    ind = 0
    for train_index, test_index in kfold.split(X_train):
        ind += 1
        print '\n'
        print 'Fold {}'.format(ind)

        X_training, X_val = X_train[train_index], X_train[test_index]
        y_training, y_val = y_train[train_index], y_train[test_index]

        print 'Fitting base model'
        model.fit(X_training, y_training)

        # TODO functionalize predicts, predict_probs
        # TODO functionalize sigmoid fit and isotonic fit
        predicts = model.predict(X_val)
        predict_probs = model.predict_proba(X_val)[:, 1]
        print 'Updating score dictionary with base scores'
        score_dict = add_scores(score_dict, 'base',
                                y_val, predicts, predict_probs)

        # isotonic
        print 'Fitting isotonic calibration'
        iso = calibration._CalibratedClassifier(model, method='isotonic')
        iso.fit(X_training, y_training)
        iso_probs = iso.predict_proba(X_val)[:, 1]
        iso_preds = iso_probs > 0.5
        print 'updating score dict w/ isotonic scores'
        score_dict = add_scores(score_dict, 'isotonic',
                                y_val, iso_preds, iso_probs)

        # sigmoid / Platt Scaling
        print 'Fitting sigmoidal calibration'
        sigmoid = calibration._CalibratedClassifier(model, method='sigmoid')
        sigmoid.fit(X_training, y_training)
        sigmoid_preds = sigmoid.predict(X_val)
        sigmoid_probs = sigmoid.predict_proba(X_val)[:, 1]
        print 'updating score dict w/ sigmoid scores'
        score_dict = add_scores(score_dict, 'sigmoidal',
                                y_val, sigmoid_preds, sigmoid_probs)


    return score_dict


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print 'Please enter file location'

    df = pd.read_csv('data/masterdf_20170920.csv', low_memory=False)

    y, X = data_preprocessing(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    X_train = X_train.reset_index().values[:, 1:] #removes added col
    X_test = X_test.reset_index().values[:, 1:]
    y_train = y_train.values
    y_test = y_test.values

    # can change model to anything
    # TODO make this take any model?
    # TODO define use case for this

    rf = RandomForestClassifier(n_estimators=10, n_jobs=3)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', rf)])

    # Create dictionary of scores w/ and w/o probability calibration
    score_dict = prob_calibration_cross_val(pipe, X_train, y_train)

    # fit rf to whole data, score, print feature importances
    # TODO functionalize, shouldn't be using test set yet
    pipe.fit(X_train, y_train)
    feat_imp = pipe.named_steps['model'].feature_importances_
    cols = X.columns
    col_fi = sorted(zip(cols, feat_imp), key = lambda x: x[1])

    print '\n\n'
    print 'Feature Importances: '
    for col, fi, in col_fi:
        print col, fi


    # construct reliability curve
    # TODO create calibration set
    # TODO functionalize
    # TODO move to different script

    plt.figure(figsize=(12,8))
    # base probs
    base_probs = pipe.predict_proba(X_test)[:,1]
    base_pred = pipe.predict(X_test)
    base_pos_frac, base_mean_predict = (
        calibration_curve(y_test, base_probs, n_bins=10)
                                       )

    plt.plot(base_mean_predict, base_pos_frac, 'o-', label='base')

    # Platt scaling
    sigmoid = CalibratedClassifierCV(pipe, method='sigmoid', cv=10)
    sigmoid.fit(X_train, y_train)
    sigmoid_probs = sigmoid.predict_proba(X_test)[:, 1]
    sigmoid_preds = sigmoid.predict(X_test)
    sigmoid_pos_frac, sigmoid_mean_predict = (
        calibration_curve(y_test, sigmoid_probs, n_bins=10)
                                             )

    plt.plot(sigmoid_mean_predict, sigmoid_pos_frac, 's-', label='sigmoid')

    # isotonic scaling
    iso = CalibratedClassifierCV(pipe, method='isotonic', cv=10)
    iso.fit(X_train, y_train)
    iso_probs = iso.predict_proba(X_test)[:, 1]
    iso_preds = iso.predict(X_test)
    iso_pos_frac, iso_mean_predict = (
        calibration_curve(y_test, iso_probs, n_bins=10)
                                             )

    plt.plot(iso_mean_predict, iso_pos_frac, 's-', label='isotonic')

    # plot perfrect line
    perfect = np.linspace(0, 1, 10)
    plt.plot(perfect, perfect, '--', label='perfect')


    model_names = ['base', 'isotonic', 'sigmoidal']
    score_dict = {key : defaultdict(list) for key in model_names}

    score_dict = add_scores(score_dict, 'base', y_test, base_pred, base_probs)
    score_dict = add_scores(score_dict, 'sigmoidal', y_test, sigmoid_preds, sigmoid_probs)
    score_dict = add_scores(score_dict, 'isotonic', y_test, iso_preds, iso_probs)

    for key, value in score_dict.iteritems():
        print key, value
    plt.legend()
    plt.show()

    print '\n\n'
    print 'script finished'
    df.columns

    ### ADDING SIGMOID PROBS
    # TODO  functionalize

    all_sigmoid_probs = sigmoid.predict_proba(X)[:, 1]
    df['calibrated_probs'] = all_sigmoid_probs
    df.drop_duplicates(subset=['EAS'], inplace=True)
    len(df)
    df.to_csv('master_0920_calibrated_no_dupes.csv')
    
