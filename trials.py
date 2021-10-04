## Moritz Bergemann 2021
## Main code for trialing different models/hyperparameters.
###############################
## ATTEMPT AT FULL FUNCTIONS ##
###############################

import pickle 
import datetime as dt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks, ClusterCentroids
from imblearn.pipeline import Pipeline as ImbPipeline


import data_prep

RANDOM_STATE = 123
# TODO don't be a perfectionist!

np.random.seed(0)

# Non Sklearn-adjustable hyperparams go in here!!!
def trial(model:str, param_grid:dict, drop_attributes:list, sampling:str, bins:int, verbosity=2, san_check=True, save_folder='trials'): # TODO pick h-params to adjust
    df_train, _ = data_prep.get_prepped_dataset(bins=bins)

    ## Input validation
    assert sampling in ['none', 'smote', 'random-under', 'smote-random-under', 'smote-enn', 'smote-tomek']
    assert model in ['knn', 'dt', 'nbayes', 'svm']
    if drop_attributes != None:
        assert set(drop_attributes) <= set(list(df_train.columns)) # Each col to drop must be in cols

    if not os.path.isdir(f"./{save_folder}"):
        os.path.mkdir(f"./{save_folder}")

    ## Prepare Dataset
    # Split into labels/not labels
    X = df_train.drop('Class', axis=1)
    y = df_train['Class']

    # Drop attributes to drop
    if drop_attributes != None and len(drop_attributes) != 0:
        X = X.drop(drop_attributes, axis=1)

    print("[i] Loaded dataset")

    # Encode labels
    y = LabelEncoder().fit_transform(y)

    ## Make data pipeline
    pipe_parts = []

    # Sampling strategy
    if sampling != 'none':
        if bins != None: # If everything binned
            if sampling == 'smote':
                pipe_parts.append(('smoten', SMOTEN(sampling_strategy=1)))
            elif sampling == 'random-under':
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.5)))
            elif sampling == 'smote-random-under':
                pipe_parts.append(('smoten', SMOTEN(sampling_strategy=0.45)))
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.6)))
        else: # Add Smote-Mixed if things not binned
            if sampling == 'smote':
                pipe_parts.append(('smotenc', SMOTENC(sampling_strategy=1)))
            elif sampling == 'random-under':
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.5)))
            elif sampling == 'smote-random-under':
                pipe_parts.append(('smotenc', SMOTENC(sampling_strategy=0.45)))
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.6)))

    # One-hot encoding
    if bins != None: # Use onehot encoder if everything binned
        categories = [list(X[col].cat.categories) for col in X.columns]
        pipe_parts.append(('onehot', OneHotEncoder(categories=categories)))
    else: # Use mixed encoder if not everything binned
        categorical_atts = X.select_dtypes(include='category').columns
        numeric_atts = X.select_dtypes(include=['int64', 'float64']).columns
        assert len(categorical_atts) + len(numeric_atts) == len(X.columns)

        categories = [list(X[col].cat.categories) for col in X[categorical_atts].columns]

        col_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(categories=categories), categorical_atts)
            ], # One-hot encode categorical attributes
            remainder='passthrough' # Let through numeric attributes
        )
        
        pipe_parts.append(('selective-onehot', col_transformer))
    
    # Model itself
    if model =='knn':
        pipe_parts.append(('model_knn', KNeighborsClassifier()))
    if model =='dt':
        pipe_parts.append(('model_dt', DecisionTreeClassifier()))
    if model =='nbayes':
        raise NotImplementedError
        pipe_parts.append('model_nbayes', ) # TODO
    if model =='svm':
        raise NotImplementedError
        pipe_parts.append('model_svm', ) # TODO

    # Make final pipeline
    pipe = ImbPipeline(pipe_parts)

    # Define K-fold sampling
    skfold = StratifiedKFold(n_splits=10)
    
    ## Suggested param_grid params:
    # knn - k_neighbors
    # dt - criterion, min_samples_split, min_samples_leaf
    # nbayes - TODO
    # svm - TODO

    grid_search = GridSearchCV(pipe, param_grid, cv=skfold, verbose=verbosity)

    print("[i] Starting train...")
    start_time = dt.datetime.now()
    grid_search.fit(X, y)
    print(f"[i] Training complete! Elapsed time - {(dt.datetime.now() - start_time).total_seconds()}s")

    ## Save results
    # Make save string
    if drop_attributes == None or len(drop_attributes) == 0:
        drop_save_string = "none"
    else:
        drop_save_string = f"[{','.join(drop_attributes)}]"
    save_path = f"./{save_folder}/m={model}-s={sampling}-b={bins},d={drop_save_string}.pickle"

    print(f"[i] Trial: Saving results to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(grid_search, f)
    
    print(f"Best Score: {grid_search.best_score_}")

    if san_check:
        print(f"Sanity Check: {sanity_check(grid_search, drop_attributes=drop_attributes, bins=bins)}")

    return grid_search

def sanity_check(result:GridSearchCV, drop_attributes, bins):
    ## Get the dataset
    df_train, _ = data_prep.get_prepped_dataset(bins=bins)

    ## Prepare Dataset
    # Split into labels/not labels
    X = df_train.drop('Class', axis=1)
    y = df_train['Class']

    # Drop attributes to drop
    if drop_attributes != None and len(drop_attributes) != 0:
        df_X = X.drop(drop_attributes, axis=1)

    # Encode labels
    y = LabelEncoder().fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

    estimator = result.best_estimator_
    estimator.fit(X_train, y_train)

    perf = accuracy_score(estimator.predict(X_val), y_val)

    return perf


def main():
    param_grid = {
        'model_knn__n_neighbors': [3, 5, 7, 9]
    }

    print("No Sampling Strategy")
    trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='none', bins=None, verbosity=0)

    print("Undersampling (Random)")
    trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=10, verbosity=0)

    print("Oversampling")
    trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=10, verbosity=0)

    print("SMOTE & Random Undersampling")
    trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=10, verbosity=0)


def main2():
    # OLD STUFF
    param_grid = {
        'model_dt__criterion': ['entropy'],
        'model_dt__min_samples_split': [2, 10, 40],
        'model_dt__min_samples_leaf': [1, 10, 20]
    }

    print("No Sampling Strategy")
    result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='none', bins=10, verbosity=0)
    print(result.best_score_)

    print("Undersampling (Random)")
    result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=10, verbosity=0)
    print(result.best_score_)

    print("Oversampling")
    result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=10, verbosity=0)
    print(result.best_score_)

    print("SMOTE & Random Undersampling")
    result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=10, verbosity=0)
    print(result.best_score_)

    param_grid = {
        'model_knn__n_neighbors': [3, 5, 7, 9]
    }

    print("No Sampling Strategy")
    result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='none', bins=10, verbosity=0)
    print(f"Best Score: {result.best_score_}")
    print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

    print("Undersampling (Random)")
    result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=10, verbosity=0)
    print(f"Best Score: {result.best_score_}")
    print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

    print("Oversampling")
    result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=10, verbosity=0)
    print(f"Best Score: {result.best_score_}")
    print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

    print("SMOTE & Random Undersampling")
    result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=10, verbosity=0)
    print(f"Best Score: {result.best_score_}")
    print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

    print("SMOTE & ENN Undersampling")
    result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-enn', bins=10, verbosity=0)
    print(f"Best Score: {result.best_score_}")
    print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

    print("SMOTE & Tomek Undersampling")
    result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-tomek', bins=10, verbosity=0)
    print(f"Best Score: {result.best_score_}")
    print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")


if __name__ == '__main__':
    main()