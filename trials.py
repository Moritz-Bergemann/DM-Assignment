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
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE, SMOTEN
from imblearn.pipeline import Pipeline as ImbPipeline


import data_prep

RANDOM_STATE = 123
# TODO don't be a perfectionist!

# Non Sklearn-adjustable hyperparams go in here!!!
def trial(model:str, drop_attributes:list, sampling:str, bins:int): # TODO pick h-params to adjust
    df_train, _ = data_prep.get_prepped_dataset(bins=bins)

    ## Input validation
    assert sampling in ['none', 'over', 'under', 'both']
    assert model in ['knn', 'dt', 'nbayes', 'svm']
    if drop_attributes != None:
        assert set(drop_attributes) <= set(list(df_train.columns)) # Each col to drop must be in cols

    if not os.path.isdir("./trials"):
        os.path.mkdir("./trials")

    ## Prepare Dataset
    # Split into labels/not labels
    df_X = df_train.drop('Class', axis=1)
    df_y = df_train['Class']

    # Drop attributes to drop
    if drop_attributes != None and len(drop_attributes) != 0:
        df_X = df_X.drop(drop_attributes, axis=1)

    X = df_X.to_numpy()
    y = df_y.to_numpy()

    print("[i] Loaded dataset")

    # Encode labels
    y = LabelEncoder().fit_transform(y)

    ## Make data pipeline
    pipe_parts = []

    # Sampling strategy
    if bins != None: # If everything binned
        if sampling == 'over':
            pipe_parts.append(('smoten', SMOTEN(random_state=RANDOM_STATE, sampling_strategy='not majority')))
        elif sampling == 'under':
            raise NotImplementedError() # TODO
        elif sampling == 'both':
            raise NotImplementedError() # TODO
    else: # Add Smote-Mixed if things not binned
        raise NotImplementedError() # TODO
    
    # One-hot encoding
    if bins != None: # Use onehot encoder if everything binned
        categories = [list(df_X[col].cat.categories) for col in df_X.columns]
        pipe_parts.append(('onehot', OneHotEncoder(categories=categories)))
    else: # Use mixed encoder if not everything binned
        raise NotImplementedError() # TODO
    
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

    ## Add grid search feature selection
    if model =='knn':
        param_grid = {
            # All odd values for k-neighbors between 3 & 22
            'model_knn__k_neighbors': [ii for ii in range(3, 22, 2)]
        }
    if model =='dt':
        param_grid = {
            'model_dt__criterion': ['gini', 'entropy'],
            'model_dt__min_samples_split': [ii for ii in range(2, 41)],
            'model_dt__min_samples_leaf': [ii for ii in range(1, 21)]
        }
    if model =='nbayes':
        param_grid = {
        }
    if model =='svm':
        raise NotImplementedError
        param_grid = {

        }

    # Define K-fold sampling
    skfold = StratifiedKFold(n_splits=10)
    
    grid_search = GridSearchCV(pipe, param_grid, cv=skfold, verbose=2)

    print("[i] Starting train...")
    start_time = dt.datetime.now()
    grid_search.fit(X, y)
    print(f"[i] Training complete! Elapsed time - {(dt.datetime.now() - start_time).total_seconds}s")

    # Make save string
    if drop_attributes == None or len(drop_attributes) == 0:
        drop_save_string = "none"
    else:
        drop_save_string = f"[{','.join(drop_attributes)}]"
    save_path = f"./trials/m={model}-s={sampling}-b={bins},d={drop_save_string}.pickle"

    print(f"[i] Trial: Saving results to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(grid_search, f)

def main():
    trial(model='dt', drop_attributes=None, sampling='over', bins=10)

if __name__ == '__main__':
    main()