## Moritz Bergemann 2021
## Main code for trialing different models/hyperparameters.
###############################
## ATTEMPT AT FULL FUNCTIONS ##
###############################

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE, SMOTEN
from imblearn.pipeline import Pipeline as ImbPipeline


import data_prep

RANDOM_STATE = 123
# TODO don't be a perfectionist!

# Non Sklearn-adjustable hyperparams go in here!!!
def trial(model:str, drop_attributes:list, sampling:str, bins:int, cat_binning_threshold:int): # TODO pick h-params to adjust
    df_train, _ = data_prep.get_prepped_dataset(bins=bins, cat_binning_threshold=cat_binning_threshold)

    # Input validation
    assert sampling in ['none', 'over', 'under', 'both']
    assert model in ['knn', 'dt', 'nbayes', 'svm']
    assert set(drop_attributes) <= set(list(df_train.columns)) # Each col to drop must be in cols

    ## Prepare Dataset

    # Split into labels/not labels
    df_X = df_train.drop('Class', axis=1)
    df_y = df_train['Class']

    # Drop attributes to drop
    df_X = df_X.drop(drop_attributes, axis=1)

    X = df_X.to_numpy()
    y = df_y.to_numpy()

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
        pipe_parts.append('onehot', OneHotEncoder())
    else: # Use mixed encoder if not everything binned
        raise NotImplementedError() # TODO
    
    # Model itself
    if model =='knn':
        pipe_parts.append('model_knn', KNeighborsClassifier())
    if model =='dt':
        pipe_parts.append('model_dt', DecisionTreeClassifier())
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
            'knn__k_neighbors': [ii for ii in range(3, 22, 2)]
        }
    if model =='dt':
        param_grid = {
            'dt__criterion': ['gini', 'entropy'],
            'dt__min_samples_split': [ii for ii in range(2, 41)],
            'dt__min_samples_leaf': [ii for ii in range(1, 21)]
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
    
    grid_search = GridSearchCV(pipe, param_grid, cv=skfold, verbose=1)

    grid_search.fit(X, y)

    save_path = f"./trials/m={model}-s={sampling}-b={bins},d=[{','.join(drop_attributes)}]"

    print(f"Trial: Saving results to {save_path}")