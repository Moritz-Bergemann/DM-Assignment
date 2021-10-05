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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.naive_bayes import GaussianNB, ComplementNB, CategoricalNB
from imblearn.over_sampling import SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import TransformerMixin

import data_prep

RANDOM_STATE = 123
# TODO don't be a perfectionist!

np.random.seed(0)

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

# Non Sklearn-adjustable hyperparams go in here!!!
def trial(model:str, param_grid:dict, drop_attributes:list, sampling:str, bins:int, verbosity=2, save_folder='trials'): # TODO pick h-params to adjust
    df_train, _ = data_prep.get_prepped_dataset(bins=bins, normalize=True)

    ## Input validation
    assert sampling in ['none', 'smote', 'random-under', 'smote-random-under', 'smote-enn', 'smote-tomek']
    assert model in ['knn', 'dt', 'gnbayes', 'comnbayes', 'svc']
    if drop_attributes != None:
        assert set(drop_attributes) <= set(list(df_train.columns)) # Each col to drop must be in cols

    if not os.path.isdir(f"./{save_folder}"):
        os.mkdir(f"./{save_folder}")

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

    # First, define which are numeric and which are categorical attributes (if data is mixed)
    if bins == None:
        categorical_atts = X.select_dtypes(include='category').columns
        numeric_atts = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Get cat feature mask needed by SMOTENC
        categorical_features = [col in categorical_atts for col in X.columns]

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
                pipe_parts.append(('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=1)))
            elif sampling == 'random-under':
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.5)))
            elif sampling == 'smote-random-under':
                pipe_parts.append(('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=0.45)))
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.6)))

    # One-hot encoding
    if bins != None: # Use onehot encoder if everything binned
        categories = [list(X[col].cat.categories) for col in X.columns]
        pipe_parts.append(('onehot', OneHotEncoder(categories=categories)))
    else: # Use mixed encoder if not everything binned
        assert len(categorical_atts) + len(numeric_atts) == len(X.columns)

        categories = [list(X[col].cat.categories) for col in X[categorical_atts].columns]

        # # Choose scaling type # TODO remove this 
        # if scaling == 'none':
        #     scaler = 'passthrough'
        # elif scaling == 'minmax':
        #     scaler = MinMaxScaler()
        # elif scaling == 'standard':
        #     scaler = StandardScaler()
        # elif scaling == 'quantile':
        #     scaler = QuantileTransformer(n_quantiles=len(X)//2)

        col_transformer = ColumnTransformer(
            transformers=[
                # ('onehot', OrdinalEncoder(categories=categories), categorical_atts)
                ('onehot', OneHotEncoder(categories=categories), categorical_atts)
            ], # One-hot encode categorical attributes
            remainder='passthrough' # Scale numeric attributes
        )
        
        pipe_parts.append(('selective-onehot', col_transformer))

    # Model itself
    if model =='knn':
        pipe_parts.append(('model_knn', KNeighborsClassifier()))
    elif model =='dt':
        pipe_parts.append(('model_dt', DecisionTreeClassifier()))
    elif model =='gnbayes':
        pipe_parts.append(('to_dense', DenseTransformer()))
        pipe_parts.append(('model_gnbayes', GaussianNB())) 
    elif model =='comnbayes':
        pipe_parts.append(('model_comnbayes', ComplementNB()))
    # elif model =='catnbayes':
    #     assert bins != None # Cannot have categorical nbayes with no binning
    #     pipe_parts.append(('to_dense', DenseTransformer())) # FIXME CHECK THIS
    #     pipe_parts.append(('model_catnbayes', CategoricalNB()))
    elif model =='svc':
        pipe_parts.append(('model_svc', SVC()))

    # Make final pipeline
    pipe = ImbPipeline(pipe_parts)

    # Define K-fold sampling
    skfold = StratifiedKFold(n_splits=10)
    
    ## Suggested param_grid params:
    # knn - k_neighbors
    # dt - criterion, min_samples_split, min_samples_leaf
    # nbayes - TODO
    # svc - TODO

    grid_search = GridSearchCV(pipe, param_grid, cv=skfold, verbose=verbosity, n_jobs=-1)

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
    
    print(f"[s] Best Score: {grid_search.best_score_}")

    sanity_check_acc, sanity_check_f1 = _check_results(grid_search, drop_attributes=drop_attributes, bins=bins)
    print(f"[s] Sanity Check: {sanity_check_acc}")
    print(f"[s] F1 Score: {sanity_check_f1}")
    
    # sanity_check_dict = {
    #     'acc': sanity_check_acc,
    #     'f1': sanity_check_f1
    # }

    return grid_search, grid_search.best_score_, sanity_check_f1 #, sanity_check_dict

def _check_results(result:GridSearchCV, drop_attributes, bins):
    ## Get the dataset
    df_train, _ = data_prep.get_prepped_dataset(bins=bins, normalize=True)

    ## Prepare Dataset
    # Split into labels/not labels
    X = df_train.drop('Class', axis=1)
    y = df_train['Class']

    # Drop attributes to drop
    if drop_attributes != None and len(drop_attributes) != 0:
        X = X.drop(drop_attributes, axis=1)

    # Encode labels
    y = LabelEncoder().fit_transform(y)

    # Sanity check
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

    estimator = result.best_estimator_
    estimator.fit(X_train, y_train)

    perf = accuracy_score(estimator.predict(X_val), y_val)

    # F1 score using k-fold crossval
    f1 = np.average(cross_val_score(estimator, X, y, cv=10, scoring='f1')) #F1 binary scoring (since we have binary class labels)

    return perf, f1

def main():

    param_grid = {
        'model_svc__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
    }

    result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=5, save_folder="./final", verbosity=0)

    

    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=5, save_folder="./final", verbosity=0)
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=5, save_folder="./final", verbosity=0)
    
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=None, save_folder="./final", verbosity=0)
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=None, save_folder="./final", verbosity=0)
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=None, save_folder="./final", verbosity=0)



def experiment2():
    """Dropping strategy checking"""

    print("[M] STARTING EXPERIMENT 2")
    save_folder = "experiment2"

    search_space = [
        {
            'model': 'knn',
            'param_grid': {
                'model_knn__n_neighbors': [3, 5, 7]
            }
        },
        {
            'model': 'dt',
            'param_grid': {
                'model_dt__criterion': ['entropy'],
                'model_dt__min_samples_split': [2, 10, 40],
                'model_dt__min_samples_leaf': [1, 10, 20]
            }
        },
        {
            'model': 'gnbayes',
            'param_grid': {
                'model_gnbayes__var_smoothing': np.logspace(0,-9, num=10)
            }
        },
        {
            'model': 'comnbayes',
            'param_grid': {
                'model_comnbayes__alpha': [0.01, 0.1, 1]
            }
        },
        {
            'model': 'svc',
            'param_grid': {
                'model_svc__C': [0.1, 1, 10],
                'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
            }
        }
    ]

    df_train, _ = data_prep.get_prepped_dataset(bins=5)
    cols = df_train.drop('Class', axis=1).columns
    result_data = []

    for col in cols:
        for item in search_space:
            print(f"[M] m={item['model']}, drop={col}")
            _, best_score, f1 = trial(model=item['model'], param_grid=item['param_grid'], drop_attributes=[col], sampling='none', bins=5, save_folder=save_folder, verbosity=0)
            result_data.append([col, item['model'], best_score, f1])
    
    results_table = pd.DataFrame(data=result_data, columns=["dropped_column", "model", "best_score", "f1"])
    with open('./experiment_logs/experiment2.pickle', 'wb') as f:
        pickle.dump(results_table, f)

def experiment1():
    """Binning strategy checking"""
    print("[M] STARTING EXPERIMENT 1")
    save_folder = "experiment1"

    no_bins_search_space = [
        {
            'model': 'knn',
            'param_grid': {
                'model_knn__n_neighbors': [3, 5, 7]
            }
        },
        {
            'model': 'dt',
            'param_grid': {
                'model_dt__criterion': ['entropy'],
                'model_dt__min_samples_split': [2, 10, 40],
                'model_dt__min_samples_leaf': [1, 10, 20]
            }
        },
        {
            'model': 'gnbayes',
            'param_grid': {
                'model_gnbayes__var_smoothing': np.logspace(0,-9, num=10)
            }
        },
        {
            'model': 'comnbayes',
            'param_grid': {
                'model_comnbayes__alpha': [0.01, 0.1, 1]
            }
        },
        {
            'model': 'svc',
            'param_grid': {
                'model_svc__C': [0.1, 1, 10],
                'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
            }
        }
    ]

    bins_search_space = [
        {
            'model': 'knn',
            'param_grid': {
                'model_knn__n_neighbors': [3, 5, 7]
            }
        },
        {
            'model': 'dt',
            'param_grid': {
                'model_dt__criterion': ['entropy'],
                'model_dt__min_samples_split': [2, 10, 40],
                'model_dt__min_samples_leaf': [1, 10, 20]
            }
        },
        {
            'model': 'gnbayes',
            'param_grid': {
                'model_gnbayes__var_smoothing': np.logspace(0,-9, num=10)
            }
        },
        {
            'model': 'comnbayes',
            'param_grid': {
                'model_comnbayes__alpha': [0.01, 0.1, 1]
            }
        },
        {
            'model': 'svc',
            'param_grid': {
                'model_svc__C': [0.1, 1, 10],
                'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
            }
        }
    ]

    bin_values = [5, 10, 20]
    result_data = []

    # First do none
    for item in no_bins_search_space:
        print(f"[M] m={item['model']}, bins=None")
        _, best_score, f1 = trial(model=item['model'], param_grid=item['param_grid'], drop_attributes=None, sampling='none', bins=None, save_folder=save_folder, verbosity=0)
        result_data.append(["No bins", item['model'], best_score, f1])

    for bin_value in bin_values:
        for item in bins_search_space:
            print(f"[M] m={item['model']}, bins={bin_value}")
            _, best_score, f1 = trial(model=item['model'], param_grid=item['param_grid'], drop_attributes=None, sampling='none', bins=bin_value, save_folder=save_folder, verbosity=0)
            result_data.append([bin_value, item['model'], best_score, f1])

    print("[M] DUMPING FINAL RESULTS")
    results_table = pd.DataFrame(data=result_data, columns=["bins", "model", "best_score", "f1"])
    with open('./experiment_logs/experiment1.pickle', 'wb') as f:
        pickle.dump(results_table, f)

# def experimentX():
    # """Sampling strategy checking"""
    # print("[M] STARTING EXPERIMENT 1")

    # save_folder = "experiment1"

    # search_space = [
    #     {
    #         'model': 'knn',
    #         'param_grid': {
    #             'model_knn__n_neighbors': [3, 5, 7]
    #         }
    #     },
    #     {
    #         'model': 'dt',
    #         'param_grid': {
    #             'model_dt__criterion': ['entropy'],
    #             'model_dt__min_samples_split': [2, 10, 40],
    #             'model_dt__min_samples_leaf': [1, 10, 20]
    #         }
    #     },
    #     {
    #         'model': 'gnbayes',
    #         'param_grid': {
    #             'model_gnbayes__var_smoothing': np.logspace(0,-9, num=10)
    #         }
    #     },
    #     {
    #         'model': 'comnbayes',
    #         'param_grid': {
    #             'model_comnbayes__alpha': [0.01, 0.1, 1]
    #         }
    #     },
    #     {
    #         'model': 'svc',
    #         'param_grid': {
    #             'model_svc__C': [0.1, 1, 10],
    #             'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
    #         }
    #     }
    # ]

    # scaling_types = ['none']

    # for scaling_type in scaling_types:
    #     for item in search_space:
    #         print(f"[M] m={item['model']}, sc={scaling_type}")
    #         trial(model=item['model'], param_grid=item['param_grid'], drop_attributes=None, sampling='none', bins=None, scaling=scaling_type, save_folder=save_folder, verbosity=0)

    # print("[M] EXPERIMENT 1 COMPLETE")

# def main2():
#     # OLD STUFF
#     param_grid = {
#         'model_dt__criterion': ['entropy'],
#         'model_dt__min_samples_split': [2, 10, 40],
#         'model_dt__min_samples_leaf': [1, 10, 20]
#     }

#     print("No Sampling Strategy")
#     result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='none', bins=10, verbosity=0)
#     print(result.best_score_)

#     print("Undersampling (Random)")
#     result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=10, verbosity=0)
#     print(result.best_score_)

#     print("Oversampling")
#     result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=10, verbosity=0)
#     print(result.best_score_)

#     print("SMOTE & Random Undersampling")
#     result = trial(model='dt', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=10, verbosity=0)
#     print(result.best_score_)

#     param_grid = {
#         'model_knn__n_neighbors': [3, 5, 7, 9]
#     }

#     print("No Sampling Strategy")
#     result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='none', bins=10, verbosity=0)
#     print(f"Best Score: {result.best_score_}")
#     print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

#     print("Undersampling (Random)")
#     result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=10, verbosity=0)
#     print(f"Best Score: {result.best_score_}")
#     print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

#     print("Oversampling")
#     result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=10, verbosity=0)
#     print(f"Best Score: {result.best_score_}")
#     print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

#     print("SMOTE & Random Undersampling")
#     result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=10, verbosity=0)
#     print(f"Best Score: {result.best_score_}")
#     print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

#     print("SMOTE & ENN Undersampling")
#     result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-enn', bins=10, verbosity=0)
#     print(f"Best Score: {result.best_score_}")
#     print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")

#     print("SMOTE & Tomek Undersampling")
#     result = trial(model='knn', param_grid=param_grid, drop_attributes=None, sampling='smote-tomek', bins=10, verbosity=0)
#     print(f"Best Score: {result.best_score_}")
#     print(f"Sanity check: {sanity_check(result, drop_attributes=None, bins=10)}")


if __name__ == '__main__':
    main()