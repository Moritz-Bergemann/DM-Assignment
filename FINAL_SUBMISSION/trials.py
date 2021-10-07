## Moritz Bergemann 2021
## Main code for trialing different models/hyperparameters.
###############################
## ATTEMPT AT FULL FUNCTIONS ##
###############################

import pickle 
import datetime as dt
import itertools
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.naive_bayes import GaussianNB, ComplementNB
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

# Non Sklearn-adjustable hyperparams go in here!!! # TODO drop-trials rather than drop_attributes
def trial(model:str, param_grid:dict, drop_attributes:list, sampling:str, bins:int, verbosity=2, save_folder='trials', scaling_technique='none', scoring='accuracy', just_get_pipeline=False): # TODO pick h-params to adjust
    df_train, _ = data_prep.get_prepped_dataset(bins=bins, normalize=False)

    df_train.to_csv("check_myself.csv", index=False)

    ## Input validation
    assert sampling in ['none', 'smote', 'random-under', 'smote-random-under']
    assert scoring in ['accuracy', 'f1', 'f1_macro']
    assert scaling_technique in ['none', 'minmax', 'standard', 'quantile']
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

    # print(X.columns)
    # exit(0)

    print("[i] Loaded dataset")

    # Encode labels
    y = LabelEncoder().fit_transform(y)

    ## Make data pipeline
    pipe_parts = []

    # First, define which are numeric and which are categorical attributes (if data is mixed, if data was retrieved with binning everything will be categorical)
    if bins == None:
        categorical_atts = X.select_dtypes(include='category').columns
        numeric_atts = X.select_dtypes(include=['int64', 'float64']).columns

        # Get cat feature mask needed by SMOTENC
        categorical_features = [col in categorical_atts for col in X.columns]

    # Feature selection (if this is to be a pipeline component!)

    # Sampling strategy
    if sampling != 'none':
        if bins != None: # If everything binned
            if sampling == 'smote':
                pipe_parts.append(('smoten', SMOTEN(sampling_strategy=1)))
            elif sampling == 'random-under':
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.5)))
            elif sampling == 'smote-random-under':
                pipe_parts.append(('smoten', SMOTEN(sampling_strategy=0.5)))
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.75)))
        else: # Add Smote-Mixed if things not binned
            if sampling == 'smote':
                pipe_parts.append(('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=1)))
            elif sampling == 'random-under':
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.5)))
            elif sampling == 'smote-random-under':
                pipe_parts.append(('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=0.5)))
                pipe_parts.append(('undersample', RandomUnderSampler(sampling_strategy=0.75)))

    # One-hot encoding
    if bins != None: # Use onehot encoder if everything binned
        categories = [list(X[col].cat.categories) for col in X.columns]
        pipe_parts.append(('onehot', OneHotEncoder(categories=categories)))
    else: # Use mixed encoder if not everything binned
        assert len(categorical_atts) + len(numeric_atts) == len(X.columns)

        categories = [list(X[col].cat.categories) for col in X[categorical_atts].columns]

        # scaler = 'passthrough'
        # Choose scaling type 
        if scaling_technique == 'none':
            scaler = 'passthrough'
        elif scaling_technique == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_technique == 'standard':
            scaler = StandardScaler()
        elif scaling_technique == 'quantile':
            scaler = QuantileTransformer(n_quantiles=len(X)//2)

        col_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(categories=categories, sparse='True'), categorical_atts)
            ], # One-hot encode categorical attributes
            sparse_threshold=1, # Always return sparse (why is this even a feature???)
            remainder=scaler # Scale numeric attributes
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
        # pipe_parts.append(('to_dense', DenseTransformer()))
        pipe_parts.append(('model_comnbayes', ComplementNB()))
    elif model =='svc':
        pipe_parts.append(('model_svc', SVC()))

    # Make final pipeline
    pipe = ImbPipeline(pipe_parts)

    # Just return the pipeline if this is what we want (this should be a seperate function but time is of the essence)
    if (just_get_pipeline):
        return pipe, X, y

    # Define K-fold sampling
    skfold = StratifiedKFold(n_splits=10)
    
    ## Perform grid search hyperparameter optimization    
    grid_search = GridSearchCV(pipe, param_grid, cv=skfold, verbose=verbosity, n_jobs=-1, scoring=scoring)

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
    save_path = f"./{save_folder}/m={model}-s={sampling}-b={bins},d={drop_save_string},sc={scaling_technique}.pickle"

    print(f"[i] Trial: Saving results to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(grid_search, f)
    
    print(f"[s] Best Score: {grid_search.best_score_}")

    sanity_check_acc, sanity_check_f1 = _check_results(grid_search, drop_attributes=drop_attributes, bins=bins)
    print(f"[s] Sanity Check: {sanity_check_acc}")
    print(f"[s] F1 Score: {sanity_check_f1}")
    
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
    # experiment1()
    # experiment2()
    # experiment2b()
    experiment3()

    #'C14', 'C23', 'C24', 'C27'

    #'C23', 'C24', 'C19', 'C14', 'C27', 'C29', 'C28'



    # param_grid = {
    #     'model_comnbayes__alpha': [0.01, 0.1, 1]
    # }
    # _, best_score, _ = trial(model='comnbayes', param_grid=param_grid, scaling_technique='quantile', drop_attributes=None, sampling='none', bins=None, save_folder='./beans', scoring='accuracy', verbosity=0)
    

    # param_grid = {
    #     'model_svc__C': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
    # }

    # param_grid = {
    #     'model_svc__C': [1],
    #     'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
    # }
    
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=['C23', 'C19', 'C14', 'C27', 'C28'], scaling_technique='quantile', sampling='smote-random-under', bins=None, save_folder="./sanity", scoring='f1_macro', verbosity=0)
    # result, _, _ = trial(model='comnbayes', param_grid=param_grid, drop_attributes=['C24', 'C19', 'C14', 'C28'], scaling_technique='none', sampling='smote-random-under', bins=5, save_folder="./sanity", scoring='f1_macro', verbosity=0)

    # print(result.best_score_)

    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=5, save_folder="./final", verbosity=0)
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=5, save_folder="./final", verbosity=0)
    
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote-random-under', bins=None, save_folder="./final", verbosity=0)
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='smote', bins=None, save_folder="./final", verbosity=0)
    # result, _, _ = trial(model='svc', param_grid=param_grid, drop_attributes=None, sampling='random-under', bins=None, save_folder="./final", verbosity=0)

def experiment3():
    """Final hyperparam checking"""
    search_space = [
        {
            'model': 'knn',
            'scaling': 'none',
            'bins': 5,
            'drop': ['C14', 'C27', 'C29'],
            'param_grid': {
                'model_knn__n_neighbors': list(range(3, 99, 2))
            }
        },
        {
            'model': 'dt',
            'scaling': 'none',
            'bins': 5,
            'drop': ['C19', 'C14', 'C28'],
            'param_grid': {
                'model_dt__max_depth': [None] + list(range(1, 50, 5)),
                'model_dt__criterion': ['entropy', 'gini'],
                'model_dt__min_samples_split': range(2, 60, 4),
                'model_dt__min_samples_leaf': range(1, 21, 2)
            }
        },
        {
            'model': 'gnbayes',
            'scaling': 'standard',
            'bins': None,
            'drop': ['C23', 'C24', 'C19', 'C27', 'C29', 'C28'],
            'param_grid': {
                'model_gnbayes__var_smoothing': np.logspace(0,-9, num=100)
            }
        },
        {
            'model': 'comnbayes',
            'scaling': 'none',
            'bins': 5,
            'drop': ['C24', 'C19', 'C14', 'C28'],
            'param_grid': {
                'model_comnbayes__alpha': np.logspace(-4,4, num=400)
            }
        },
        {
            'model': 'svc',
            'scaling': 'quantile',
            'bins': None,
            'drop': ['C23', 'C19', 'C14', 'C27', 'C28'],
            'param_grid': {
                'model_svc__C': [2 ** ii for ii in range (-5, 11, 1)],
                'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'model_svc__gamma': [2 ** ii for ii in range (-15, 2, 1)]
            }
        }
    ]

    save_folder = 'experiment3'
    sampling = 'smote-random-under'

    skips = 4
    init_skips = skips

    result_data = []
    for scoring in ['accuracy', 'f1_macro', 'f1']:
        for item in search_space:
            print(f"{item['model']} {scoring}")
            if skips > 0:
                print("Skipping")
                skips -= 1
                continue
            result, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique=item['scaling'], drop_attributes=item['drop'], sampling=sampling, bins=item['bins'], save_folder=save_folder, scoring=scoring, verbosity=2)
            result_data.append([scoring, item['model'], best_score, str(result.best_params_)])
        
            results_table = pd.DataFrame(data=result_data, columns=["metric", "model", "score", "best_params"])
            results_table.to_csv(f'./experiment_logs/experiment3-{init_skips}skips.csv', index=False)




def experiment2b():
    """Verifying feature selection technique"""
    search_space = [
        {
            'model': 'knn',
            'scaling': 'none',
            'bins': 5,
            'param_grid': {
                'model_knn__n_neighbors': [3, 5, 7]
            }
        },
        {
            'model': 'dt',
            'scaling': 'none',
            'bins': 5,
            'param_grid': {
                'model_dt__criterion': ['entropy'],
                'model_dt__min_samples_split': [2, 10, 40],
                'model_dt__min_samples_leaf': [1, 10, 20]
            }
        },
        {
            'model': 'gnbayes',
            'scaling': 'standard',
            'bins': None,
            'param_grid': {
                'model_gnbayes__var_smoothing': np.logspace(0,-9, num=10)
            }
        },
        {
            'model': 'comnbayes',
            'scaling': 'none',
            'bins': 5,
            'param_grid': {
                'model_comnbayes__alpha': [0.01, 0.1, 1]
            }
        },
        {
            'model': 'svc',
            'scaling': 'quantile',
            'bins': None,
            'param_grid': {
                'model_svc__C': [0.1, 1, 10],
                'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
            }
        }
    ]

    save_folder = './experiment2'

    print("[M] STARTING EXPERIMENT 2B (Feature Selection Validation)")

    df, _ = data_prep.get_prepped_dataset()

    cols = ['C23', 'C24', 'C19', 'C14', 'C27', 'C29', 'C28']

    # col_combinations = [[col] for col in cols] + [list(combo) for combo in list(itertools.combinations(cols, 2))] + [list(combo) for combo in list(itertools.combinations(cols, 3))]
    col_combinations = [None]
    for ii in range(len(cols)):
        col_combinations = col_combinations + [list(combo) for combo in list(itertools.combinations(cols, ii))]

    print(f"[M] Testing {len(col_combinations)} combinations")

    # Just do by F1
    result_data = []
    for col_combination in col_combinations:
        for item in search_space:
            print(f"[M] m={item['model']} d={col_combination}")
            _, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique=item['scaling'], drop_attributes=col_combination, sampling='none', bins=item['bins'], save_folder=save_folder, scoring='f1_macro', verbosity=0)
            result_data.append([col_combination, item['model'], best_score])
    
        results_table = pd.DataFrame(data=result_data, columns=["scaling_technique", "model", "best_score_f1"])
        results_table.to_csv('./experiment_logs/experiment2b_f1.csv')

def experiment2():
    """Feature selection technique checking"""

    search_space = [
        {
            'model': 'knn',
            'scaling': 'none',
            'bins': 5,
            'param_grid': {
                'model_knn__n_neighbors': [3, 5, 7]
            }
        },
        {
            'model': 'dt',
            'scaling': 'none',
            'bins': 5,
            'param_grid': {
                'model_dt__criterion': ['entropy'],
                'model_dt__min_samples_split': [2, 10, 40],
                'model_dt__min_samples_leaf': [1, 10, 20]
            }
        },
        {
            'model': 'gnbayes',
            'scaling': 'standard',
            'bins': None,
            'param_grid': {
                'model_gnbayes__var_smoothing': np.logspace(0,-9, num=10)
            }
        },
        {
            'model': 'comnbayes',
            'scaling': 'none',
            'bins': 5,
            'param_grid': {
                'model_comnbayes__alpha': [0.01, 0.1, 1]
            }
        },
        {
            'model': 'svc',
            'scaling': 'quantile',
            'bins': None,
            'param_grid': {
                'model_svc__C': [0.1, 1, 10],
                'model_svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]
            }
        }
    ]

    save_folder = './experiment2'

    print("[M] STARTING EXPERIMENT 2 (Feature Selection)")

    df, _ = data_prep.get_prepped_dataset()

    cols = df.drop('Class', axis=1).columns

    # col_combinations = [[col] for col in cols] + [list(combo) for combo in list(itertools.combinations(cols, 2))] + [list(combo) for combo in list(itertools.combinations(cols, 3))]
    col_combinations = [None] + [[col] for col in cols] + [list(combo) for combo in list(itertools.combinations(cols, 2))]

    print(f"[M] Testing {len(col_combinations)} combinations")

    # Just do by F1
    result_data = []
    for col_combination in col_combinations:
        for item in search_space:
            print(f"[M] m={item['model']} d={col_combination}")
            _, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique=item['scaling'], drop_attributes=col_combination, sampling='none', bins=item['bins'], save_folder=save_folder, scoring='f1_macro', verbosity=0)
            result_data.append([col_combination, item['model'], best_score])
    
        results_table = pd.DataFrame(data=result_data, columns=["scaling_technique", "model", "best_score_f1"])
        results_table.to_csv('./experiment_logs/experiment2_f1.csv')

def experiment1():
    """Scaling technique checking"""
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

    save_folder = './experiment1'
    scaling_techniques = ['none', 'minmax', 'standard', 'quantile']
    bin_sizes = [5, 10, 20]

    print("[M] STARTING EXPERIMENT 1")
    print("[M] Accuracy Metric")
    result_data = []
    for scaling_technique in scaling_techniques:
        for item in search_space:
            print(f"[M] m={item['model']}, scaling={scaling_technique}")

            # Don't do SVC without scaling 
            if (item['model'] == 'svc' and scaling_technique == 'none') or (item['model'] == 'comnbayes' and scaling_technique == 'standard'):
                print("[M] Skipping svc + none or comnbayes + scaling (non-functional)")
                continue
            
            _, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique=scaling_technique, drop_attributes=None, sampling='none', bins=None, save_folder=save_folder, scoring='accuracy', verbosity=0)
            result_data.append([scaling_technique, item['model'], best_score])

    # Do binning-related 'scaling'
    for bin_size in bin_sizes:
        for item in search_space:
            print(f"[M] m={item['model']}, scaling={bin_size}bins")

            _, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique='none', drop_attributes=None, sampling='none', bins=bin_size, save_folder=save_folder, scoring='accuracy', verbosity=0)
            result_data.append([f"{bin_size} bins", item['model'], best_score])


    results_table = pd.DataFrame(data=result_data, columns=["scaling_technique", "model", "best_score_acc"])
    results_table.to_csv('./experiment_logs/experiment1_accuracy.csv', index=False)

    print("[M] F1 Metric")
    result_data = []
    for scaling_technique in scaling_techniques:
        for item in search_space:
            print(f"[M] m={item['model']}, scaling={scaling_technique}")

            # Don't do SVC without scaling 
            if (item['model'] == 'svc' and scaling_technique == 'none') or (item['model'] == 'comnbayes' and scaling_technique == 'standard'):
                print("[M] Skipping svc + none or comnbayes + scaling (non-functional)")
                continue

            _, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique=scaling_technique, drop_attributes=None, sampling='none', bins=None, save_folder=save_folder, scoring='f1_macro', verbosity=0)
            result_data.append([scaling_technique, item['model'], best_score])

    # Do binning-related 'scaling'
    for bin_size in bin_sizes:
        for item in search_space:
            print(f"[M] m={item['model']}, scaling={bin_size}bins")

            _, best_score, _ = trial(model=item['model'], param_grid=item['param_grid'], scaling_technique='none', drop_attributes=None, sampling='none', bins=bin_size, save_folder=save_folder, scoring='f1_macro', verbosity=0)
            result_data.append([f"{bin_size} bins", item['model'], best_score])


    results_table = pd.DataFrame(data=result_data, columns=["scaling_technique", "model", "best_score_f1"])
    results_table.to_csv('./experiment_logs/experiment1_f1.csv', index=False)

if __name__ == '__main__':
    main()