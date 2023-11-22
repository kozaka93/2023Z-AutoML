## load libraries
import openml
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint
from sklearn import set_config
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV, space

set_config(transform_output = "pandas")

hw1_path = os.path.join(os.path.expanduser("~"), f'Desktop\\AutoML\\HW1')

## in order to run this script, you need to pass --random or --bayes argument (or both)
run_random = '--random' in sys.argv
run_bayes = '--bayes' in sys.argv

iterations_count = 50

def run_train_iteration(dataset_id: int, target_column_name: str, model, random_search_params: dict, bayes_params: dict):
    ## load dataset
    dataset = openml.datasets.get_dataset(dataset_id)

    ## define features and target
    X, _, _, _ = dataset.get_data(dataset_format="dataframe")
    y = X.loc[:, target_column_name]
    X = X.drop([target_column_name], axis = 1)

    ## define pipeline
    col_trans = ColumnTransformer(transformers=[
            ('num_pipeline', MinMaxScaler(), make_column_selector(dtype_include=np.number)),
            ('cat_pipeline', OneHotEncoder(handle_unknown='error', sparse_output=False), make_column_selector(dtype_include=['category', np.object_]))
        ],
        remainder='drop',
        n_jobs=-1
    )

    model_pipeline = Pipeline([('preprocessing', col_trans),
                            ('model', model)])

    if run_random:
        print(f'Tuning using RandomizedSearchCV started for {type(model).__name__} on dataset {dataset.name}')
        ## tune hyperparams using grid search
        grid_search = RandomizedSearchCV(model_pipeline, random_search_params, n_iter=iterations_count, cv=10, scoring='roc_auc', n_jobs=-1, random_state=42)
        grid_search.fit(X, y)

        ## save results to csv
        path = os.path.join(hw1_path, f'results\\{type(model).__name__}\\random\\{dataset.name}.csv')

        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv(path)

    if run_bayes:
        print(f'Tuning using BayesSearchCV started for {type(model).__name__} on dataset {dataset.name}')
        ## tune hyperparams using bayesian optimization
        bayes_search = BayesSearchCV(model_pipeline, bayes_params, n_iter=iterations_count, cv=10, scoring='roc_auc', n_jobs=-1)
        bayes_search.fit(X, y)

        ## save results to csv
        path = os.path.join(hw1_path, f'results\\{type(model).__name__}\\bayes\\{dataset.name}.csv')

        results = pd.DataFrame(bayes_search.cv_results_)
        results.to_csv(path)


if __name__ == '__main__':
    datasets = [
        (1120, 'class:'),
        (1046, 'state'),
        (4534, 'Result'),
        (846, 'binaryClass')
    ]

    algorithms = [
        (
            DecisionTreeClassifier(),
            {
                'model__max_depth': randint(1, 31),
                'model__min_samples_leaf': randint(1, 61),
                'model__min_samples_split': randint(2, 61)
            },
            {
                'model__max_depth': space.Integer(1, 30),
                'model__min_samples_leaf': space.Integer(1, 60),
                'model__min_samples_split': space.Integer(2, 60)
            }
        ),
        (
            RandomForestClassifier(),
            {
                'model__n_estimators': randint(1, 501),
                'model__max_samples': uniform(loc=0.1, scale=0.9),
                'model__max_features': uniform(loc=0.1, scale=0.9)
            },
            {
                'model__n_estimators': space.Integer(1, 500),
                'model__max_samples': space.Real(0.1, 1, prior='uniform'),
                'model__max_features': space.Real(0.1, 1, prior='uniform')
            }
        ),
        (
            LogisticRegression(penalty='elasticnet', solver='saga'),
            {
                'model__C': loguniform(1e-3, 1e2),
                'model__l1_ratio': uniform(),
            },
            {
                'model__C': space.Real(1e-3, 1e2, prior='log-uniform'),
                'model__l1_ratio': space.Real(0, 1, prior='uniform'),
            }
        ),
    ]

    ## tune hyperparameters
    for dataset_id, target_column_name in datasets:
        for model, random_search_params, bayes_params in algorithms:
            run_train_iteration(dataset_id, target_column_name, model, random_search_params, bayes_params)

    print('Success')