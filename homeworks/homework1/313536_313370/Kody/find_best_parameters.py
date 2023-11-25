import argparse
import openml as oml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
import warnings


warnings.filterwarnings('ignore')

HYPERPARAMETERS_SPACE_DT = {
    'max_features': [None, 'log2', 'sqrt'],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': np.arange(1, 16), 
    'min_samples_split': np.arange(2, 20),  
    'min_samples_leaf': np.arange(1, 20),
}

HYPERPARAMETERS_SPACE_RF = {
    'n_estimators': np.arange(1, 2001),
    'bootstrap': [True, False],
    'max_features': np.arange(0.01, 1.01, 0.01),
    'max_depth': np.arange(1, 16),
    'min_samples_split': np.arange(2, 20),
    'min_samples_leaf': np.arange(1, 20),
    'criterion': ['gini', 'entropy'],
}

HYPERPARAMETERS_SPACE_ENET = {
    "alpha": np.arange(0, 1, 0.001),
    "l1_ratio": [2**x for x in np.arange(-10, 0.1, 0.01)],
}

target_labels = {
    37: 'class',
    1464: 'Class',
    1489: 'Class',
    40983: 'class'
}

CV = 3
RANDOM_STATE = 42
METRIC = 'roc_auc'


def get_openML_data(dataset_id: str) -> pd.DataFrame:
    """
    Downloads dataset from OpenML and returns it as a pandas dataframe.
    param dataset_id: id of the dataset to download
    """
    df = oml.datasets.get_dataset(dataset_id, download_data = True, download_qualities = True, download_features_meta_data=True)
    df_result, _, _, _ = df.get_data(dataset_format ="dataframe")
    return df_result


def prepare_split(df: pd.DataFrame, target: str, rs: int = 42) -> pd.DataFrame:
    """
    Splits the dataset into train and test sets.
    Parameters:
    - df: dataset to split
    - target: target variable
    - rs: random state (optional)
    """
    y = df[target]
    X = df.drop(target, axis=1)
    return train_test_split(X, y, test_size=0.3, random_state=rs)

def random_search_and_results(df: pd.DataFrame, 
                              target: str, 
                              model, 
                              param_dist: dict, 
                              n_iter: int = 100, 
                              verbose: int = 1, 
                              rs: int = 42, 
                              cv: int = 5, 
                              scoring: str = 'roc_auc') -> pd.DataFrame:
    """
    Applies Random Search to tune hyperparameters within a given parameter space.

    Parameters:
    - df: Dataset to perform the search on.
    - target: Target variable.
    - model: Model to tune.
    - param_dist: Parameter space to explore.
    - n_iter: Number of iterations (optional).
    - verbose: Verbosity level (optional).
    - rs: Random state (optional).
    - cv: Number of cross-validation folds (optional).
    - scoring: Scoring metric (optional).
    """
        
    X_train, X_test, y_train, y_test = prepare_split(df, target)
    
    random_model = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, 
                                    cv=cv, random_state=rs, scoring=scoring, verbose=verbose)

    random_model.fit(X_train, y_train)
    return pd.DataFrame(random_model.cv_results_)

def bayes_search_and_results(df: pd.DataFrame, 
                              target: str, 
                              model, 
                              param_dist: dict, 
                              n_iter: int = 100, 
                              verbose: int = 1, 
                              rs: int = 42, 
                              cv: int = 5, 
                              scoring: str = 'roc_auc') -> pd.DataFrame:
    """
    Applies Bayes Search sampling method to tune the hyperparameters from given parameter space
    on a provided dataset.
    Parameters:
    - df: Dataset to perform the search on.
    - target: Target variable.
    - model: Model to tune.
    - param_dist: Parameter space to explore.
    - n_iter: Number of iterations (optional).
    - verbose: Verbosity level (optional).
    - rs: Random state (optional).
    - cv: Number of cross-validation folds (optional).
    - scoring: Scoring metric (optional).
        """
        
    X_train, X_test, y_train, y_test = prepare_split(df, target)
    
    random_model = BayesSearchCV(model, search_spaces=param_dist, n_iter=n_iter, 
                                    cv=cv, random_state=rs, scoring=scoring, verbose=verbose)

    random_model.fit(X_train, y_train)
    return pd.DataFrame(random_model.cv_results_)

def main(args):
    model = args.model
    sampling = args.sampling
    iterations = args.iterations
    
    if model == 'DT': 
        clf = DecisionTreeClassifier()
        param_dist = HYPERPARAMETERS_SPACE_DT
    elif model == 'RF':
        clf = RandomForestClassifier()
        param_dist = HYPERPARAMETERS_SPACE_RF
    elif model == 'ENET':
        clf = ElasticNet()
        param_dist = HYPERPARAMETERS_SPACE_ENET

    else:
        raise ValueError('Model not supported')
    
    results_list = []

    for dataset_id, label in target_labels.items():
        df = get_openML_data(dataset_id)
        
        df[label] = np.where(df[label] == df[label].cat.categories[0], 0, 1)

        # apply the sampling method of choice
        if sampling == 'random':      
            results = random_search_and_results(df = df, target = label, model = clf, param_dist=param_dist,
                                  rs=RANDOM_STATE, n_iter=iterations, cv = CV, scoring=METRIC) 
        elif sampling == 'bayes':
            results = bayes_search_and_results(df = df, target = label, model = clf, param_dist=param_dist,
                                  rs=RANDOM_STATE, n_iter=iterations, cv = CV, scoring=METRIC)
        else:
            raise ValueError('Sampling method not supported')
        
        # add info about dataset id
        results['dataset_id'] = dataset_id

        results_list.append(results)
    
    # concatenate results of experiments on all datasets
    results_df = pd.concat(results_list)
    
    # calculate mean metric score
    results_df['total_mean_auc'] = results_df.groupby(results_df['params'].astype(str))['mean_test_score'].transform('mean')
    
    destination_path = "results/" + args.model + "_" + args.sampling + "_search_results.csv"
    results_df.to_csv(destination_path, index=False)
    print(sampling, 'search for model',  model, 'completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search best default hyperparameters')
    parser.add_argument('--model', type=str, default='DT', help='Model to optimize hyperparameters for')
    parser.add_argument('--sampling', type=str, default='random', help='Type of sampling method to perform')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for method')
    args = parser.parse_args()
    main(args)
