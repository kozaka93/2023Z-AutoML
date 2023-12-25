import openml
import pickle
import pandas as pd
import numpy as np
import json


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
# import Bayesian optimization from scikit-optimize
from skopt import BayesSearchCV


def count_correlation_with_target_select_randomly_m_out_of_top_n(df, target_name, top_n, m):
    df = df.copy(deep=True)
    df = df.dropna()
    
    df_numeric = df.select_dtypes(exclude=['object'])
    df_text = df.select_dtypes(include=['object'])

    # pearson for numeric
    corr_matrix_num = pd.DataFrame(df_numeric.corr().abs())
    correlation_num_with_target = corr_matrix_num[target_name]
    correlation_num_with_target = correlation_num_with_target.drop(index=[target_name])



    # spearman for text
    corr_matrix_text = pd.DataFrame(df_text.corr(method='spearman').abs())
    if not corr_matrix_text.empty:
        print(corr_matrix_text)
        correlation_text_with_target = corr_matrix_text[target_name]
        correlation_text_with_target = correlation_text_with_target.drop(index=[target_name])
        # concat
        corr_with_target = pd.concat([correlation_num_with_target, correlation_text_with_target])
    else:
        corr_with_target = correlation_num_with_target


    # get top n
    corr_with_target = corr_with_target.sort_values(ascending=False)
    corr_with_target = corr_with_target[:top_n]
    
    # select m out of top n randomly
    selected_features = corr_with_target.sample(m, ).index.tolist()


    return selected_features + [target_name]


def get_datasets():
    # List all datasets and their properties
    openml.datasets.list_datasets(output_format="dataframe")

    # Get dataset by ID
    df1 = openml.datasets.get_dataset(44098)
    df2 = openml.datasets.get_dataset(1461)
    df3 = openml.datasets.get_dataset(1120)
    df4 = openml.datasets.get_dataset(40945)

    return df1, df2, df3, df4

def uniform_search(model_type):

    if model_type == 'rfc':
        hyperparameters = {
            'n_estimators': np.random.randint(10, 1000),
            'max_depth': np.random.randint(2, 10),
            'min_samples_split': np.random.randint(2, 10),
            'min_samples_leaf': np.random.randint(1, 10),
            'max_features': np.random.uniform(0.1, 1.0)
        }
    elif model_type == 'xgb':  
        hyperparameters = {
            'n_estimators': np.random.randint(10, 1000),
            'max_depth': np.random.randint(2, 10),
            'learning_rate': np.random.uniform(0.01, 1.0),
            'subsample': np.random.uniform(0.5, 1.0),
            'min_child_weight': np.random.randint(1, 10),
            'max_leaves': np.random.randint(0, 10),
            'gamma': np.random.uniform(0.01, 1.0),
            'reg_alpha': np.random.uniform(0.01, 10.0),
            'reg_lambda': np.random.uniform(0.01, 10.0),
            'nthread': 4
            
        }
    elif model_type == 'lgbm':
        hyperparameters = {
            'n_estimators': np.random.randint(10, 1000),
            'max_depth': np.random.randint(2, 10),
            'learning_rate': np.random.uniform(0.01, 1.0),
            'subsample': np.random.uniform(0.5, 1.0),
            'min_child_weight': np.random.randint(1, 10),
            'num_leaves': np.random.randint(2, 50),
            'reg_alpha': np.random.uniform(0.01, 10.0),
            'reg_lambda': np.random.uniform(0.01, 10.0),

        }
    else:
        raise ValueError("Invalid model type. Supported types are 'rfc', 'xgb' and 'lgbm'.")
    
    return hyperparameters



def train_model(X_train, y_train, model_type, hyperparameters):

    if model_type == 'rfc':
        model = RandomForestClassifier(**hyperparameters)
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(**hyperparameters)
    elif model_type == 'lgbm':
        model = lgb.LGBMClassifier(**hyperparameters, verbose=-1)
    else:
        raise ValueError("Invalid model type. Supported types are 'rfc', 'xgb' and 'lgbm'.")

    # Train the model
    model.fit(X_train, y_train)

    return model


def estimate_best_defaults(df, model_type, metrics=['AUC'], aggregation_metric = 'AUC'):
    metric_values = {}
    # print(df)
    # return

    # find best hyperparamters for aggregation_metric
    d = df[df['Model'] == model_type]
    d = d[d['Metric'] == aggregation_metric]
    idx = d.groupby(['ID'], as_index=False)['Value'].mean().sort_values(by=['Value'], ascending=False, ignore_index=True).iloc[0]['ID']
    hyperparameters = d[d['ID'] == idx]['Hyperparameters'].reset_index(drop=True)[0]
    # print(idx, hyperparameters)
    # print(d.groupby(['ID'], as_index=False)['Value'].mean().sort_values(by=['Value'], ascending=False, ignore_index=True))

    f = df[df['Model'] == model_type]
    f = f[f['ID'] == idx]

    # find metric values for best hyperparameters
    for metric in metrics:
        # print(metric)
        g = f[f['Metric'] == metric]
        # print(g['Value'], g['Value'].mean(), sep='\n')
        # print(g[['Dataset','Value']], "\n-------------------------------------------\n")
        
        temp_dict = {}
        for i in range(len(g)):
            temp_dict[g.iloc[i]['Dataset']] = g.iloc[i]['Value']
        # print(temp_dict)
        
        metric_values[metric] = temp_dict


    return hyperparameters, metric_values

def create_and_save_best_defaults_dict(df, model_types, metrics=['AUC'], aggregation_metric = 'AUC', save=True):
    print(f'Selecting best defaults for {model_types} by {aggregation_metric}')
    best_defaults_dict = {}
    
    for model_type in model_types:
        best_defaults_dict[model_type] = estimate_best_defaults(df, model_type, metrics=metrics, aggregation_metric = aggregation_metric)
        

    with open("data/best_defaults_random_search.json", "w") as outfile: 
        json.dump(best_defaults_dict, outfile)
    return best_defaults_dict


def get_hyper_space_and_defaults(model_type, best_defaults_model):
    best_defaults_ = best_defaults_model[model_type][0]
    # print(best_defaults_)
    
  
    if model_type == 'rfc':
        hyperparameters_space = {
            'n_estimators': (10, 1000),
            'max_depth': (2, 10),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': (0.1, 1.0),
        }
    elif model_type == 'xgb':
        hyperparameters_space = {
            'nthread': (4,),
            'n_estimators': (10, 1000),
            'max_depth': (2, 10),
            'learning_rate': (0.01, 1.0),
            'subsample': (0.5, 1.0),
            'min_child_weight': (1, 10),
            'max_leaves': (0, 10),
            'gamma': (0.01, 1.0),
            'reg_alpha': (0.01, 10.0),
            'reg_lambda': (0.01, 10.0),
        }
    elif model_type == 'lgbm':
        hyperparameters_space = {
            'n_estimators': (10, 1000),
            'max_depth': (2, 10),
            'learning_rate': (0.01, 1.0),
            'subsample': (0.5, 1.0),
            'min_child_weight': (1, 10),
            'num_leaves': (2, 50),
            'reg_alpha': (0.01, 10.0),
            'reg_lambda': (0.01, 10.0),
            'nthread': (4,),
        }

    else:
        raise ValueError("Invalid model type. Supported types are 'rfc', 'xgb' and 'lgbm'.")
    
    return hyperparameters_space, best_defaults_


def train_hyperparam_bayes(X_train, y_train, model_type, default_hypers, hyper_space, scoring='roc_auc'):

    if model_type == 'rfc':
        # print(default_hypers)
        model_defaults = RandomForestClassifier(**default_hypers)
        
        model_bare = RandomForestClassifier()
        # do bayesian optimization
        model_bayes = BayesSearchCV(model_bare, hyper_space, n_iter=50, cv=5, scoring=scoring, verbose=0, n_points = 50, n_jobs=18)
        
    elif model_type == 'xgb':
        model_defaults = xgb.XGBClassifier(**default_hypers)
        model_bare = xgb.XGBClassifier()
        # do bayesian optimization
        model_bayes = BayesSearchCV(model_bare, hyper_space, n_iter=50, cv=5, scoring=scoring, verbose=0, n_points = 50 , n_jobs=18)

    elif model_type == 'lgbm':
        model_defaults = lgb.LGBMClassifier(**default_hypers, verbose=-1)
        model_bare = lgb.LGBMClassifier(verbose=-1)
        # do bayesian optimization
        model_bayes = BayesSearchCV(model_bare, hyper_space, n_iter=50, cv=5, scoring=scoring, verbose=0, n_points = 50, n_jobs=18)
    
    
    else:
        raise ValueError("Invalid model type. Supported types are 'rfc', 'xgb' and 'lgbm'.")

    # Train the model
    model_bayes.fit(X_train, y_train)

    return model_bayes, model_bayes.best_params_ 

