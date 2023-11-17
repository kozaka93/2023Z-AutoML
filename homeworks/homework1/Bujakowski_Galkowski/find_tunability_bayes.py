from collections import defaultdict
import pickle
import argparse
import openml
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV

warnings.filterwarnings('ignore')

HYPERPARAMETERS_SPACE_RFC = {
    'n_estimators': list(range(1, 2001)),
    'max_depth': [None] + list(range(10, 51)),
    'min_samples_split': list(range(2, 11)),
    'min_samples_leaf': list(range(1, 5)),
    'max_features': ['auto', 'sqrt', 'log2'] + list(np.arange(0.1, 1.1, 0.1)),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced'],
    'max_samples': [None] + list(np.arange(0.1, 1.1, 0.1)),
}

HYPERPARAMETERS_SPACE_LR = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'l1_ratio': np.random.uniform(0, 1, 100),
    'C': np.random.uniform(0, 250, 500),
    'fit_intercept': [True],
}

HYPERPARAMETERS_SPACE_XGB = {
    "n_estimators": np.arange(1, 1000),
    "max_depth": np.arange(1, 1000),
    "max_leaves": np.arange(0, 10000),
    'min_child_weight':np.arange(1, 50, 1),
    "grow_policy": ['depthwise', 'lossguide'],
    "learning_rate": np.random.uniform(0, 1, 100),
    "booster": ['gbtree', 'gblinear', 'dart'],
    "gamma": np.random.uniform(0, 1, 100),
    "subsumple": np.random.uniform(0, 1, 100),
    "colsample_bytree": np.random.uniform(0, 1, 100),
    "reg_alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
    "reg_lambda": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
}

NO_ITER = 250

labels = {
    44: 'class',
    1504: 'Class',
    37: 'class',
    1494: 'Class'
}


def prepare_data(dataset_number):
    dataset = openml.datasets.get_dataset(dataset_number)
    df = dataset.get_data()[0]
    df[labels[dataset_number]] = np.where(df[labels[dataset_number]] == df[labels[dataset_number]].cat.categories[0], 0, 1)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[labels[dataset_number]])

    X_train = train.drop(labels[dataset_number], axis=1)
    y_train = train[labels[dataset_number]]
    X_test = test.drop(labels[dataset_number], axis=1)
    y_test = test[labels[dataset_number]]

    return X_train, y_train, X_test, y_test

def main(args):
    model = args.model
    print('Model: ', model)
    with open(f'best_hparams_auc_{model}.pickle', 'rb') as handle:
        best_hparams_auc = pickle.load(handle)

    default_auc_values = {}
    for dataset_number in labels.keys():
        X_train, y_train, X_test, y_test = prepare_data(dataset_number)

        if model == 'RFC':
            clf = RandomForestClassifier(**best_hparams_auc)
        elif model == 'LR':
            clf = LogisticRegression(**best_hparams_auc)
        elif model == 'XGB':
            clf = XGBClassifier(**best_hparams_auc)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        default_auc_values[dataset_number] = roc_auc_score(y_test, y_pred)

    print('Default AUC values: ', default_auc_values)

    hyperparams_tunability = defaultdict(list)
    for hyperparam in best_hparams_auc.keys():
        print('Hyperparameter: ', hyperparam)
        for _ in range(NO_ITER):
            dataset_number = int(np.random.choice(list(labels.keys())))
            X_train, y_train, X_test, y_test = prepare_data(dataset_number)

            while True:
                try:
                    if model == 'RFC':
                        # new_hyperparam_value = np.random.choice(HYPERPARAMETERS_SPACE_RFC[hyperparam])
                        best_hparams_copy = best_hparams_auc.copy()
                        # best_hparams_copy[hyperparam] = new_hyperparam_value
                        del best_hparams_copy[hyperparam]
                        opt = BayesSearchCV(
                            RandomForestClassifier(**best_hparams_copy),
                            # HYPERPARAMETERS_SPACE_RFC, # tylko jeden
                            HYPERPARAMETERS_SPACE_RFC[hyperparam],
                            n_iter=50,
                            n_jobs=-1,
                            cv=3,
                            scoring='roc_auc',
                            random_state=42,
                        )
                    elif model == 'LR':
                        # new_hyperparam_value = np.random.choice(HYPERPARAMETERS_SPACE_RFC[hyperparam])
                        best_hparams_copy = best_hparams_auc.copy()
                        # best_hparams_copy[hyperparam] = new_hyperparam_value
                        opt = BayesSearchCV(
                            LogisticRegression(**best_hparams_copy),
                            HYPERPARAMETERS_SPACE_LR[hyperparam],
                            n_iter=50,
                            n_jobs=-1,
                            cv=3,
                            scoring='roc_auc',
                            random_state=42,
                        )
                    elif model == 'XGB':
                        # new_hyperparam_value = np.random.choice(HYPERPARAMETERS_SPACE_RFC[hyperparam])
                        best_hparams_copy = best_hparams_auc.copy()
                        # best_hparams_copy[hyperparam] = new_hyperparam_value
                        opt = BayesSearchCV(
                            XGBClassifier(**best_hparams_copy),
                            HYPERPARAMETERS_SPACE_XGB[hyperparam],
                            n_iter=50,
                            n_jobs=-1,
                            cv=3,
                            scoring='roc_auc',
                            random_state=42,
                        )

                    opt.fit(X_train, y_train)
                except:
                    continue
                else:
                    break

            new_hyperparam_value = opt.best_params_[hyperparam]
            best_hparams_copy = best_hparams_auc.copy()
            best_hparams_copy[hyperparam] = new_hyperparam_value

            if model == 'RFC':
                clf = RandomForestClassifier(**best_hparams_copy)
            elif model == 'LR':
                clf = LogisticRegression(**best_hparams_copy)
            elif model == 'XGB':
                clf = XGBClassifier(**best_hparams_copy)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            new_auc = roc_auc_score(y_test, y_pred)

            difference = new_auc - default_auc_values[dataset_number]
            hyperparams_tunability[hyperparam].append(difference)

    for hyperparam in hyperparams_tunability.keys():
        with open(f'tunability/{model}/{model}_{hyperparam}_tunabilitybayes.pickle', 'wb') as handle:
            pickle.dump(hyperparams_tunability[hyperparam], handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RFC')
    args = parser.parse_args()
    main(args)
