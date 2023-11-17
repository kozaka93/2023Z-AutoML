import pickle
import argparse
import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings

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

HYPERPARAMETERS_SPACE_LR = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'l1_ratio': np.random.uniform(0, 1, 100),
    'C': np.random.uniform(0, 250, 500),
    'fit_intercept': [True],
}

NO_ITER = 250

def randomly_choose_hyperparameters(model):
    if model == 'RFC':
        hyperparameters_space = HYPERPARAMETERS_SPACE_RFC
    elif model == 'LR':
        hyperparameters_space = HYPERPARAMETERS_SPACE_LR
    elif model == 'XGB':
        hyperparameters_space = HYPERPARAMETERS_SPACE_XGB

    hyperparameters = {}
    for key, value in hyperparameters_space.items():
        hyperparameters[key] = np.random.choice(value)
    return hyperparameters

labels = {
    44: 'class',
    1504: 'Class',
    37: 'class',
    1494: 'Class'
}


def main(args):
    model = args.model

    hparams_results = {}
    for epoch in tqdm(range(NO_ITER)):
        while True:
            try:
                chosen_hyperparameters = randomly_choose_hyperparameters(model)

                results = []

                for dataset_number, label in labels.items():
                    dataset = openml.datasets.get_dataset(dataset_number)
                    df = dataset.get_data()[0]
                    df[label] = np.where(df[label] == df[label].cat.categories[0], 0, 1)
                    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label])

                    X_train = train.drop(label, axis=1)
                    y_train = train[label]
                    X_test = test.drop(label, axis=1)
                    y_test = test[label]

                    if model == 'RFC':
                        clf = RandomForestClassifier(**chosen_hyperparameters)
                    elif model == 'LR':
                        clf = LinearRegression(**chosen_hyperparameters)
                    elif model == 'XGB':
                        clf = XGBClassifier(**chosen_hyperparameters, enable_categorical=True)
                    else:
                        raise ValueError('Model not supported')

                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    auc = roc_auc_score(y_test, y_pred)
                    results.append(auc)

                avg_auc = np.mean(results)

                if avg_auc in hparams_results:
                    hparams_results[avg_auc].append(chosen_hyperparameters)
                else:
                    hparams_results[avg_auc] = [chosen_hyperparameters]
            except:
                print('Error occured')
                continue
            else:
                break

    best_avg_auc = max(hparams_results.keys())
    hparams_results[best_avg_auc][0]

    best_hparams_auc = hparams_results[best_avg_auc][0]

    print(best_avg_auc)
    print(best_hparams_auc)

    with open(f'best_hparams_auc_{model}.pickle', 'wb') as handle:
        pickle.dump(best_hparams_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search best default hyperparameters')
    parser.add_argument('--model', type=str, default='RFC', help='Model to search best default hyperparameters for')
    args = parser.parse_args()
    main(args)