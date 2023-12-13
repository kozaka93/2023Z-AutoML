import argparse
import os
import pickle
import warnings

import numpy as np
import openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

np.random.seed(42)

HYPERPARAMETERS_SPACE_RFC = {
    "n_estimators": list(range(1, 200)),
    "max_depth": [None] + list(range(10, 51)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 11)),
    "max_features": np.random.uniform(0.1, 1, 100),
    "bootstrap": [True],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
    "max_samples": np.random.uniform(0.1, 1, 100),
}

HYPERPARAMETERS_SPACE_XGB = {
    "n_estimators": np.arange(1, 150),
    "max_depth": np.arange(1, 15),
    "learning_rate": np.random.uniform(0, 1, 100),
    "booster": ["gbtree", "gblinear", "dart"],
    "gamma": np.random.uniform(0.001, 1024, 1000),
    "subsumple": np.random.uniform(0.1, 1, 10),
    "colsample_bytree": np.random.uniform(0, 1, 10),
    "colsample_bylevel": np.random.uniform(0, 1, 10),
    "reg_alpha": np.random.uniform(0.001, 1024, 1000),
    "reg_lambda": np.random.uniform(0.001, 1024, 1000),
}

HYPERPARAMETERS_SPACE_TREE = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": np.arange(1, 30),
    "min_samples_split": np.arange(2, 30),
    "min_samples_leaf": np.arange(1, 30),
    "max_features": np.random.uniform(0.1, 1, 100),
}

NO_ITER = 500


def randomly_choose_hyperparameters(model):
    if model == "RFC":
        hyperparameters_space = HYPERPARAMETERS_SPACE_RFC
    elif model == "XGB":
        hyperparameters_space = HYPERPARAMETERS_SPACE_XGB
    elif model == "TREE":
        hyperparameters_space = HYPERPARAMETERS_SPACE_TREE

    hyperparameters = {}
    for key, value in hyperparameters_space.items():
        hyperparameters[key] = np.random.choice(value)
    return hyperparameters


labels = {44: "class", 1504: "Class", 37: "class", 1494: "Class"}


def main(args):
    model = args.model

    hparams_results = {}
    for _ in tqdm(range(NO_ITER)):
        chosen_hyperparameters = randomly_choose_hyperparameters(model)

        results = []

        for dataset_number, label in labels.items():
            dataset = openml.datasets.get_dataset(dataset_number)
            df = dataset.get_data()[0]
            df[label] = np.where(df[label] == df[label].cat.categories[0], 0, 1)
            train, test = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df[label]
            )

            X_train = train.drop(label, axis=1)
            y_train = train[label]
            X_test = test.drop(label, axis=1)
            y_test = test[label]

            if model == "RFC":
                clf = RandomForestClassifier(**chosen_hyperparameters, random_state=42)
            elif model == "XGB":
                clf = XGBClassifier(
                    **chosen_hyperparameters, enable_categorical=True, random_state=42
                )
            elif model == "TREE":
                clf = DecisionTreeClassifier(**chosen_hyperparameters, random_state=42)
            else:
                raise ValueError("Model not supported")

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            results.append(auc)

        avg_auc = np.mean(results)

        if avg_auc in hparams_results:
            hparams_results[avg_auc].append(chosen_hyperparameters)
        else:
            hparams_results[avg_auc] = [chosen_hyperparameters]

    best_avg_auc = max(hparams_results.keys())
    hparams_results[best_avg_auc][0]

    best_hparams_auc = hparams_results[best_avg_auc][0]

    print(best_avg_auc)
    print(best_hparams_auc)

    path_to_save = "../Wyniki/best_default_models/"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    with open(
        os.path.join(path_to_save, f"best_hparams_auc_{model}.pickle"), "wb"
    ) as handle:
        pickle.dump(best_hparams_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search best default hyperparameters")
    parser.add_argument(
        "--model",
        type=str,
        default="TREE",
        help="Model to search best default hyperparameters for",
    )
    args = parser.parse_args()
    main(args)
