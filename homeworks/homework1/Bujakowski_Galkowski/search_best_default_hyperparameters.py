import argparse
import os
import pickle
import warnings

import numpy as np
import openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HYPERPARAMETERS_SPACE_RFC = {
    "n_estimators": list(range(1, 2001)),
    "max_depth": [None] + list(range(10, 51)),
    "min_samples_split": list(range(2, 11)),
    "min_samples_leaf": list(range(1, 5)),
    "max_features": ["auto", "sqrt", "log2"] + list(np.arange(0.1, 1.1, 0.1)),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
    "max_samples": [None] + list(np.arange(0.1, 1.1, 0.1)),
}

HYPERPARAMETERS_SPACE_XGB = {
    "n_estimators": np.arange(1, 150),
    "max_depth": np.arange(1, 15),
    "learning_rate": np.random.uniform(0, 1, 50),
    "booster": ["gbtree", "gblinear", "dart"],
    "gamma": [2**i for i in range(-10, 10, 1)],
    "subsumple": np.random.uniform(0.1, 1, 10),
    "colsample_bytree": np.random.uniform(0, 1, 10),
    "colsample_bylevel": np.random.uniform(0, 1, 10),
    "reg_alpha": [2**i for i in range(-10, 10, 1)],
    "reg_lambda": [2**i for i in range(-10, 10, 1)],
}

NO_ITER = 500


def randomly_choose_hyperparameters(model):
    if model == "RFC":
        hyperparameters_space = HYPERPARAMETERS_SPACE_RFC
    elif model == "XGB":
        hyperparameters_space = HYPERPARAMETERS_SPACE_XGB

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
                clf = RandomForestClassifier(**chosen_hyperparameters)
            elif model == "XGB":
                clf = XGBClassifier(**chosen_hyperparameters, enable_categorical=True)
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

    path_to_save = "best_default_models/"

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
        default="XGB",
        help="Model to search best default hyperparameters for",
    )
    args = parser.parse_args()
    main(args)
