import os
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

HYPERPARAMETERS_SPACE_LR = {
    "l1": {
        "penalty": ["l1"],
        "solver": ["liblinear"],
        "C": np.random.uniform(0, 500, 500),
    },
    "l2": {
        "penalty": ["l2"],
        "C": np.random.uniform(0, 500, 500),
    },
    "elasticnet": {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "C": np.random.uniform(0, 500, 500),
        "l1_ratio": np.random.uniform(0, 1, 200),
    },
    "none": {
        "penalty": ["none"],
        "C": np.random.uniform(0, 250, 500),
    },
}

labels = {44: "class", 1504: "Class", 37: "class", 1494: "Class"}


def prepare_data(dataset_number):
    dataset = openml.datasets.get_dataset(dataset_number)
    df = dataset.get_data()[0]
    df[labels[dataset_number]] = np.where(
        df[labels[dataset_number]] == df[labels[dataset_number]].cat.categories[0], 0, 1
    )
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[labels[dataset_number]]
    )

    X_train = train.drop(labels[dataset_number], axis=1)
    y_train = train[labels[dataset_number]]
    X_test = test.drop(labels[dataset_number], axis=1)
    y_test = test[labels[dataset_number]]

    return X_train, y_train, X_test, y_test


def main():
    for penalty in tqdm(HYPERPARAMETERS_SPACE_LR.keys(), desc="Penalty", position=0):
        hyperparams_penalty = HYPERPARAMETERS_SPACE_LR[penalty].copy()
        with open(
            f"best_default_models/best_hparams_auc_{penalty}_LR.pickle", "rb"
        ) as handle:
            best_hparams_auc = pickle.load(handle)

        default_auc_values = {}
        for dataset_number in labels.keys():
            X_train, y_train, X_test, y_test = prepare_data(dataset_number)

            clf = LogisticRegression(**best_hparams_auc)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            default_auc_values[dataset_number] = roc_auc_score(y_test, y_pred)

        print("Default AUC values: ", default_auc_values)

        hyperparams_tunability = defaultdict(list)
        for hyperparam in tqdm(
            best_hparams_auc.keys(), position=1, desc="Hyperparameter"
        ):
            if hyperparam in ["penalty", "solver"]:
                continue
            print("Hyperparameter: ", hyperparam)

            no_iters = int(
                np.ceil(len(hyperparams_penalty[hyperparam]) * len(labels) * 0.8)
            )

            for _ in tqdm(range(no_iters), position=2, leave=False, desc="Iteration"):
                dataset_number = int(np.random.choice(list(labels.keys())))
                X_train, y_train, X_test, y_test = prepare_data(dataset_number)

                new_hyperparam_value = np.random.choice(hyperparams_penalty[hyperparam])
                best_hparams_copy = best_hparams_auc.copy()
                best_hparams_copy[hyperparam] = new_hyperparam_value

                clf = LogisticRegression(**best_hparams_copy)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                new_auc = roc_auc_score(y_test, y_pred)

                difference = new_auc - default_auc_values[dataset_number]
                hyperparams_tunability[hyperparam].append(difference)

            path_to_save = f"tunability/LR/random_search"

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            path_to_save = os.path.join(path_to_save, penalty)

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            for hyperparam in hyperparams_tunability.keys():
                file_name = f"{hyperparam}_tunability.pickle"
                with open(os.path.join(path_to_save, file_name), "wb") as handle:
                    pickle.dump(hyperparams_tunability[hyperparam], handle)


if __name__ == "__main__":
    main()
