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

NO_ITER = 500


def randomly_choose_hyperparameters(penalty):
    hyperparameters = HYPERPARAMETERS_SPACE_LR[penalty].copy()
    for key, value in hyperparameters.items():
        hyperparameters[key] = np.random.choice(value)
    return hyperparameters


labels = {44: "class", 1504: "Class", 37: "class", 1494: "Class"}


def main():
    hparams_results = {}
    for penalty in tqdm(HYPERPARAMETERS_SPACE_LR.keys(), desc="Penalty", position=0):
        for _ in tqdm(range(NO_ITER), desc="Iteration", position=1, leave=False):
            while True:
                try:
                    chosen_hyperparameters = randomly_choose_hyperparameters(penalty)

                    results = []

                    for dataset_number, label in labels.items():
                        dataset = openml.datasets.get_dataset(dataset_number)
                        df = dataset.get_data()[0]
                        df[label] = np.where(
                            df[label] == df[label].cat.categories[0], 0, 1
                        )
                        train, test = train_test_split(
                            df, test_size=0.2, random_state=42, stratify=df[label]
                        )

                        X_train = train.drop(label, axis=1)
                        y_train = train[label]
                        X_test = test.drop(label, axis=1)
                        y_test = test[label]

                        clf = LogisticRegression(**chosen_hyperparameters)
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
                    print("Error occured")
                    continue
                else:
                    break

        best_avg_auc = max(hparams_results.keys())
        hparams_results[best_avg_auc][0]

        best_hparams_auc = hparams_results[best_avg_auc][0]

        print(penalty)
        print(best_avg_auc)
        print(best_hparams_auc)
        print()

        path_to_save = "best_default_models"

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        file_name = f"best_hparams_auc_{penalty}_LR.pickle"

        with open(os.path.join(path_to_save, file_name), "wb") as handle:
            pickle.dump(best_hparams_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
