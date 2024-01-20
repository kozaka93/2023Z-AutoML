from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from mlflow_utils import DATA_PATH, MLFLOW_TRACKING_URI
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Homework2 - AutoML")


def main():
    with mlflow.start_run(
        run_name=f"CatBoost_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ):
        X_train = pd.read_csv(DATA_PATH / "X_train.csv")
        y_train = pd.read_csv(DATA_PATH / "y_train.csv")
        X_val = pd.read_csv(DATA_PATH / "X_val.csv")
        y_val = pd.read_csv(DATA_PATH / "y_val.csv")

        # change y values from -1, 1 to 0, 1
        y_train = y_train.replace(-1, 0)
        y_val = y_val.replace(-1, 0)

        # merge train and val data, because we decided to use cross validation
        X_train = pd.concat([X_train, X_val])
        y_train = pd.concat([y_train, y_val])

        mlflow.log_input(mlflow.data.from_pandas(X_train), context="train_x")
        mlflow.log_input(mlflow.data.from_pandas(y_train), context="train_y")

        param_dist = {
            "depth": np.arange(6, 11),
            "learning_rate": np.linspace(0.01, 0.1, 10),
            "iterations": np.arange(100, 301, 100),
            "l2_leaf_reg": np.arange(1, 10, 2),
            "subsample": np.linspace(0.8, 1.0, 3),
            "colsample_bylevel": np.linspace(0.8, 1.0, 3),
            "border_count": [32, 64, 128],
            "thread_count": [4],
            "loss_function": ["Logloss", "CrossEntropy"],
            "eval_metric": ["Logloss", "AUC"],
            "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
        }

        mlflow.log_params(param_dist)

        catboost_model = CatBoostClassifier()

        random_search = RandomizedSearchCV(
            catboost_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=10,
            scoring="balanced_accuracy",
            verbose=2,
            random_state=42,
        )
        random_search.fit(X_train, y_train)

        mlflow.log_metric("best_score", random_search.best_score_)
        best_params = random_search.best_params_

        model = CatBoostClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))

        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric("train_balanced_acc", train_acc)

        mlflow.catboost.log_model(model, "catboost_model")

        mlflow.set_tag("model_name", "CatBoost")
        mlflow.set_tag("search_type", "RANDOM")


if __name__ == "__main__":
    main()
