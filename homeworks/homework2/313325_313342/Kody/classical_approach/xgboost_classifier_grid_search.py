from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from mlflow_utils import DATA_PATH, MLFLOW_TRACKING_URI
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Homework2 - AutoML")


def main():
    with mlflow.start_run(
        run_name=f"XGBoost_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

        param_grid = {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
        }

        mlflow.log_params(param_grid)

        xgb_model = xgb.XGBClassifier()

        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=10,
            scoring="balanced_accuracy",
            verbose=2,
        )
        grid_search.fit(X_train, y_train)

        mlflow.log_metric("best_score", grid_search.best_score_)

        best_params = grid_search.best_params_

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))

        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric("train_balanced_acc", train_acc)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", signature=signature
        )

        mlflow.set_tag("model_name", "XGBoost")
        mlflow.set_tag("search_type", "GRID")


if __name__ == "__main__":
    main()
