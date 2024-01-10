from datetime import datetime

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from mlflow_utils import DATA_PATH, MLFLOW_TRACKING_URI
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV

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

        mlflow.log_input(mlflow.data.from_pandas(X_train), context="train_x")
        mlflow.log_input(mlflow.data.from_pandas(y_train), context="train_y")
        mlflow.log_input(mlflow.data.from_pandas(X_val), context="val_x")
        mlflow.log_input(mlflow.data.from_pandas(y_val), context="val_y")

        param_grid = {
            'depth': [6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.9, 1.0],
        }


        mlflow.log_params(param_grid)

        catboost_model = CatBoostClassifier()

        grid_search = GridSearchCV(
            catboost_model,
            param_grid,
            cv=5,
            scoring="balanced_accuracy",
            verbose=2,
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        model = CatBoostClassifier(**best_params)
        model.fit(X_train, y_train)

        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
        val_acc = balanced_accuracy_score(y_val, model.predict(X_val))

        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                "train_balanced_acc": train_acc,
                "val_balanced_acc": val_acc,
            },
        )

        mlflow.catboost.log_model(model, "catboost_model")

        mlflow.set_tag("model_name", "CatBoost")
        mlflow.set_tag("search_type", "GRID")


if __name__ == "__main__":
    main()
