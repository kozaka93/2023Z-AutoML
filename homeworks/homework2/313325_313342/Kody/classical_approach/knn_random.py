from datetime import datetime

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow_utils import DATA_PATH, MLFLOW_TRACKING_URI
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Homework2 - AutoML")


def main():
    with mlflow.start_run(
        run_name=f"KNN_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ):
        X_train = pd.read_csv(DATA_PATH / "X_train.csv")
        y_train = pd.read_csv(DATA_PATH / "y_train.csv")
        X_val = pd.read_csv(DATA_PATH / "X_val.csv")
        y_val = pd.read_csv(DATA_PATH / "y_val.csv")

        mlflow.log_input(mlflow.data.from_pandas(X_train), context="train_x")
        mlflow.log_input(mlflow.data.from_pandas(y_train), context="train_y")
        mlflow.log_input(mlflow.data.from_pandas(X_val), context="val_x")
        mlflow.log_input(mlflow.data.from_pandas(y_val), context="val_y")

        param_dist = {
            "n_neighbors": [3, 5, 7, 10, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        }

        mlflow.log_params(param_dist)

        knn_model = KNeighborsClassifier()

        random_search = RandomizedSearchCV(
            knn_model,
            param_distributions=param_dist,
            n_iter=10,
            cv=5,
            scoring="balanced_accuracy",
            verbose=1,
            random_state=42,
        )
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_

        model = KNeighborsClassifier(**best_params)
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
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", signature=signature
        )
        mlflow.set_tag("model_name", "KNN")
        mlflow.set_tag("search_type", "RANDOM")


if __name__ == "__main__":
    main()
