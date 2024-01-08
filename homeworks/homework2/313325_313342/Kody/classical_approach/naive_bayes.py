from datetime import datetime

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.utils import mlflow_tags
from mlflow_utils import DATA_PATH, MLFLOW_TRACKING_URI
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Homework2 - AutoML")


def main():
    with mlflow.start_run(
        run_name=f"NaiveBayes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ):
        X_train = pd.read_csv(DATA_PATH / "X_train.csv")
        y_train = pd.read_csv(DATA_PATH / "y_train.csv")
        X_val = pd.read_csv(DATA_PATH / "X_val.csv")
        y_val = pd.read_csv(DATA_PATH / "y_val.csv")

        mlflow.log_input(mlflow.data.from_pandas(X_train), context="train_x")
        mlflow.log_input(mlflow.data.from_pandas(y_train), context="train_y")
        mlflow.log_input(mlflow.data.from_pandas(X_val), context="val_x")
        mlflow.log_input(mlflow.data.from_pandas(y_val), context="val_y")

        model = GaussianNB()
        model.fit(X_train, y_train)

        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
        val_acc = balanced_accuracy_score(y_val, model.predict(X_val))

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
        mlflow.set_tag("model_name", "NaiveBayes")
        mlflow.set_tag("search_type", "None")


if __name__ == "__main__":
    main()
