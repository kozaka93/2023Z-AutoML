import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tpot import TPOTClassifier
import h2o
from h2o.automl import H2OAutoML
from autogluon.tabular import TabularPredictor
import mlflow


def train_auto_gluon(X_train, X_val, y_train, y_val, training_time):
    """
    Trains an AutoGluon model on the provided data
        Args:
            X_train (pd.DataFrame): Training data.
            X_val (pd.DataFrame): Validation data.
            y_train (pd.DataFrame): Training labels.
            y_val (pd.DataFrame): Validation labels.
            training_time (int): Training time in seconds.
        Returns:
            predictor (AutoGluon): Trained Auto-sklearn classifier.
            train_acc (float): Balanced accuracy on training data.
            val_acc (float): Balanced accuracy on validation data.
    """
    mlflow.set_experiment("auto-gluon")
    with mlflow.start_run():
        predictor = TabularPredictor(label="y")
        predictor.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            tuning_data=pd.concat([X_val, y_val], axis=1),
            time_limit=training_time,
        )

        y_train_pred = predictor.predict(X_train)
        y_val_pred = predictor.predict(X_val)

        train_acc = balanced_accuracy_score(y_train, y_train_pred)
        val_acc = balanced_accuracy_score(y_val, y_val_pred)
        mlflow.log_params(
            predictor.info()["model_info"][predictor.info()["best_model"]][
                "hyperparameters"
            ]
        )
        mlflow.log_metric("train_balanced_accuracy", train_acc)
        mlflow.log_metric("val_balanced_accuracy", val_acc)

        print(
            f"AutoGluon balanced accuracy on train/val: {train_acc:.2f}/{val_acc:.2f}"
        )
    return predictor, val_acc


def train_tpot(X_train, X_val, y_train, y_val, training_time):
    """
    Trains an TPOT model on the provided data
        Args:
            X_train (pd.DataFrame): Training data.
            X_val (pd.DataFrame): Validation data.
            y_train (pd.DataFrame): Training labels.
            y_val (pd.DataFrame): Validation labels.
            training_time (int): Training time in minutes.
        Returns:
            tpot_classifier (TPOTClassifier): Trained TPOT classifier.
            train_acc (float): Balanced accuracy on training data.
            val_acc (float): Balanced accuracy on validation data.
    """
    mlflow.set_experiment("tpot")
    with mlflow.start_run():
        tpot_classifier = TPOTClassifier(
            generations=5,
            population_size=20,
            verbosity=2,
            random_state=42,
            max_time_mins=training_time,
        )
        tpot_classifier.fit(X_train, y_train)
        y_train_pred = tpot_classifier.predict(X_train)
        y_val_pred = tpot_classifier.predict(X_val)
        train_acc = balanced_accuracy_score(y_train, y_train_pred)
        val_acc = balanced_accuracy_score(y_val, y_val_pred)

        mlflow.log_params(tpot_classifier.get_params())
        mlflow.log_metric("train_balanced_accuracy", train_acc)
        mlflow.log_metric("val_balanced_accuracy", val_acc)

        print(f"TPOT balanced accuracy on train/val: {train_acc:.2f}/{val_acc:.2f}")
    return tpot_classifier, val_acc


def train_h2o(X_train, X_val, y_train, y_val, training_time):
    """
    Trains an H2O model on the provided data
        Args:
            X_train (pd.DataFrame): Training data.
            X_val (pd.DataFrame): Validation data.
            y_train (pd.DataFrame): Training labels.
            y_val (pd.DataFrame): Validation labels.
            training_time (int): Training time in seconds.
        Returns:
            automl.leader (H2OAutoML.leader): Trained H2O classifier.
            train_acc (float): Balanced accuracy on training data.
            val_acc (float): Balanced accuracy on validation data.
    """
    mlflow.set_experiment("h2o")
    with mlflow.start_run():
        h2o.init()

        train_df = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        val_df = h2o.H2OFrame(pd.concat([X_val, y_val], axis=1))

        train_df["y"] = train_df["y"].asfactor()
        val_df["y"] = val_df["y"].asfactor()

        x = train_df.columns[:-1]
        y = train_df.columns[-1]

        automl = H2OAutoML(max_runtime_secs=training_time)
        automl.train(x=x, y=y, training_frame=train_df)

        y_train_pred = (
            automl.leader.predict(train_df[:, :-1])
            .as_data_frame()["predict"]
            .astype(str)
        )
        y_val_pred = (
            automl.leader.predict(val_df[:, :-1]).as_data_frame()["predict"].astype(str)
        )

        train_acc = balanced_accuracy_score(y_train, y_train_pred)
        val_acc = balanced_accuracy_score(y_val, y_val_pred)

        mlflow.log_metric("train_balanced_accuracy", train_acc)
        mlflow.log_metric("val_balanced_accuracy", val_acc)

        print(f"H2O balanced accuracy on train/val: {train_acc:.2f}/{val_acc:.2f}")
    return automl.leader, val_acc


def train_models_and_return_best(
    X_train, X_val, y_train, y_val, training_time, output_dir="models/"
):
    """
    Trains all models on the provided data. Returns the best model.
    """
    # Training models
    auto_gluon_clf, auto_gluon_val_acc = train_auto_gluon(
        X_train, X_val, y_train, y_val, 60 * training_time
    )
    tpot_clf, tpot_val_acc = train_tpot(X_train, X_val, y_train, y_val, training_time)
    h2o_clf, h2o_val_acc = train_h2o(X_train, X_val, y_train, y_val, 60 * training_time)

    # Finding the best model based on validation accuracy
    best_model_val_acc = max(auto_gluon_val_acc, tpot_val_acc, h2o_val_acc)

    if auto_gluon_val_acc == best_model_val_acc:
        best_model = auto_gluon_clf
        model_name = "auto_gluon"

    elif tpot_val_acc == best_model_val_acc:
        best_model = tpot_clf
        model_name = "tpot"
    elif h2o_val_acc == best_model_val_acc:
        best_model = h2o_clf
        model_name = "h2o"
    else:
        raise ValueError("No best model found")

    return best_model, model_name
