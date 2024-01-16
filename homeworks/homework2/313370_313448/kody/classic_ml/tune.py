# Based on: https://github.com/optuna/optuna-examples
import logging
import os

import lightgbm
import mlflow
import optuna
import pandas as pd
import xgboost
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import balanced_accuracy_score


def lgbm_objective(trial, X_train, y_train, X_val, y_val):
    train_dataset = lightgbm.Dataset(X_train, label=y_train)

    hparams = {
        "seed": 42,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    lgbm_clf = lightgbm.train(hparams, train_dataset)
    y_val_pred = lgbm_clf.predict(X_val).round()
    val_acc = balanced_accuracy_score(y_val, y_val_pred)
    return val_acc


def xgb_objective(trial, X_train, y_train, X_val, y_val):
    train_dataset = xgboost.DMatrix(X_train, label=y_train)
    valid_dataset = xgboost.DMatrix(X_val, label=y_val)

    hparams = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "tree_method": "exact",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if hparams["booster"] in ["gbtree", "dart"]:
        hparams["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        hparams["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        hparams["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-8, 1.0, log=True
        )
        hparams["min_split_loss"] = trial.suggest_float(
            "min_split_loss", 1e-8, 1.0, log=True
        )
        hparams["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if hparams["booster"] == "dart":
        hparams["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        hparams["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        hparams["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        hparams["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    xgb_clf = xgboost.train(hparams, train_dataset)
    y_val_pred = xgb_clf.predict(valid_dataset).round()
    val_acc = balanced_accuracy_score(y_val, y_val_pred)
    return val_acc


def log_best_trial(study):
    logging.info("Best trial:")
    trial = study.best_trial
    logging.info("  Value: {}".format(trial.value))
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info("    {}: {}".format(key, value))


def get_best_hparams(
    model_name: str,
    n_trials: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
):
    logging.info(f"Training {model_name} model.")
    if model_name == "lgbm":
        objective = lgbm_objective
    elif model_name == "xgb":
        objective = xgb_objective
    else:
        raise ValueError("Model not supported.")

    mlflow.set_experiment(experiment_name=model_name)
    mlflow_callback = MLflowCallback(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        metric_name="val_balanced_accuracy",
        create_experiment=False,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        callbacks=[mlflow_callback],
    )

    logging.info("Number of finished trials: {}".format(len(study.trials)))
    log_best_trial(study)
    if model_name == "lgbm":
        study.best_params["objective"] = "binary"
        study.best_params["metric"] = "binary_logloss"
    elif model_name == "xgb":
        study.best_params["objective"] = "binary:logistic"

    return study.best_params
