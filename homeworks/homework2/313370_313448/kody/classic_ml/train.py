import lightgbm
import pandas as pd
import xgboost


def get_default_xgb_hparams():
    return {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "learning_rate": 0.3,
        "min_split_loss": 0,
        "max_depth": 6,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "subsample": 1,
        "colsample_bytree": 1,
        "tree_method": "hist",
        "n_estimators": 100,
        "seed": 42,
    }


def get_default_lgbm_hparams():
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "lambda_l1": 0,
        "lambda_l2": 0,
        "num_leaves": 31,
        "feature_fraction": 1,
        "bagging_fraction": 1,
        "bagging_freq": 0,
        "min_child_samples": 20,
        "seed": 42,
    }


DEFAULT_HPARAMS = {
    "xgb": get_default_xgb_hparams(),
    "lgbm": get_default_lgbm_hparams(),
}


def train_model(
    model_class: str, model_params: dict, X_train: pd.DataFrame, y_train: pd.DataFrame
):
    if model_class == "lgbm":
        train_dataset = lightgbm.Dataset(X_train, label=y_train)
        clf = lightgbm.train(model_params, train_dataset)
    elif model_class == "xgb":
        train_dataset = xgboost.DMatrix(X_train, label=y_train)
        clf = xgboost.train(model_params, train_dataset)
    else:
        raise ValueError(f"Unknown model class {model_class}.")

    return clf
