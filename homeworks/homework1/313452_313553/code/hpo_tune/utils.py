import numpy as np
import pandas as pd
import json
import ast

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform, randint

from typing import Dict


def get_generic_preprocessing_pipeline() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(sparse=False, handle_unknown="ignore")),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (cat_pipeline, make_column_selector(dtype_include=object)),
                    (num_pipeline, make_column_selector(dtype_include=np.number)),
                ),
            )
        ]
    )

    return pipeline


def get_model_pipeline(model: any, preprocessing_pipeline: Pipeline) -> Pipeline:
    model_pipeline = Pipeline(
        [("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    return model_pipeline


def parse_hp_config_random(config: dict) -> dict:
    """
    change lists like [0, 1] to random generators
    config (dict): dictionary with params distribution
    """

    new_config = {}

    for key in config:
        item = config[key]
        if type(item) != list:
            raise ValueError(
                "wrong value for hp params. Hp params in .yaml should be in form [str, str, ...] or [number, number]"
            )
        if type(config[key][0]) != str:
            low = config[key][0]
            high = config[key][1]
            if isinstance(low, int) and isinstance(high, int):
                new_config[("model__" + key)] = randint(low=low, high=high + 1)
            else:
                new_config[("model__" + key)] = uniform(loc=low, scale=(high - low))
        else:
            new_config[("model__" + key)] = config[key]

    return new_config


def parse_hp_config_bayes(config: dict) -> dict:
    """
    change lists like [0, 1] to random generators
    config (dict): dictionary with params distribution
    """

    new_config = {}

    for key in config:
        item = config[key]
        if type(item) != list:
            raise ValueError(
                "wrong value for hp params. Hp params in .yaml should be in form [str, str, ...] or [number, number]"
            )
        if type(config[key][0]) != str:
            low = config[key][0]
            high = config[key][1]
            new_config[("model__" + key)] = (low, high)
        else:
            new_config[("model__" + key)] = config[key]

    return new_config


def random_search_results_to_dict(search: RandomizedSearchCV) -> Dict[str, any]:
    parsed_best_params = {}

    for key in search.best_params_:
        parsed_best_params[key[7:]] = search.best_params_[key]

    param_history = {}
    sufix_len = len("param_model__")
    for key in search.cv_results_:
        if "param_model__" in key:
            param_history[key[sufix_len:]] = list(search.cv_results_[key].data)

    res_dir = {
        "best_params": parsed_best_params,
        "best_score": search.best_score_,
        "mean_score_history": search.cv_results_["mean_test_score"],
        "param_history": param_history,
    }

    return res_dir


def load_best_params(
    model_name: str, results_path: str = "data/output/results.csv"
) -> dict:
    df = pd.read_csv(results_path)

    df_model = df[df["model"] == model_name]

    arr = np.ndarray(
        shape=(
            len(df_model["mean_score_history"]),
            len(_string_to_list(df_model["mean_score_history"].iloc[0])),
        )
    )
    for idx, row in enumerate(df_model["mean_score_history"]):
        arr[idx] = _string_to_list(row)

    argmax_ = np.argmax(arr.mean(axis=0))

    params = json.loads(str.replace(df_model["param_history"].iloc[0], "'", '"'))

    for key in params:
        params[key] = params[key][argmax_]

    return params


def _string_to_list(list_: str) -> list:
    list_ = " ".join(list_.split())
    return ast.literal_eval(str.replace(str.replace(list_, " ", ","), '\n', ''))
