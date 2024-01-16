import pandas as pd
import numpy as np

from loguru import logger
from sklearn.model_selection import RandomizedSearchCV

from .utils import (
    get_generic_preprocessing_pipeline,
    get_model_pipeline,
    random_search_results_to_dict,
)


def search_for_best_hp(
    Model: type,
    hparams: dict,
    X: pd.DataFrame,
    y: pd.DataFrame,
    OptimizerCls: type,
    n_iter: int = 50,
    seed: int = 123,
) -> pd.DataFrame:
    model = Model()

    preprocessing_pipeline = get_generic_preprocessing_pipeline()
    model_pipeline = get_model_pipeline(model, preprocessing_pipeline)

    np.random.seed(123)
    search = OptimizerCls(
        model_pipeline, hparams, scoring="accuracy", verbose=0, n_iter=n_iter
    )
    search.fit(X, y)

    params = random_search_results_to_dict(search)

    series = pd.Series(params)
    df = pd.DataFrame(series).T

    return df


def search_single_best_hp(
    Model: type,
    hparams: dict,
    X: pd.DataFrame,
    y: pd.DataFrame,
    best_hp: dict,
    OptimizerCls: type,
    n_iter: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame()

    for key in hparams:
        logger.info(f"Param={key}")
        model = Model(**best_hp)
        preprocessing_pipeline = get_generic_preprocessing_pipeline()
        model_pipeline = get_model_pipeline(model, preprocessing_pipeline)

        search = OptimizerCls(
            model_pipeline,
            {key: hparams[key]},
            scoring="accuracy",
            verbose=0,
            n_iter=n_iter,
        )
        search.fit(X, y)

        params = random_search_results_to_dict(search)
        series = pd.Series(params)
        df_row = pd.DataFrame(series).T

        df = pd.concat([df, df_row])

    return df
