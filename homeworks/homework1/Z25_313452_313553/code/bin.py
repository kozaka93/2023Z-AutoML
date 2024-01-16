import yaml
import pandas as pd

from datetime import datetime
from loguru import logger
from pathlib import Path
from typing import Union
import pickle as pkl

from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from hpo_tune.experiments import search_for_best_hp, search_single_best_hp
from hpo_tune.utils import (
    parse_hp_config_bayes,
    parse_hp_config_random,
    load_best_params,
)


def main(
    model_config_path: Union[str, Path],
    data_config_path: Union[str, Path],
    n_iter: int = 50,
    n_iter_single: int = 20,
):
    with open(model_config_path) as f:
        model_config = yaml.load(f, Loader=yaml.CLoader)

    with open(data_config_path) as f:
        data_config = yaml.load(f, Loader=yaml.CLoader)

    timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    # all

    df_random = pd.DataFrame()
    df_bayes = pd.DataFrame()
    for data_dir in data_config["datasets"]:
        for model_dir in model_config["models"]:
            df = pd.read_csv(data_dir["path"])
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            with open(model_dir["path"], "rb") as f:
                Model = pkl.load(f)

            hparams = model_dir["hyperparams"]
            hparams_random = parse_hp_config_random(hparams)
            hparams_bayes = parse_hp_config_bayes(hparams)

            logger.info(
                f'Start searching bayes. data={data_dir["name"]}, model={model_dir["name"]}. Type=all'
            )
            df = search_for_best_hp(
                Model, hparams_bayes, X, y, BayesSearchCV, n_iter=n_iter
            )
            df["model"] = model_dir["name"]
            df["data"] = data_dir["name"]
            df_bayes = pd.concat([df_bayes, df])

            logger.info(
                f'Start searching random. data={data_dir["name"]}, model={model_dir["name"]}. Type=all'
            )
            df = search_for_best_hp(
                Model, hparams_random, X, y, RandomizedSearchCV, n_iter=n_iter
            )
            df["model"] = model_dir["name"]
            df["data"] = data_dir["name"]
            df_random = pd.concat([df_random, df])

    df_random.to_csv(f"data/output/results_random-{timestamp}.csv")
    df_bayes.to_csv(f"data/output/results_bayes-{timestamp}.csv")

    # single

    df_random = pd.DataFrame()
    df_bayes = pd.DataFrame()
    for data_dir in data_config["datasets"]:
        for model_dir in model_config["models"]:
            df = pd.read_csv(data_dir["path"])
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            best_hp = load_best_params(
                model_dir["name"],
                f"data/output/results_random-{timestamp}.csv",
            )

            with open(model_dir["path"], "rb") as f:
                Model = pkl.load(f)

            hparams = model_dir["hyperparams"]
            hparams_random = parse_hp_config_random(hparams)
            hparams_bayes = parse_hp_config_bayes(hparams)

            logger.info(
                f'Start searching bayes. data={data_dir["name"]}, model={model_dir["name"]}. Type=single'
            )
            df = search_single_best_hp(
                Model,
                hparams_bayes,
                X,
                y,
                best_hp,
                BayesSearchCV,
                n_iter=n_iter_single,
            )
            df["model"] = model_dir["name"]
            df["data"] = data_dir["name"]
            df_bayes = pd.concat([df_bayes, df])

            logger.info(
                f'Start searching random. data={data_dir["name"]}, model={model_dir["name"]}. Type=single'
            )
            df = search_single_best_hp(
                Model,
                hparams_random,
                X,
                y,
                best_hp,
                RandomizedSearchCV,
                n_iter=n_iter_single,
            )
            df["model"] = model_dir["name"]
            df["data"] = data_dir["name"]
            df_random = pd.concat([df_random, df])

    df_random.to_csv(f"data/output/results_single_random-{timestamp}.csv")
    df_bayes.to_csv(f"data/output/results_signle_bayes-{timestamp}.csv")


if __name__ == "__main__":
    main("config/models.yaml", "config/data.yaml")
