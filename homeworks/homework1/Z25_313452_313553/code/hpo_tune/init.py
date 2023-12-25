import yaml
import pandas as pd
import pickle as pkl

from loguru import logger
from openml import datasets
from pathlib import Path
from functools import partial
from typing import Union

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def download_datasets(
    data_config_path: Union[str, Path], download_dir: Union[str, Path]
) -> None:
    download_dir = Path(download_dir)

    with open(data_config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    data_ids = map(lambda x: x["id"], config["datasets"])

    for data_id in data_ids:
        logger.info(f"downloading dataset id={data_id}")

        openml_dataset = datasets.get_dataset(
            data_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        openml_data = openml_dataset.get_data()[0]

        openml_data.iloc[:, -1] = pd.get_dummies(openml_data.iloc[:, -1]).astype(int)

        file_path = download_dir / (openml_dataset.name + ".csv")

        openml_data.to_csv(file_path, index=False)


def download_models(model_config_path="config/models.yaml"):
    with open(model_config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    logger.info(f"Save model: logistic-regression")
    Model = partial(
        LogisticRegression,
        **{"max_iter": 10000, "solver": "saga", "penalty": "elasticnet"},
    )
    with open(config["models"][0]["path"], "wb") as f:
        pkl.dump(Model, f)

    logger.info(f"Save model: svm")
    Model = partial(
        SVC,
        **{"max_iter": 10000, "kernel": "poly"},
    )
    with open(config["models"][1]["path"], "wb") as f:
        pkl.dump(Model, f)

    logger.info(f"Save model: gradient boosting")
    Model = partial(
        GradientBoostingClassifier,
        **{"n_iter_no_change": 1000},
    )
    with open(config["models"][2]["path"], "wb") as f:
        pkl.dump(Model, f)


def main(
    data_config_path="config/data.yaml",
    model_config_path="config/models.yaml",
    download_dir="data/input",
) -> None:
    download_datasets(data_config_path, download_dir)
    download_models(model_config_path)


if __name__ == "__main__":
    main()
