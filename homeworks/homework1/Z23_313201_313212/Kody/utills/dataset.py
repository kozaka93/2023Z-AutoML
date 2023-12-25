from typing import Tuple

import openml
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def load_dataset_from_id(id: int) -> DataFrame:
    return openml.datasets.get_dataset(dataset_id=id).get_data(
        dataset_format="dataframe"
    )[0]


def split_dataset(
    data: pd.DataFrame, class_: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X: pd.DataFrame = data.drop(labels=class_, axis=1)
    y: pd.DataFrame = data[class_]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
