import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

_TRAIN_DATA_PATH: Path = Path("../../data/artificial_train.data")
_TRAIN_LABELS_PATH: Path = Path("../../data/artificial_train.labels")

_TEST_DATA_PATH: Path = Path("../../data/artificial_test.data")


def load_train_data(path: str = _TRAIN_DATA_PATH) -> np.ndarray:
    df = pd.read_csv(path, delimiter=" ", header=None)
    df = df.to_numpy()[:, :-1]
    return df


def load_test_data(path: str = _TEST_DATA_PATH) -> np.ndarray:
    df = pd.read_csv(path, delimiter=" ", header=None)
    df = df.to_numpy()[:, :-1]
    return df


def load_train_labels(path: str = _TRAIN_LABELS_PATH) -> np.ndarray:
    df = pd.read_csv(path, delimiter=" ", header=None)
    df = df.to_numpy()[:]
    return df


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred)
