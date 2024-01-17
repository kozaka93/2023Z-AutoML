import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging


def load_data_df(
    data_dir: str, val_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from data directory.
    Args:
        data_dir (str): Path to data directory.
    """
    train = pd.read_csv(
        os.path.join(data_dir, "artificial_train.data"),
        sep=" ",
        names=[f"x_{i}" for i in range(500)],  # number based on the provided knowledge
    ).reset_index(drop=True)
    train_labels = pd.read_csv(
        os.path.join(data_dir, "artificial_train.labels"), sep=" ", names=["y"]
    ).reset_index(drop=True)
    X_test = pd.read_csv(
        os.path.join(data_dir, "artificial_test.data"),
        sep=" ",
        names=[f"x_{i}" for i in range(500)],
    ).reset_index(drop=True)

    # drop duplicates
    train_combined = pd.concat([train, train_labels], axis=1)
    train_combined.drop_duplicates(inplace=True)
    train_deduplicated = train_combined.iloc[:, :-1].reset_index(drop=True)
    train_labels_deduplicated = train_combined.iloc[:, -1].reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(
        train_deduplicated,
        train_labels_deduplicated,
        test_size=val_size,
        random_state=42,
    )

    return X_train, X_val, y_train, y_val, X_test


def save_predictions(y_test_pred_proba: np.ndarray, data_dir: str):
    """
    Saves predictions to disk.
    Args:
        y_test_pred_proba (np.ndarray): Predicted probabilities.
        data_dir (str): Path to data directory.
    """
    save_path = os.path.join(data_dir, "y_test_pred_proba.txt")
    np.savetxt(
        save_path,
        y_test_pred_proba,
        fmt="%.6f",
    )
    logging.info(f"Predictions saved to {save_path}")


def configure_logging():
    """
    Configures logging.
    """
    logging.basicConfig(
        format=(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info("Configured logging")
