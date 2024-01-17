import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
import logging
from typing import Tuple


def map_features(y_train, y_val):
    """
    Maps labels to 0 and 1.
    """
    y_train = y_train.map({-1: "0", 1: "1"})
    y_val = y_val.map({-1: "0", 1: "1"})

    return y_train, y_val


def remove_constant_features(
    X_train,
    X_val,
    X_test,
):
    """
    Removes constant features from all dataframes.
    """
    columns_to_drop = [
        column for column in X_train.columns if len(X_train[column].unique()) == 1
    ]
    logging.info(f"Removing {len(columns_to_drop)} constant features.")
    X_train = X_train.drop(columns=columns_to_drop, axis=1)
    X_val = X_val.drop(columns=columns_to_drop, axis=1)
    X_test = X_test.drop(columns=columns_to_drop, axis=1)

    return X_train, X_val, X_test


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs feature scaling with z-scores on all dataframes.
    """
    logging.info("Scaling features.")
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_val, X_test


def impute_missing_values(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imputes missing values in all dataframes.
    """
    logging.info(f"Imputing missing values with {strategy}.")

    for column in X_train.columns:
        if X_train[column].isnull().all():
            X_train.drop(column, axis=1, inplace=True)
            X_val.drop(column, axis=1, inplace=True)
            X_test.drop(column, axis=1, inplace=True)

    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_train)

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val = pd.DataFrame(
        imputer.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_val, X_test


def select_features_kbest(
    X_train,
    y_train,
    X_val,
    X_test,
    num_features: int = 50,
):
    """
    Selects features in all dataframes based on the KBest method.
    Args:
        num_features (int): Number of features to select (k).
    """
    logging.info(f"Selecting {num_features} features based on KBest feature selection.")
    selector = SelectKBest(f_classif, k=num_features)
    X_train = selector.fit_transform(X_train, y_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    return X_train, X_val, X_test


def select_features_tree(
    X_train,
    y_train,
    X_val,
    X_test,
    model=DecisionTreeClassifier(),
    num_features: int = 50,
):
    """
    Selects features in all dataframes based on the training a
    tree-based model and their feature importance.
    Args:
        model (sklearn.model): Tree-based model to train.
        num_features (int): Number of features to select.
    """
    logging.info(
        f"Selecting {num_features} features based on tree-based feature importance."
    )
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    indices = feature_importances.argsort()[-num_features:][::-1]
    X_train = X_train.iloc[:, indices]
    X_val = X_val.iloc[:, indices]
    X_test = X_test.iloc[:, indices]

    return X_train, X_val, X_test


def remove_higly_correlated_features(
    X_train,
    X_val,
    X_test,
    corr_threshold: float = 0.9,
):
    """
    Removes highly correlated features from all dataframes.
    Args:
        corr_threshold (float): Threshold for correlation.
    """
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    columns_to_drop = [
        column for column in upper.columns if any(upper[column] > corr_threshold)
    ]
    logging.info(f"Removing {len(columns_to_drop)} highly correlated features.")
    X_train = X_train.drop(columns=columns_to_drop, axis=1)
    X_val = X_val.drop(columns=columns_to_drop, axis=1)
    X_test = X_test.drop(columns=columns_to_drop, axis=1)

    return X_train, X_val, X_test


def reduce_dimensionality(
    X_train, X_val, X_test, explained_variance_threshold: float = 0.8
):
    """
    Reduces the number of features based on the Principal Component Analysis.
    """
    pca = PCA(n_components=explained_variance_threshold)

    pca.fit(X_train)

    X_train_reduced = pd.DataFrame(pca.transform(X_train))
    X_val_reduced = pd.DataFrame(pca.transform(X_val))
    X_test_reduced = pd.DataFrame(pca.transform(X_test))

    logging.info(
        f"Reduced number of features from {X_train.shape[1]} to {X_train_reduced.shape[1]}."
    )

    return X_train_reduced, X_val_reduced, X_test_reduced


def remove_outliers(X_train, y_train, method="isolation_forest"):
    """
    Removes outliers from all dataframes.
    """
    if method == "isolation_forest":
        clf = IsolationForest(
            random_state=42,
            contamination="auto",
            n_estimators=100,
            max_samples=256,
            n_jobs=-1,
        )
        outliers = np.where(clf.fit_predict(X_train) == -1)[0]
    else:
        raise ValueError("Invalid outlier removal method.")

    logging.info(f"Removing {len(outliers)} outliers in trainset.")
    X_train = X_train.drop(outliers, axis=0)
    y_train = np.delete(y_train, outliers, axis=0)

    return X_train, y_train


def remap_to_zero_one(y_train, y_val):
    """
    Remap labels from -1, 1 to 0, 1.
    """
    logging.info("Remapping labels from -1, 1 to 0, 1.")
    y_train = np.where(y_train == -1, 0, 1)
    y_val = np.where(y_val == -1, 0, 1)

    return y_train, y_val


def preprocess_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    scale_features_flag=True,
    remove_higly_correlated_features_flag=True,
    remove_outliers_flag=True,
    imputatuion_strategy="mean",
    corr_threshold=0.9,
    feature_selection="kbest",
    num_features=50,
    pca_threshold=0.8,
    feature_selection_model=None,
):
    """
    Preprocesses all dataframes.
    Args:
        scale_features_flag (bool): Flag for scaling features.
        remove_higly_correlated_features_flag (bool): Flag for removing highly correlated features.
        reduce_dimensionality_flag (bool): Flag for reducing dimensionality.
        imputatuion_strategy (str): Strategy for imputing missing values.
        corr_threshold (float): Threshold for correlation.
        feature_selection (str): Feature selection method.
        num_features (int): Number of features to select.
        pca_threshold (float): Threshold for PCA.
        feature_selection_model (sklearn.model): Model for feature selection.
    Returns:
        X_train, X_val, X_test, y_train, y_val (pd.DataFrame): Preprocessed dataframes.
    """

    logging.info("Preprocessing data.")
    X_train, X_val, X_test = remove_constant_features(X_train, X_val, X_test)
    X_train, X_val, X_test = impute_missing_values(
        X_train, X_val, X_test, imputatuion_strategy
    )
    if scale_features_flag:
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    if remove_higly_correlated_features_flag:
        X_train, X_val, X_test = remove_higly_correlated_features(
            X_train, X_val, X_test, corr_threshold
        )
    if remove_outliers_flag:
        X_train, y_train = remove_outliers(X_train, y_train)

    if feature_selection == "kbest":
        X_train, X_val, X_test = select_features_kbest(
            X_train, y_train, X_val, X_test, num_features
        )
    elif feature_selection == "tree":
        model = (
            DecisionTreeClassifier()
            if not feature_selection_model
            else feature_selection_model
        )
        X_train, X_val, X_test = select_features_tree(
            X_train, y_train, X_val, X_test, model, num_features
        )
    elif feature_selection == "pca":
        X_train, X_val, X_test = reduce_dimensionality(
            X_train, X_val, X_test, pca_threshold
        )
    else:
        raise ValueError("Invalid feature selection method.")

    y_train, y_val = map_features(y_train, y_val)

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_val, X_test, y_train, y_val
