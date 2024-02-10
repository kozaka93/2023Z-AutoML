import openml
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    OutlierHandler is a transformer that handles outliers in a dataset.

    Parameters:
    -----------
    factor : float, optional (default=1.5)
        The factor used to determine the lower and upper bounds for outlier detection.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(X)
        Transform the data by clipping values outside the lower and upper bounds.

    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Calculate the IQR for each column
        self.lower_bound = np.percentile(
            X, 25, axis=0) - self.factor * np.subtract(*np.percentile(X, [75, 25], axis=0))
        self.upper_bound = np.percentile(
            X, 75, axis=0) + self.factor * np.subtract(*np.percentile(X, [75, 25], axis=0))
        return self

    def transform(self, X):
        """
        Transform the data by clipping values outside the lower and upper bounds.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_clipped : array-like, shape (n_samples, n_features)
            The transformed data with outliers clipped.
        """
        # Clip values outside the lower and upper bounds
        X_clipped = np.clip(X, self.lower_bound, self.upper_bound)
        return X_clipped


def get_dataset(identifier=None):
    """
    Get a dataset from OpenML and return it as a pandas DataFrame.

    Parameters
    ----------
    identifier:  int or str, optional
        The OpenML id or name of the dataset.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - X: pd.DataFrame
            The dataset features as a pandas DataFrame.
        - y: pd.Series
            The dataset target variable as a pandas Series.
        - categorical_indicator: list
            A list indicating whether each feature is categorical (True) or not (False).
        - attribute_names: list
            A list of attribute names corresponding to the dataset features.

    Notes
    -----
    This function retrieves a dataset from OpenML using the provided id and name. It then converts the dataset into a pandas DataFrame and returns it along with additional information about the dataset.
    - If both `id` and `name` are provided, `id` takes precedence.
    - If neither `id` nor `name` is provided, None is returned.
    """
    if not identifier:
        return None
    dataset = openml.datasets.get_dataset(identifier)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    return X, y, categorical_indicator, attribute_names

def create_column_transformer():
    """
    Create a column transformer for preprocessing numerical and categorical columns.

    Returns:
        transformer (ColumnTransformer): The column transformer object.
    """
    num_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_handler', OutlierHandler()),
            ('std_scaler', StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(
                handle_unknown='ignore')),
        ]
    )
    transformer = ColumnTransformer(
        [
            ('num', num_pipeline, make_column_selector(dtype_exclude='object')),
            ('cat', cat_pipeline, make_column_selector(dtype_include='object')),
        ],
        remainder='passthrough',
    )
    return transformer


def make_binary(y: pd.Series):
    """
    Map the target variable to a binary variable.

    Parameters
    ----------
    y : pd.Series
        The dataset target variable as a pandas Series.

    Returns
    -------
    pd.Series
        The mapped target variable as a pandas Series.

    Notes
    --------
    If the target variable has more than two unique values, the most frequent value is mapped to 1 and the rest are mapped to 0.
    """
    target = y.value_counts().index[0]
    return y.map(lambda x: 1 if x == target else 0)


def apply_boruta(X: pd.DataFrame, y: pd.Series, max_iter=100, max_depth=5):
    """
    Apply the Boruta algorithm to select features.

    Parameters
    ----------
    X : pd.DataFrame
        The dataset features as a pandas DataFrame.
    y : pd.Series
        The dataset target variable as a pandas Series.
    max_iter : int, optional
        The maximum number of iterations for the Boruta algorithm. Default is 100.
    max_depth : int, optional
        The maximum depth of the random forest classifier used in the Boruta algorithm. Default is 5.

    Returns
    -------
    list
        A list of selected features.
    """
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=max_depth)
    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=max_iter,
        verbose=1,
        random_state=42,
    )
    boruta.fit(X.values, y.values)
    selected_features = X.columns[boruta.support_].to_list()
    return selected_features