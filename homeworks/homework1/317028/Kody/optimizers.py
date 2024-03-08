import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import utils
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
import gc


class DefaultRandomOptimizer:
    """
    A class representing a default random optimizer.

    Parameters:
    - datasets (dict): A dictionary containing the datasets to be used for optimization.
    - default_params (dict): Default hyperparameters for the model.

    Methods:
    - optimize(n_iter=1000, cv=5, optimizer='random'): Runs the optimization process.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets: dict, default_params=None):
        self.model = None
        self.datasets = datasets
        self.scores = self._generate_scores_table()
        self._transformer = utils.create_column_transformer()
        self.default_params = default_params

    def optimize(self, n_iter=1000, cv=5, optimizer='random', default="random"):
        """
        Runs the optimization process.

        Parameters:
        - n_iter (int): Number of iterations to run.
        - cv (int): Number of cross-validation folds.
        - optimizer (str): Optimization algorithm to use.

        Returns:
        None
        """
        for i in range(n_iter):
            print(f"Running iteration {i+1}/{n_iter}")
            if default == "random":
                run_params = self._get_run_params()
                run_model = self.model(**run_params)
            run_results = {
                'hyperparameters': str(run_params),
                'mean_brier': 0,
                'mean_roc_auc': 0,
                'mean_accuracy': 0
            }
            for dataset_name, dataset in self.datasets.items():
                print(f"Running dataset {dataset_name}")
                X, y = dataset['X'], dataset['y']
                if default == "bayes":
                    run_params = self._get_run_params(dataset_name)
                    run_model = self.model(**run_params)
                kf = StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=42)
                brier, roc_auc, accuracy = 0, 0, 0

                for train_index, test_index in kf.split(X, y):
                    X_train_fold, X_test_fold = X.iloc[train_index,
                                                       :], X.iloc[test_index, :]
                    y_train_fold, y_test_fold = y[train_index], y[test_index]

                    self._transformer.fit(X_train_fold)
                    X_train_fold = self._transformer.transform(X_train_fold)
                    X_test_fold = self._transformer.transform(X_test_fold)

                    run_model.fit(X_train_fold, y_train_fold)
                    y_pred_proba = run_model.predict_proba(X_test_fold)[:, 1]
                    y_pred = run_model.predict(X_test_fold)

                    brier += brier_score_loss(y_test_fold, y_pred_proba)
                    roc_auc += roc_auc_score(y_test_fold, y_pred_proba)
                    accuracy += accuracy_score(y_test_fold, y_pred)
                    gc.collect()

                run_results[f"{dataset_name}_brier"] = brier / cv
                run_results[f"{dataset_name}_roc_auc"] = roc_auc / cv
                run_results[f"{dataset_name}_accuracy"] = accuracy / cv

                run_results['mean_brier'] += brier / cv
                run_results['mean_roc_auc'] += roc_auc / cv
                run_results['mean_accuracy'] += accuracy / cv

            run_results['mean_brier'] /= len(self.datasets)
            run_results['mean_roc_auc'] /= len(self.datasets)
            run_results['mean_accuracy'] /= len(self.datasets)

            self.scores = pd.concat([self.scores, pd.DataFrame(
                run_results, columns=self.scores.columns, index=[0])], ignore_index=True)
            gc.collect()

    def get_optimal_hyperparameters(self, metric: str):
        """
        Returns the optimal hyperparameters based on the specified metric.

        Parameters:
        - metric (str): The metric to optimize for.

        Returns:
        - tuple: A tuple containing the optimal hyperparameters and the corresponding metric value.
        """
        if metric in self.scores.columns:
            if "brier" in metric:
                best = self.scores.sort_values(
                    by=metric, ascending=True).iloc[0]
            else:
                best = self.scores.sort_values(
                    by=metric, ascending=False).iloc[0]
            return best['hyperparameters'], best[metric]
        else:
            raise ValueError(f"Metric {metric} not found in scores table.")

    def _generate_scores_table(self):
        """
        Generates an empty scores table.

        Returns:
        - pd.DataFrame: An empty DataFrame representing the scores table.
        """
        columns = ['hyperparameters', 'mean_brier',
                   'mean_roc_auc', 'mean_accuracy']

        for dataset in self.datasets.keys():
            columns.append(f"{dataset}_brier")
            columns.append(f"{dataset}_roc_auc")
            columns.append(f"{dataset}_accuracy")

        self.scores = pd.DataFrame(columns=columns)
        return self.scores

    def _get_run_params(self, dataset_name: str = None):
        """
        Gets the hyperparameters for a single run.

        Raises:
        - NotImplementedError: This method should be implemented in a subclass.

        Returns:
        None
        """
        raise NotImplementedError

    def set_default_params(self, params: dict):
        """
        Sets the default hyperparameters for the model.

        Parameters:
        - params (dict): A dictionary containing the hyperparameters.

        Returns:
        None
        """
        self.default_params = params


class DTCRandomOptimizer(DefaultRandomOptimizer):
    """
    Random optimizer for Decision Tree Classifier.

    Parameters:
    - datasets (list): List of datasets to be used for optimization.

    Methods:
    - optimize(n_iter=1000, cv=5, optimizer='random'): Runs the optimization process.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.model = DecisionTreeClassifier

    def _get_run_params(self, dataset_name: str = None):
        """
        Get random parameters for the optimization run.

        Returns:
            dict: Randomly generated parameters for the optimization run.
        """
        params = {
            'random_state': 42,
            'ccp_alpha': random.random(),
            'max_depth': random.randint(1, 30),
            'min_samples_leaf': random.randint(1, 60),
            'min_samples_split': random.randint(2, 60),
        }
        if self.default_params:
            dp = self.default_params
            if dataset_name:
                dp = dp[dataset_name]
            for key, value in dp.items():
                params[key] = value
        return params


class KNNRandomOptimizer(DefaultRandomOptimizer):
    """
    Random optimizer for K-Nearest Neighbors classifier.

    Parameters:
    - datasets (list): List of datasets to be used for optimization.

    Methods:
    - optimize(n_iter=1000, cv=5, optimizer='random'): Runs the optimization process.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets: dict):
        super().__init__(datasets)
        self.model = KNeighborsClassifier
        self._call_counter = 0

    def _get_run_params(self, dataset_name: str = None):
        self._call_counter += 1
        params = {
            'n_jobs': -1,
            'n_neighbors': self._call_counter,
        }
        return params


class XGBoostRandomOptimizer(DefaultRandomOptimizer):
    """
    Random optimizer for XGBoost classifier.

    Parameters:
    - datasets (list): List of datasets to be used for optimization.

    Methods:
    - optimize(n_iter=1000, cv=5, optimizer='random'): Runs the optimization process.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets: dict):
        super().__init__(datasets)
        self.model = XGBClassifier

    def _get_run_params(self, dataset_name: str = None) -> dict:
        """
        Generates random hyperparameters for the XGBoost classifier.

        Returns:
            dict: A dictionary containing the random hyperparameters.

        """
        params = {
            'random_state': 42,
            'n_estimators': random.randint(1, 5000),
            'learning_rate': random.uniform(1e-10, 1.0),
            'subsample': random.uniform(0.1, 1.0),
            'booster': random.choice(['gbtree', 'gblinear', 'dart']),
            'max_depth': random.randint(1, 15),
            'min_child_weight': random.uniform(1.0, 1e+7),
            'colsample_bytree': random.uniform(0.0, 1.0),
            'colsample_bylevel': random.uniform(0.0, 1.0),
            'reg_alpha': random.uniform(1e-10, 1e+10),
            'reg_lambda': random.uniform(1e-10, 1e+10),
        }
        if self.default_params:
            dp = self.default_params
            if dataset_name:
                dp = dp[dataset_name]
            for key, value in dp.items():
                params[key] = value
        return params


class RFRandomOptimizer(DefaultRandomOptimizer):
    def __init__(self, datasets: dict):
        super().__init__(datasets)
        self.model = RandomForestClassifier

    def _get_run_params(self, dataset_name: str = None) -> dict:
        """
        Generates random hyperparameters for the Random Forest classifier.

        Returns:
            dict: A dictionary containing the random hyperparameters.

        """
        params = {
            'n_jobs': -1,
            'random_state': 42,
            'n_estimators': random.randint(1, 2000),
            # 'subsample': random.uniform(0.1, 1.0),  # sample.fraction
            'max_depth': random.randint(1, 15),
            'min_samples_leaf': random.randint(1, 60),
            'min_samples_split': random.randint(2, 60),
        }
        if self.default_params:
            dp = self.default_params
            if dataset_name:
                dp = dp[dataset_name]
            for key, value in dp.items():
                params[key] = value
        return params


class DefaultBayesOptimizer:

    def __init__(self, datasets: dict):
        self.model = None
        self.datasets = datasets
        self.best_scores = self._generate_scores_table()
        self._transformer = utils.create_column_transformer()
        self.default_params = None
        self.iter_scores = None

    def optimize(self, n_iter=250, cv=5, scoring='roc_auc', n_jobs=-1):
        """
        Runs the optimization process on the datasets.

        Parameters:
        - n_iter (int): The number of iterations for the optimization process.
        - cv (int): The number of cross-validation folds.
        - scoring (str): The scoring metric to optimize for.
        - n_jobs (int): The number of parallel jobs to run.

        Returns:
        - None
        """
        search_space = self._generate_search_space()
        search_pipe = Pipeline(
            [('transformer', self._transformer), ('model', self.model())])
        bayes_search = BayesSearchCV(
            search_pipe,
            search_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )

        for dataset_name, dataset in self.datasets.items():
            print(f"Running dataset {dataset_name}")
            X, y = dataset['X'], dataset['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            bayes_search.fit(X_train, y_train)
            self._update_iter_scores(
                bayes_search.cv_results_, dataset_name, scoring)
            self._update_scores(bayes_search, dataset_name,
                                X_test, y_test)
            gc.collect()

    def get_optimal_hyperparameters(self, metric: str) -> pd.DataFrame:
        """
        Returns the optimal hyperparameters based on the specified metric for each unique value of best['dataset'].

        Parameters:
        - metric (str): The metric to optimize for.

        Returns:
        - pd.DataFrame: A DataFrame containing the optimal hyperparameters and the corresponding metric value for each unique value of best['dataset'].
        """
        if metric in self.scores.columns:
            if "brier" in metric:
                grouped_scores = self.scores.groupby('dataset').apply(
                    lambda x: x.loc[x[metric].idxmin()])
            else:
                grouped_scores = self.scores.groupby('dataset').apply(
                    lambda x: x.loc[x[metric].idxmax()])
            return grouped_scores
        else:
            raise ValueError(f"Metric {metric} not found in scores table.")

    def _update_iter_scores(self, iter_data, dataset_name: str, metric: str):
        """
        Update the iteration scores dataframe with the results from a single iteration.

        Args:
            iter_data (dict): Dictionary containing the iteration data.
            dataset_name (str): Name of the dataset.
            metric (str): Name of the metric.

        Returns:
            None
        """
        if self.iter_scores is None:
            self.iter_scores = pd.DataFrame(
                columns=['iter', 'dataset', 'hyperparameters', metric])
        for i in range(iter_data['mean_test_score'].shape[0]):
            self.iter_scores = pd.concat([self.iter_scores, pd.DataFrame(
                {'iter': i,
                 'dataset': dataset_name,
                 'hyperparameters': [dict(iter_data['params'][i])],
                 metric: iter_data['mean_test_score'][i]}, index=[0])], ignore_index=True)

    def _generate_scores_table(self):
        """
        Generates an empty DataFrame to store the optimization scores.

        Returns:
        - pd.DataFrame: An empty DataFrame with the columns for the optimization scores.
        """
        columns = ['hyperparameters', 'dataset',
                   'test_brier', 'test_roc_auc', 'test_accuracy']
        scores = pd.DataFrame(columns=columns)
        return scores

    def _update_scores(self, bayes_search: BayesSearchCV, dataset_name, X, y):
        """
        Updates the optimization scores with the results of a dataset.

        Parameters:
        - bayes_search (BayesSearchCV): The Bayesian search object used for optimization.
        - dataset_name (str): The name of the dataset.
        - X (array-like): The input features of the dataset.
        - y (array-like): The target variable of the dataset.

        Returns:
        - None
        """
        best_params = bayes_search.best_params_
        y_pred_proba = bayes_search.predict_proba(X)[:, 1]
        y_pred = bayes_search.predict(X)

        brier = brier_score_loss(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)

        dataset_scores = {
            'hyperparameters': str(best_params),
            'dataset': dataset_name,
            'test_brier': brier,
            'test_roc_auc': roc_auc,
            'test_accuracy': accuracy
        }

        self.best_scores = pd.concat([self.best_scores, pd.DataFrame(
            dataset_scores, columns=self.best_scores.columns, index=[0])], ignore_index=True)

    def _generate_search_space(self):
        """
        Generates the search space for hyperparameter optimization.

        Returns:
        - dict: A dictionary representing the search space for hyperparameter optimization.
        """
        raise NotImplementedError

    def set_default_params(self, params: dict):
        """
        Sets the default hyperparameters for the model.

        Parameters:
        - params (dict): A dictionary containing the default hyperparameters.

        Returns:
        - None
        """
        self.default_params = params


class DTCBayesOptimizer(DefaultBayesOptimizer):
    """
    A Bayesian optimizer for tuning hyperparameters of Decision Tree classifier.

    Attributes:
    - model: The machine learning model to be optimized.
    - datasets: A dictionary containing the datasets to be used for optimization.
    - scores: A DataFrame containing the optimization scores for each dataset.
    - default_params: The default hyperparameters for the model.

    Methods:
    - optimize(n_iter, cv, scoring, n_jobs): Runs the optimization process on the datasets.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.model = DecisionTreeClassifier

    def _generate_search_space(self):
        search_space = {
            'model__ccp_alpha': Real(0, 1),
            'model__max_depth': Integer(1, 30),
            'model__min_samples_leaf': Integer(1, 60),
            'model__min_samples_split': Integer(2, 60),
            'model__random_state': [42],
        }
        if self.default_params:
            for key, value in self.default_params.items():
                search_space[key] = value
        return search_space


class KNNBayesOptimizer(DefaultBayesOptimizer):
    """
    A Bayesian optimizer for tuning hyperparameters of KNN classifier.

    Attributes:
    - model: The machine learning model to be optimized.
    - datasets: A dictionary containing the datasets to be used for optimization.
    - scores: A DataFrame containing the optimization scores for each dataset.
    - default_params: The default hyperparameters for the model.

    Methods:
    - optimize(n_iter, cv, scoring, n_jobs): Runs the optimization process on the datasets.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.model = KNeighborsClassifier

    def _generate_search_space(self):
        search_space = {
            'model__n_neighbors': Integer(1, 30),
        }
        return search_space


class XGBBayesOptimizer(DefaultBayesOptimizer):
    """
    A Bayesian optimizer for tuning hyperparameters of XGBoost classifier.

    Attributes:
    - model: The machine learning model to be optimized.
    - datasets: A dictionary containing the datasets to be used for optimization.
    - scores: A DataFrame containing the optimization scores for each dataset.
    - default_params: The default hyperparameters for the model.

    Methods:
    - optimize(n_iter, cv, scoring, n_jobs): Runs the optimization process on the datasets.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.model = XGBClassifier

    def _generate_search_space(self):
        search_space = {
            'model__n_estimators': Integer(1, 5000),
            'model__learning_rate': Real(1e-10, 1.0, 'log-uniform'),  # eta
            'model__subsample': Real(0.1, 1.0, 'uniform'),
            'model__booster': ['gbtree', 'gblinear', 'dart'],
            'model__max_depth': Integer(1, 15),
            'model__min_child_weight': Real(1.0, 1e+7, 'log-uniform'),
            'model__colsample_bytree': Real(0.0, 1.0, 'uniform'),
            'model__colsample_bylevel': Real(0.0, 1.0, 'uniform'),
            'model__reg_alpha': Real(1e-10, 1e+10, 'log-uniform'),
            'model__reg_lambda': Real(1e-10, 1e+10, 'log-uniform'),
        }
        return search_space


class RFBayesOptimizer(DefaultBayesOptimizer):
    """
    A Bayesian optimizer for tuning hyperparameters of Random Forest classifier.

    Attributes:
    - model: The machine learning model to be optimized.
    - datasets: A dictionary containing the datasets to be used for optimization.
    - scores: A DataFrame containing the optimization scores for each dataset.
    - default_params: The default hyperparameters for the model.

    Methods:
    - optimize(n_iter, cv, scoring, n_jobs): Runs the optimization process on the datasets.
    - get_optimal_hyperparameters(metric): Returns the optimal hyperparameters based on the specified metric.
    - set_default_params(params): Sets the default hyperparameters for the model.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.model = RandomForestClassifier

    def _generate_search_space(self):
        search_space = {
            'model__n_estimators': Integer(1, 2000),
            # 'model__subsample': Real(0.1, 1.0, 'uniform'),  # sample.fraction
            'model__max_depth': Integer(1, 15),
            'model__min_samples_leaf': Integer(1, 60),
            'model__min_samples_split': Integer(2, 60),
            'model__random_state': [42],
        }
        return search_space
