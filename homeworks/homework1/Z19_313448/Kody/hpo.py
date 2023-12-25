import hashlib
import json
import logging
import os
import random
import time
import warnings
from collections import Counter, namedtuple
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, ttest_ind, uniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

seed = 42
k_fold = 5
warnings.filterwarnings("ignore")
metric = "roc_auc"
n_iterations = 100
results_dir = "Wyniki"


# binary classification datasets from the OpenML-CC18 benchmarking suite
dataset_names = [
    "blood-transfusion-service-center",
    "credit-g",
    "diabetes",
    "churn",
    "ozone-level-8hr",
    "sick",
    "qsar-biodeg",
    "kc1",
]
model_names = ["logistic_regression", "random_forest", "xgb"]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Set seed to {seed}")


def configure_logging():
    logging.basicConfig(
        format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info("Configured logging")


def get_model(model_name, **model_hparams):
    if model_name == "logistic_regression":
        return LogisticRegression(**model_hparams)
    elif model_name == "random_forest":
        return RandomForestClassifier(**model_hparams)
    elif model_name == "xgb":
        return XGBClassifier(**model_hparams)
    else:
        raise ValueError(f"Unknown model {model_name}")


def get_dataset(dataset_name):
    logging.info(f"Downloading dataset {dataset_name}")
    dataset = openml.datasets.get_dataset(dataset_name)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    return X, y, categorical_indicator, attribute_names


def get_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("scale", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("cat_pipeline", categorical_transformer, make_column_selector(dtype_include=np.object_)),
        ],
        remainder="drop",
        n_jobs=-1,
    )
    return preprocessor


def get_model_pipeline(preprocessor, model):
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate(model, X, y, cv=5, scoring="roc_auc"):
    cv_results = cross_validate(model, X, y, scoring=scoring, cv=cv)
    return cv_results["test_score"]


def test_statistical_difference(sample1, sample2, alpha=0.05):
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=False)
    return p_value < alpha


def get_short_hash(params):
    params_string = json.dumps(params, sort_keys=True)
    hash_object = hashlib.md5(params_string.encode())
    return hash_object.hexdigest()


def get_optimal_defaults(model_names, dataset_names, results_dir):
    df = pd.DataFrame(columns=["model", "parameters_hash", "parameters_id", "parameters", "dataset", "mean_test_score"])

    for model_name in model_names:
        for dataset_name in dataset_names:
            results_path = os.path.join(results_dir, f"rs-{model_name}-{dataset_name}.pkl")
            rs_results = joblib.load(results_path)
            n_iterations = len(rs_results.cv_results_['mean_test_score'])
            cur_df = pd.DataFrame({
                "model": [model_name] * n_iterations, 
                "parameters_hash": [get_short_hash(params) for params in rs_results.cv_results_["params"]],
                "parameters_id": list(range(n_iterations)), 
                "parameters": rs_results.cv_results_["params"],
                "dataset": [dataset_name] * n_iterations,
                "mean_test_score": rs_results.cv_results_['mean_test_score']
            })
            df = pd.concat([df, cur_df])

    optimal_defaults_df = df[["model", "parameters_hash", "mean_test_score"]].groupby(["model", "parameters_hash"]).mean().reset_index().groupby("model").max()
    optimal_defaults_df = df.loc[df['parameters_hash'].isin(optimal_defaults_df['parameters_hash'].tolist()), ['model', 'dataset', 'parameters', 'mean_test_score']]
    return optimal_defaults_df

def get_package_defaults(model_name):
    model = get_model(model_name)
    return model.get_params()


def get_hyperparameters_ranges(model_name, format="bo"):
    if model_name == "logistic_regression":
        if format == "bo":
            return {
                "model__C": Real(1e-3, 1e3, prior="log-uniform"),
                "model__penalty": ["elasticnet"],
                "model__solver": ["saga"],
                "model__l1_ratio": Real(0.0, 1.0, prior="uniform"),
            }
        else:
            return {
                "model__C": loguniform(1e-3, 1e3),
                "model__penalty": ["elasticnet"],
                "model__solver": ["saga"],
                "model__l1_ratio": uniform(0.0, 1.0),
            }
    elif model_name == "random_forest":
        if format == "bo":
            return {
                "model__n_estimators": Integer(10, 1000),
                "model__max_depth": Integer(1, 100),
                "model__min_samples_split": Integer(2, 10),
                "model__criterion": ["gini", "entropy"],
                "model__min_samples_leaf": Integer(1, 10),
                "model__max_samples": Real(0.1, 1.0, prior="uniform"),
            }
        else:
            return {
                "model__n_estimators": list(range(10, 1001)),
                "model__max_depth": list(range(1, 101)),
                "model__min_samples_split": list(range(2, 11)),
                "model__criterion": ["gini", "entropy"],
                "model__min_samples_leaf": list(range(1, 11)),
                "model__max_samples": uniform(0.1, 0.9), # 0.1 - 1.0
            }
    elif model_name == "xgb":
        if format == "bo":
            return {
                "model__n_estimators": Integer(10, 1000),
                "model__max_depth": Integer(1, 15),
                "model__learning_rate": Real(1e-3, 1, prior="log-uniform"),
                "model__subsample": Real(0.1, 1.0, prior="uniform"),
                "model__colsample_bytree": Real(0.1, 1.0, prior="uniform"),
                "model__min_child_weight": Integer(1, 10),
                "model__reg_alpha": Real(1e-4, 1e4, prior="log-uniform"),
                "model__reg_lambda": Real(1e-4, 1e4, prior="log-uniform"),
            }
        else:
            return {
                "model__n_estimators": list(range(10, 1001)),
                "model__max_depth": list(range(1, 16)),
                "model__learning_rate": loguniform(1e-3, 1),
                "model__subsample": uniform(0.1, 0.9),
                "model__colsample_bytree": uniform(0.1, 0.9),
                "model__min_child_weight": list(range(1, 11)),
                "model__reg_alpha": loguniform(1e-4, 1e4),
                "model__reg_lambda": loguniform(1e-4, 1e4),
            }
    else:
        raise ValueError(f"Unknown model {model_name}")


def get_optimal_hyperparameters(optimzer):
    return optimzer.best_params_


def random_sampling(model, X, y, model_hparams_grid, cv=5, n_iterations=100, seed=42, scoring="roc_auc"):
    rs = RandomizedSearchCV(
        model,
        model_hparams_grid,
        random_state=seed,
        n_iter=n_iterations,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    rs.fit(X, y)
    return rs


# dummy classifier just to sample hyperparameters space
class DummyClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        counter = Counter(y)
        self.most_common = counter.most_common(1)[0][0]
        return np.array([self.most_common] * len(y))

    def predict(self, X):
        return np.array([self.most_common] * len(X))

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **params):
        self.kwargs.update(params)
        return self


# wrapper for compatibility with sklearn API
SearchCVResult = namedtuple("SearchCVResult", ["cv_results_", "best_params_", "best_score_"])


def randomly_sample_hyperparameters_space(model_hparams_grid, n_iterations=100, seed=42):
    dummy_model = DummyClassifier()
    dummy_X = np.random.rand(100, 10)
    dummy_y = np.random.randint(0, 2, 100)
    random_search = random_sampling(
        dummy_model,
        dummy_X,
        dummy_y,
        model_hparams_grid,
        n_iterations=n_iterations,
        seed=seed,
        scoring="accuracy",  # to not raise errors
    )
    search_space = random_search.cv_results_["params"]
    return search_space


def evaluate_grid(model, X, y, model_hparams_grid, cv=5, scoring="roc_auc", seed=42):
    mean_test_score = []
    std_test_score = []

    for params_set in model_hparams_grid:
        model.set_params(**params_set)
        model.fit(X, y)
        test_score = evaluate(model, X, y, cv=cv, scoring=scoring)
        mean_test_score.append(test_score.mean())
        std_test_score.append(test_score.std())

    best_idx = np.argmax(mean_test_score)
    best_params = model_hparams_grid[best_idx]
    best_score = mean_test_score[best_idx]

    return SearchCVResult(
        cv_results_={
            "params": model_hparams_grid,
            "mean_test_score": np.array(mean_test_score),
            "std_test_score": np.array(std_test_score),
        },
        best_params_=best_params,
        best_score_=best_score,
    )


def bayesian_sampling(model, X, y, model_hparams_grid, cv=5, n_iterations=100, scoring="roc_auc", seed=42):
    bs = BayesSearchCV(
        model,
        model_hparams_grid,
        random_state=seed,
        n_iter=n_iterations,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    bs.fit(X, y)
    return bs


def plot_optimization(rs, bo):
    fig, ax = plt.subplots(figsize=(8, 6))

    y_rs = [np.maximum.accumulate(rs[dataset_name] / rs[dataset_name].max()) for dataset_name in rs.keys()]
    y_rs_mean = np.mean(y_rs, axis=0)
    y_rs_std = np.std(y_rs, axis=0)

    y_bo = [np.maximum.accumulate(bo[dataset_name] / bo[dataset_name].max()) for dataset_name in bo.keys()]
    y_bo_mean = np.mean(y_bo, axis=0)
    y_bo_std = np.std(y_bo, axis=0)

    ax.plot(y_rs_mean, "ro-", label="Random search")
    ax.fill_between(range(len(y_rs_mean)), y_rs_mean - y_rs_std, y_rs_mean + y_rs_std, alpha=0.1, color="r")
    ax.plot(y_bo_mean, "bo-", label="Bayesian optimization")
    ax.fill_between(range(len(y_bo_mean)), y_bo_mean - y_bo_std, y_bo_mean + y_bo_std, alpha=0.1, color="b")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("% of Best ROC AUC")
    ax.set_xticks(range(len(y_rs_mean)))
    ax.legend(loc="lower right")
    return fig


def plot_tunability(results, sampling_method="random_search"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=results, x="algorithm", y=f"{sampling_method}_tunability", hue="algorithm", showfliers=False, ax=ax
    )
    sns.swarmplot(data=results, x="algorithm", y=f"{sampling_method}_tunability", color="k", ax=ax)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("ROC AUC Tunability")
    fig.tight_layout()
    return fig


def save_results(optimizer, save_dir, file_name):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    joblib.dump(optimizer, save_path)
    logging.info(f"Saved results to {save_path}")


def main():
    configure_logging()
    set_seed(seed)

    start_time = time.time()

    results = []
    for model_name in model_names:
        logging.info(f"Starting search for {model_name}")
        rs_hyperparameters_ranges = get_hyperparameters_ranges(model_name, format="rs")
        rs_search_space = randomly_sample_hyperparameters_space(
            rs_hyperparameters_ranges, n_iterations=n_iterations, seed=seed
        )
        for dataset_name in dataset_names:
            X, y, _, _ = get_dataset(dataset_name)
            y = LabelEncoder().fit_transform(y)

            # default hyperparameters
            default_hparams = get_package_defaults(model_name)
            model = get_model(model_name, **default_hparams)
            preprocessor = get_preprocessor()
            model_pipeline = get_model_pipeline(preprocessor, model)
            default_score = evaluate(model_pipeline, X, y, scoring=metric, cv=k_fold).mean()
            logging.info(f"Default {k_fold}-fold cross-validation ROC AUC: {default_score:.3f}")

            # random search
            rs_results = evaluate_grid(
                model_pipeline,
                X,
                y,
                rs_search_space,
                cv=k_fold,
                scoring=metric,
                seed=seed,
            )
            save_results(rs_results, results_dir, f"rs-{model_name}-{dataset_name}.pkl")
            logging.info(f"Random search best {k_fold}-fold cross-validation ROC AUC: {rs_results.best_score_:.3f}")

            # bayesian optimization
            model_hparams_grid = get_hyperparameters_ranges(model_name, format="bo")
            bo_results = bayesian_sampling(
                model_pipeline,
                X,
                y,
                model_hparams_grid,
                n_iterations=n_iterations,
                cv=k_fold,
                scoring=metric,
                seed=seed,
            )
            save_results(bo_results, results_dir, f"bo-{model_name}-{dataset_name}.pkl")
            logging.info(
                f"Bayesian optimization best {k_fold}-fold cross-validation ROC AUC: {bo_results.best_score_:.3f}"
            )

            results.append(
                {
                    "algorithm": model_name,
                    "dataset": dataset_name,
                    "package_default": default_score,
                    "random_search": rs_results.best_score_,
                    "random_search_package_tunability": rs_results.best_score_ - default_score,
                    "bayesian_optimization": bo_results.best_score_,
                    "bayesian_optimization_package_tunability": bo_results.best_score_ - default_score,
                }
            )
            logging.info(results[-1])

    results_df = pd.DataFrame(results)
    # save results
    results_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)

    # find optimal defaults
    optimal_defaults = get_optimal_defaults(model_names, dataset_names, results_dir)
    for _, row in optimal_defaults.iterrows():
        results_df.loc[(results_df['algorithm'] == row['model']) & (results_df['dataset'] == row['dataset']), 'optimal_default'] = row['mean_test_score']
    
    # tunability for optimal defaults
    results_df['random_search_optimal_tunability'] = results_df['random_search'] - results_df['optimal_default']
    results_df['bayesian_optimization_optimal_tunability'] = results_df['bayesian_optimization'] - results_df['optimal_default']

    # save results
    results_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)

    end_time = time.time()
    logging.info(f"Total time: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()
