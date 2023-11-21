from typing import Any, Dict, List

import pip
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


def get_bayes_model(
    pipeline: Pipeline,
    search_space: Dict[str, Any],
    n_iter=50,
) -> BayesSearchCV:
    return BayesSearchCV(
        pipeline,
        # [(space, # of evaluations)]
        search_spaces=search_space,
        n_iter=n_iter,
        n_jobs=-1,
        cv=5,  # Set cv=None to disable cross-validation
        random_state=9999,
    )


def find_best_config_using_bayes(
    get_pipeline,
    search_space: Dict[str, Any],
    X: DataFrame,
    y: DataFrame,
    n_iter,
):
    pipeline: Pipeline = get_pipeline()
    opt: BayesSearchCV = get_bayes_model(pipeline, search_space, n_iter)
    opt.fit(X, y)
    iteration_scores = opt.cv_results_["mean_test_score"]

    # Optional: Print the score for each iteration
    for i, score in enumerate(iteration_scores):
        print(f"Iteration {i + 1}: Score = {score}")
    print("Best score:", opt.best_score_)
    print(opt.n_iter)
    return (dict(opt.best_params_), iteration_scores)


def find_best_configs_in_search_space_with_bayes(
    search_space, get_pipeline, train_datasets
):
    configs = []
    history: List[List[float]] = []
    for train in train_datasets:
        config, iteration_scores = find_best_config_using_bayes(
            get_pipeline, search_space[0], train[0], train[1], search_space[1]
        )
        configs.append(config)
        history.append(iteration_scores)

    return (configs, history)
