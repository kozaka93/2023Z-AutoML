from operator import ge
from typing import Callable, List, Tuple

from pandas import DataFrame, Series
import pip
from sklearn.pipeline import Pipeline

from utills.pipeline import evaluate_model


def find_best_config_for_dataset_with_random_search(
    config_space,
    train_dataset: Tuple[DataFrame, Series],
    test_dataset: Tuple[DataFrame, Series],
    get_model,
):
    best_config = None
    best_score = float("0")
    iteration_scores: List[float] = []
    for config in config_space:
        model = get_model()
        model.set_params(**config)
        score: float = evaluate_model(
            model=model,
            X_train=train_dataset[0],
            y_train=train_dataset[1],
            X_test=test_dataset[0],
            y_test=test_dataset[1],
        )
        iteration_scores.append(score)
        if score > best_score:
            best_score = score
            best_config = config

    return (best_config, iteration_scores)


def find_best_configs_in_search_space_with_random_search(
    get_pipeline, config_space, train_datasets, test_datasets
):
    best_configs = []
    list_iteration_scores: List[List[float]] = []
    for i, (train_dataset, test_dataset) in enumerate(
        zip(train_datasets, test_datasets)
    ):
        best_config, iteration_scores = find_best_config_for_dataset_with_random_search(
            config_space=config_space,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            get_model=get_pipeline,
        )
        best_configs.append(best_config)
        list_iteration_scores.append(iteration_scores)
        pipeline = get_pipeline()
        pipeline.set_params(**best_config)
        pipeline.fit(train_dataset[0], train_dataset[1])
        print("dataset: " + str(i))
        print("score: " + str(pipeline.score(test_dataset[0], test_dataset[1])))
        print("best config: " + str(best_config))
    return best_configs, list_iteration_scores
