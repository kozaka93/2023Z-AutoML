from typing import Callable

import numpy as np
from sklearn.pipeline import Pipeline


def calculate_aggregate_tunability(tunability):
    return np.mean(tunability)


def calculate_tunability_on_each_dataset(
    train_datasets,
    test_datasets,
    best_configs,
    optimal_config,
    get_model_pipeline: Callable[[], Pipeline],
):
    tunability = []
    for train_dataset, test_dataset, best_config in zip(
        train_datasets, test_datasets, best_configs
    ):
        optimal_model: Pipeline = get_model_pipeline()
        optimal_model.set_params(**optimal_config)

        optimal_model.fit(train_dataset[0], train_dataset[1])

        best_model_for_dataset = get_model_pipeline()
        best_model_for_dataset.set_params(**best_config)

        best_model_for_dataset.fit(train_dataset[0], train_dataset[1])

        optimal_score = optimal_model.score(test_dataset[0], test_dataset[1])
        best_score = best_model_for_dataset.score(test_dataset[0], test_dataset[1])
        tunability_on_dataset = optimal_score - best_score

        tunability.append(tunability_on_dataset)

        print(f"d^j: {optimal_score} - {best_score} = {tunability_on_dataset}")
    return tunability
