from typing import Any, List, Tuple

from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline

from utills.pipeline import evaluate_model


def evaluate_config_on_each_dataset(
    train_datasets: List[Tuple[DataFrame, Series]],
    test_datasets: List[Tuple[DataFrame, Series]],
    get_model,
    config,
) -> List[float]:
    performances: List[float] = []
    for (X_train, y_train), (X_test, y_test) in zip(train_datasets, test_datasets):
        model = get_model()
        model.set_params(**config)
        performance: float = evaluate_model(
            model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        performances.append(performance)
    return performances


def find_optimal_configuration_for_all_datasets(
    config_space,
    train_datasets: List[Tuple[DataFrame, Series]],
    test_datasets: List[Tuple[DataFrame, Series]],
    get_model,
    summary_func,
) -> Tuple[Any, List[float]]:
    best_config = None
    best_summary_score = float("0")
    history_scores: List[float] = []
    for i, config in enumerate(config_space):
        print(i)
        scores: List[float] = evaluate_config_on_each_dataset(
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            get_model=get_model,
            config=config,
        )
        summary_score = summary_func(scores)
        history_scores.append(summary_score)
        if summary_score > best_summary_score:
            best_summary_score = summary_score
            best_config = config

    return (best_config, best_summary_score, history_scores)
