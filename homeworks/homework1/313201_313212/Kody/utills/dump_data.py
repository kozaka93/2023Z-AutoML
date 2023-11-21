import os
from typing import List

import pandas as pd


def dump_tunability_to_csv(tunability_per_dataset: List[float], filepath: str):
    # Create the directory path
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame(
        {
            "Dataset": range(1, len(tunability_per_dataset) + 1),
            "Tunability": tunability_per_dataset,
        }
    )
    df.to_csv(filepath, index=False)


def dump_scores_to_csv(iteration_scores: List[float], filepath: str):
    # Create the directory path
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame(
        {"Iteration": range(1, len(iteration_scores) + 1), "Score": iteration_scores}
    )
    df.to_csv(filepath, index=False)


def dump_optimal_config_search_history(history: List[float], filepath: str):
    # Create the directory path
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame({"Iteration": range(1, len(history) + 1), "Score": history})
    df.to_csv(filepath, index=False)
