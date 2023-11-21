## to w sumie niepotrzebne, ale zostawiamy, bo Daniel bardzo się starał i dzielnie walczył ze sklejaniem dataframów

import os
import pandas as pd

hw1_path = os.path.join(os.path.expanduser("~"), f'Desktop\\AutoML\\HW1')

def get_dataset_results(dataset, algorithm):
    path = os.path.join(hw1_path, f'results\\{algorithm}\\random\\{dataset}.csv')
    dataframe = pd.read_csv(path)

    dataframe.rename(columns={'mean_test_score': f'{dataset}_score'}, inplace=True)

    return dataframe[['params', f'{dataset}_score']]

def get_bayes_initial_params(datasets, algorithm):
    
    results = get_dataset_results(datasets[0], algorithm)

    for dataset in datasets[1:]:
        results = pd.merge(results, get_dataset_results(dataset, algorithm), on='params', how='outer')

    # path = os.path.join(hw1_path, f'results\\{algorithm}\\scores-merged.csv')
    # results.to_csv(path, index=False)

    results['mean_score'] = results.iloc[:, 1:].mean(axis=1)

    results_mean = results[['params', 'mean_score']]

    results_mean_sorted = results_mean.sort_values(by='mean_score', ascending=False)

    # path = os.path.join(hw1_path, f'results\\{algorithm}\\scores-merged-mean.csv')
    # results_mean_sorted.to_csv(path, index=False)

    return results_mean_sorted['params'].iloc[0]