import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
import matplotlib.pyplot as plt 


def perform_random_search(estimator,
                          parameters,
                          dataset_X,
                          dataset_y,
                          scoring='neg_mean_absolute_error',
                          n_iter=16,
                          random_state=0):
    random_search = RandomizedSearchCV(estimator=estimator,
                                       param_distributions=parameters,
                                       n_iter=n_iter,
                                       random_state=random_state,
                                       scoring=scoring)
    random_search.fit(dataset_X, np.ravel(dataset_y))
    return random_search.cv_results_['params'], random_search.cv_results_['mean_test_score']


def perform_grid_search(estimator,
                        parameters,
                        dataset_X,
                        dataset_y,
                        scoring='neg_mean_absolute_error'):
    grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring=scoring)
    grid_search.fit(dataset_X, np.ravel(dataset_y))
    return grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']


def perform_bayes_search(estimator,
                        search_spaces,
                        dataset_X,
                        dataset_y,
                        scoring='neg_mean_absolute_error',
                        n_iter=16,
                        random_state=0):
    bayes_search = BayesSearchCV(estimator=estimator,
                                 search_spaces=search_spaces,
                                 scoring=scoring,
                                 n_iter=n_iter,
                                 random_state=random_state)
    bayes_search.fit(dataset_X, np.ravel(dataset_y))
    return bayes_search.cv_results_['params'], bayes_search.cv_results_['mean_test_score']

def plot_bayes(key, dataset_names, bayes_search_results, random_search_best):
    for d in dataset_names:
        plt.plot(sorted(-bayes_search_results[key][d][1], reverse=True), label=d)

    plt.axhline(y=-random_search_best[key][1], color='red', linestyle='--')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=3, fancybox=True, shadow=True)

def get_tunability_for_datasetandmodel(estimator, dataset, results, base_results):
    return -(max(base_results[estimator][dataset][1]) - max(results[estimator][dataset][1]))
