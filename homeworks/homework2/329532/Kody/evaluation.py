import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def search_knn(X, y, num_folds=5, njobs=8, verbose=1):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    param_grid = {
        "n_neighbors": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(
        knn,
        param_grid,
        scoring="balanced_accuracy",
        cv=skf,
        verbose=verbose,
        n_jobs=njobs,
    )
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def evaluate(
    X,
    y,
    num_folds=10,
    params_knn=None,
):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    knn_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn = KNeighborsClassifier(**params_knn)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_scores.append(balanced_accuracy_score(y_test, knn_pred))

    avg_knn_score = np.mean(knn_scores)

    return {"KNN": avg_knn_score}
