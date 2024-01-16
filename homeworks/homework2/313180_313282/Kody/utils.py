import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # for classification problems


train_X_path = "data/artificial_train.data"
train_y_path = "data/artificial_train.labels"


def get_aml_data():
    X = pd.read_csv(train_X_path, sep=' ', header=None)
    X = X.drop(X.columns[-1], axis=1)
    X = X.astype(int)
    
    y = pd.read_csv(train_y_path, sep=' ', header=None)

    return X, y


def get_top_columns(X, y, top_columns):
    selector = SelectKBest(score_func=f_classif, k=top_columns)
    selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]

    X_selected = X[selected_columns]

    return X_selected, selected_columns


def find_correlated_columns(X, threshold):
    correlation_matrix = X.corr()
    
    highly_correlated_columns = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                # for any subset of correlated colums, we keep the one with highest index (because j > i)
                colname = correlation_matrix.columns[i]
                highly_correlated_columns.add(colname)

    return highly_correlated_columns


def get_data_with_filtered_columns(top_columns=None, correlation_threshold=None):
    X, y = get_aml_data()
    X_dropped = X.drop(columns=find_correlated_columns(X, correlation_threshold)) if correlation_threshold is not None else X
    
    if top_columns is None:
        return X_dropped, y, X_dropped.columns
    
    X_sel, final_columns = get_top_columns(X_dropped, y, top_columns)
    return X_sel, y, final_columns
