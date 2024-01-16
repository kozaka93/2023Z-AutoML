import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import genfromtxt, savetxt

# load full set as train or split 75-25?
def load_data(full_dataset : bool = False):
    X = pd.read_csv('artificial_train.data', header=None, delim_whitespace=True)
    y = pd.read_csv('artificial_train.labels', header=None, delim_whitespace=True)
    final_X_test = pd.read_csv('artificial_test.data', header=None, delim_whitespace=True)

    X.replace(-1, 0, inplace=True)
    y.replace(-1, 0, inplace=True)
    final_X_test.replace(-1, 0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    if full_dataset:
        return X, X_test, y, y_test, final_X_test
    else:
        return X_train, X_test, y_train, y_test, final_X_test
    
# Save in final format
def save_data(data, path):
    data_transformed = []
    for elem in data:
        data_transformed.append(elem[1])

    savetxt(path, data_transformed, fmt='%f', comments='', header='\"313453_313457\"')

