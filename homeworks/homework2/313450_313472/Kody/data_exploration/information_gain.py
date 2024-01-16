from sklearn.calibration import column_or_1d
from sklearn.feature_selection import mutual_info_regression

def filter_ig(X, y, random_state, threshold=1e-5):
    y = column_or_1d(y, warn=False)
    ig = mutual_info_regression(X, y, random_state = random_state)
    return ig > threshold