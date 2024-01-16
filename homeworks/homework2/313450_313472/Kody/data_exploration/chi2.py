from sklearn.feature_selection import chi2

def filter_chi2(X, y, threshold=0.1):
    _, p = chi2(X, y)
    return p < threshold