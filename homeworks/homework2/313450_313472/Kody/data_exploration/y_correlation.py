def filter_cor(X, y, threshold=1e-3):
    cor = X.corrwith(y[0]).to_numpy()
    return abs(cor) > threshold