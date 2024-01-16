import os
import sys
sys.path.append(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'AutoML', 'HW2', 'data_exploration'))

from y_correlation import filter_cor
from information_gain import filter_ig
from chi2 import filter_chi2

def filter_features(X, y, random_state=None):
    cor_mask = filter_cor(X, y)
    ig_mask = filter_ig(X, y, random_state)
    chi2_mask = filter_chi2(X, y)
    return [cor and ig and chi for cor, ig, chi in zip(cor_mask, ig_mask, chi2_mask)]