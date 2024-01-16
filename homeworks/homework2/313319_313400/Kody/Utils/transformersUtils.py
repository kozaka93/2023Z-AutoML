import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV


class OryginalData(TransformerMixin):     
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X 
    
    def get_name(self):
        return f'{type(self).__name__}'

class PCATransformer(TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca_model = PCA(n_components=n_components, random_state=42)

    def fit(self, X, y=None):
        self.pca_model.fit(X)
        return self

    def transform(self, X):
        return self.pca_model.transform(X)
    
    def get_name(self):
        return f'{type(self).__name__}_{self.n_components}'
    
class CORRTransformer(TransformerMixin):
    def __init__(self, corr_threshold):
        self.corr_threshold = corr_threshold
        self.features_to_remove = None

    def fit(self, X, y=None):
        correlation_matrix = np.corrcoef(X, rowvar=False)
        correlated_vars = np.where(np.abs(correlation_matrix) > self.corr_threshold)

        features_to_remove = []
        for var1, var2 in zip(*correlated_vars):
            if var1 != var2 and var1 < var2:
                features_to_remove.append(var2)

        self.features_to_remove = list(set(features_to_remove))
        return self

    def transform(self, X):
        if self.features_to_remove is not None:
             return X.drop(columns=X.columns[self.features_to_remove], inplace=False)
        return X
    
    def get_name(self):
        return f'{type(self).__name__}_{self.corr_threshold}'

class LassoSelector(TransformerMixin):
    def __init__(self, alphas=[0.01, 0.1, 1.0, 10.0], cv=5, random_state=42):
        self.lasso_cv_model = LassoCV(alphas=alphas, cv=cv, random_state=random_state)
        self.selected_feat = None

    def fit(self, X, y=None):
        self.lasso_cv_model.fit(X, y)
        sel = SelectFromModel(self.lasso_cv_model, prefit=True)
        sel.fit(X, y, random_state=self.lasso_cv_model.random_state)
        self.selected_feat = X.columns[(sel.get_support())]
        return self

    def transform(self, X):
        if self.selected_feat is not  None:
            return X.loc[:, self.selected_feat]
        return X
    
    def get_name(self):
        return f'{type(self).__name__}'

class RFCSelector(TransformerMixin):
    def __init__(self, param_grid=None, cv=5, random_state=42):
        self.param_grid = param_grid or {
            'n_estimators': [5, 10],
            'max_depth': [5, 2],
            'min_samples_split': [2, 5],
        }
        self.cv = cv
        self.random_state = random_state
        self.selected_feat = None

    def fit(self, X, y=None):
        rf_model = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(rf_model, self.param_grid, cv=self.cv)
        grid_search.fit(X, y)
        
        sel = SelectFromModel(grid_search.best_estimator_)
        sel.fit(X, y)
        self.selected_feat = X.columns[(sel.get_support())]

        return self

    def transform(self, X):
        if self.selected_feat is not None:
            return X.loc[:, self.selected_feat]
        return X 
    
    def get_name(self):
        return f'{type(self).__name__}'

transformers = [
    OryginalData(),
    PCATransformer(0.3),
    PCATransformer(0.4),
    PCATransformer(0.6),
    PCATransformer(0.8),
    PCATransformer(4),
    PCATransformer(5),
    PCATransformer(6),
    PCATransformer(7),
    CORRTransformer(0.05),
    CORRTransformer(0.075),
    CORRTransformer(0.7),
    LassoSelector(),
    RFCSelector(),
]