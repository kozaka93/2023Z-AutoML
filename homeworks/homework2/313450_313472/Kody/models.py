from scipy.stats import loguniform, uniform, randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector as SFS, RFECV, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV, space
from boruta import BorutaPy


def scaler_boruta_preprocessing(random_state=None):
    return Pipeline(
        [
            ('scaling', StandardScaler()),
            ('feature_selection', BorutaPy(
                RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced', max_depth=5),
                n_estimators='auto',
                verbose=2,
                random_state=random_state
            )),
        ]
    )


def scaler_anova_preprocessing():
    return Pipeline(
        [
            ('scaling', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=13)),
        ]
    )


def rf_pca(random_state=None):
    return Pipeline(
        [
            ('preprocessing', StandardScaler()),
            ('pca', PCA(n_components='mle', random_state=random_state)),
            ('model', RandomForestClassifier(random_state=random_state)),
        ]
    )


def rf_sfs(random_state=None):
    rf1 = RandomForestClassifier(random_state=random_state)
    rf2 = RandomForestClassifier(random_state=random_state)

    return Pipeline(
        [
            ('preprocessing', StandardScaler()),
            ('sfs', SFS(rf1, n_features_to_select='auto', tol=1e-3, direction='forward', cv=5, scoring='balanced_accuracy', n_jobs=-1)),
            ('model', rf2),
        ]
    )


def rs_tune_rf(model, random_state=None):
    rs_params = {
        'n_estimators': randint(60, 2001),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': randint(10, 31),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 6),
        'max_features': randint(2, 22)
    }

    return RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def bayes_tune_rf(model, random_state=None):
    bayes_params = {
        'n_estimators': space.Integer(60, 2000),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': space.Integer(10, 30),
        'min_samples_split': space.Integer(2, 10),
        'min_samples_leaf': space.Integer(1, 5),
        'max_features': space.Integer(2, 21)
    }

    return BayesSearchCV(model, bayes_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def rs_tune_et(model, random_state=None):
    rs_params = {
        'n_estimators': randint(60, 2001),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': randint(10, 31),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 6),
        'max_features': randint(2, 22)
    }

    return RandomizedSearchCV(model, rs_params, n_iter=500, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def bayes_tune_et(model, random_state=None):
    bayes_params = {
        'n_estimators': space.Integer(60, 2000),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': space.Integer(10, 30),
        'min_samples_split': space.Integer(2, 10),
        'min_samples_leaf': space.Integer(1, 5),
        'max_features': space.Integer(2, 21)
    }

    return BayesSearchCV(model, bayes_params, n_iter=500, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def rs_tune_svc(model, random_state=None):
    rs_params = {
        'gamma': loguniform(4, 33),
        'C': loguniform(1, 257)
    }

    return RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def bayes_tune_svc(model, random_state=None):
    bayes_params = {
        'gamma': space.Real(4, 32, prior='log-uniform'),
        'C': space.Real(1, 256, prior='log-uniform')
    }

    return BayesSearchCV(model, bayes_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def rs_tune_hgb(model, random_state=None):
    rs_params = {
        'learning_rate': uniform(0.01, 1, 'log-uniform'),
        'max_iter': randint(100, 1501),
        'max_leaf_nodes': randint(8, 41),
        'min_samples_leaf': randint(1, 31),
        'l2_regularization': uniform(0.01, 1, 'log-uniform'),
        'max_bins': randint(2, 256)
    }

    return RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def bayes_tune_hgb(model, random_state=None):
    bayes_params = {
        'learning_rate': space.Real(0.01, 1, prior='log-uniform'),
        'max_iter': space.Integer(100, 1500),
        'max_leaf_nodes': space.Integer(8, 40),
        'min_samples_leaf': space.Integer(1, 30),
        'l2_regularization': space.Real(0.01, 1, prior='log-uniform'),
        'max_bins': space.Integer(2, 255)
    }

    return BayesSearchCV(model, bayes_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def rs_tune_gb(model, random_state=None):
    rs_params = {
        'learning_rate': uniform(0.01, 1, 'log-uniform'),
        'n_estimators': randint(100, 1501),
        'max_leaf_nodes': randint(8, 41),
        'min_samples_leaf': randint(1, 31),
        'subsample': uniform(0.1, 1),
        'max_depth': randint(10, 31),
    }


    return RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)

def bayes_tune_gb(model, random_state=None):
    bayes_params = {
        'learning_rate': space.Real(0.01, 1, prior='log-uniform'),
        'n_estimators': space.Integer(100, 1500),
        'max_leaf_nodes': space.Integer(8, 40),
        'min_samples_leaf': space.Integer(1, 30),
        'subsample': space.Real(0.01, 1),
        'max_depth': space.Integer(10, 30),
    }

    return BayesSearchCV(model, bayes_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def rs_tune_mlp(model, random_state=None):
    rs_params = {
        'learning_rate_init': uniform(1e-3, 1e-2),
        'batch_size': [16, 32, 64, 128, 256],
    }

    return RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)


def bayes_tune_mlp(model, random_state=None):
    bayes_params = {
        'learning_rate_init': space.Real(1e-3, 1e-2),
        'batch_size': [16, 32, 64, 128, 256],
    }

    return BayesSearchCV(model, bayes_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)
