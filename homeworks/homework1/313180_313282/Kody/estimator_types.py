from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from skopt.space import Real, Integer, Categorical


class EstimatorType:
    NEURAL = "Neural"
    SVR = "SVR"
    SGD = "SGD"
    ELASTIC_NET = "ElasticNet"


def get_estimator(estimator_type):
    if estimator_type == EstimatorType.NEURAL:
        return MLPRegressor()
    elif estimator_type == EstimatorType.SVR:
        return SVR()
    elif estimator_type == EstimatorType.SGD:
        return SGDRegressor()
    elif estimator_type == EstimatorType.ELASTIC_NET:
        return ElasticNet()
    else:
        raise ValueError("Unknown estimator type: " + str(estimator_type))


def get_parameters(estimator_type):
    if estimator_type == EstimatorType.NEURAL:
        return {
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1]
                }
    elif estimator_type == EstimatorType.SVR:
        return {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'C': [1, 10, 100, 1000]
                }
    elif estimator_type == EstimatorType.SGD:
        return {
                'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive']
                }
    elif estimator_type == EstimatorType.ELASTIC_NET:
        return {
                'alpha': [0.0001, 0.05, 0.1, 0.5, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
                }
    else:
        raise ValueError("Unknown estimator type: " + str(estimator_type))


def get_bayes_parameters(estimator_type):
    if estimator_type == EstimatorType.NEURAL:
        return {
                'n_hidden_layer': Categorical([(50, 50, 50), (50, 100, 50), (100,)]),
                'activation': Categorical(['tanh', 'relu']),
                'solver': Categorical(['sgd', 'adam']),
                'alpha': Real(0.0001, 0.05),
                'learning_rate': Categorical(['constant', 'adaptive']),
                'learning_rate_init': Real(0.001, 0.1)
                }
    elif estimator_type == EstimatorType.SVR:
        return {
                'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': Categorical(['scale', 'auto']),
                'C': Integer(1, 1000)
                }
    elif estimator_type == EstimatorType.SGD:
        return {
                'loss': Categorical(['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                'penalty': Categorical(['l2', 'l1', 'elasticnet']),
                'alpha': Real(0.0001, 0.05),
                'learning_rate': Categorical(['constant', 'adaptive'])
                }
    elif estimator_type == EstimatorType.ELASTIC_NET:
        return {
                'alpha': Real(0.0001, 1.0),
                'l1_ratio': Real(0.1, 0.9)
                }
    else:
        raise ValueError("Unknown estimator type: " + str(estimator_type))
