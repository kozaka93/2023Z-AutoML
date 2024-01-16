import models
from common import load_train_data
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

random_state = 42

# load data and labels from files
X, y = load_train_data()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

# preprocess data and select features
preprocessing = models.scaler_boruta_preprocessing(random_state)

# create model
model = ExtraTreesClassifier(random_state=random_state, n_jobs=-1)

# tune hyperparams using random search  
model = models.bayes_tune_et(model, random_state)

# create and fit final pipeline
model = make_pipeline(*preprocessing, model)
model.fit(X_train, y_train)

# evaluate model on test data
y_pred = model.predict(X_test)
print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}')
print(f"Best params: {model.named_steps['bayessearchcv'].best_params_}")
print(f"Best score: {model.named_steps['bayessearchcv'].best_score_}")