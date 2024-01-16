import os
import numpy as np
import models
from common import load_test_data, load_train_data, main_path
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

random_state = 42

# load data and labels from files
X_train, y_train = load_train_data()
X_test = load_test_data()

# preprocess data and select features
preprocessing = models.scaler_boruta_preprocessing(random_state)

# create model
model = ExtraTreesClassifier(random_state=random_state, n_jobs=-1)

# tune hyperparams using random search  
model = models.bayes_tune_et(model, random_state)

# create and fit final pipeline
model = make_pipeline(*preprocessing, model)
model.fit(X_train, y_train)

# save predict probabilities with default formatter
y_pred_proba = model.predict_proba(X_test)[:, 1]
np.savetxt(os.path.join(main_path, 'output', f'313450_313472_artifical_model_prediction_{datetime.now().strftime("%H_%M_%S")}.txt'), y_pred_proba, header='313450_313472', comments='', fmt='%s')