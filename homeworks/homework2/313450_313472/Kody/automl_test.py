import os
import models
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from common import load_test_data, load_train_data, main_path
from datetime import datetime

random_state = 42

# load data and labels from files
X_train, y_train = load_train_data()
X_test = load_test_data()

# preprocess data and select features
preprocessing = models.scaler_anova_preprocessing()
X_train = preprocessing.fit_transform(X_train, y_train)
X_test = preprocessing.transform(X_test)

X_train = TabularDataset(X_train)
X_train.columns = X_train.columns.astype(str)
X_train['target'] = y_train

X_test = TabularDataset(X_test)
X_test.columns = X_test.columns.astype(str)

# create model
model = TabularPredictor(label='target', eval_metric='balanced_accuracy', problem_type='binary')

# create and fit final pipeline
model.fit(X_train, presets='best_quality', num_stack_levels=3, num_bag_folds=5, time_limit=8*3600)

# evaluate model on test data
y_pred_proba = model.predict_proba(X_test)[1]
np.savetxt(os.path.join(main_path, 'output', f'313450_313472_artifical_automl_prediction_{datetime.now().strftime("%H_%M_%S")}.txt'), y_pred_proba, header='313450_313472', comments='', fmt='%s')