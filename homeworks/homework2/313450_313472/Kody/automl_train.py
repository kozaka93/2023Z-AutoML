import models
from autogluon.tabular import TabularDataset, TabularPredictor
from common import load_train_data
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42

# load data and labels from files
X, y = load_train_data()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

preprocessing = models.scaler_boruta_preprocessing()
X_train = preprocessing.fit_transform(X_train, y_train)
X_test = preprocessing.transform(X_test)

X_train = TabularDataset(X_train)
X_train.columns = X_train.columns.astype(str)
X_train['target'] = y_train

X_test = TabularDataset(X_test)
X_test.columns = X_test.columns.astype(str)
X_test['target'] = y_test

# create model
model = TabularPredictor(label='target', eval_metric='balanced_accuracy', problem_type='binary')

# create and fit final pipeline
model.fit(X_train, presets='best_quality', num_stack_levels=3, num_bag_folds=5, time_limit=8*3600)

# evaluate model on test data
y_pred = model.predict(X_test)
print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}')