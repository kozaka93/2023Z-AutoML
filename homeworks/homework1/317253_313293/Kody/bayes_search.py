from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
import openml
import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def hypertuneParameters(algorithm, X, y, params, output_file) :
    search = BayesSearchCV(estimator = algorithm, search_spaces = params, n_iter=20, cv=5, n_jobs=-1, scoring='accuracy', random_state=42)
    search.fit(X, y)
    all_configs = search.cv_results_['params']
    all_scores = search.cv_results_['mean_test_score']
    with open(output_file, 'w') as f:
        f.write("All Configurations and Scores:\n")
        for config, score in zip(all_configs, all_scores):
            f.write(str(config) + ' - Score: ' + str(score) + '\n')

def getScoresFromFile(file_path) :
    # Read data from the file
    with open(file_path, 'r') as file:
        data = file.read()
    lines = data.strip().split('\n')
    print(lines)
    scores = [float(line.split(':')[-1]) for line in lines if line.strip() and line.split(':')[-1].strip()]
    return scores

def processAllDataSets(dataset_ids, dataset_targets) :
    datasets = []
    for id, col in zip(dataset_ids, dataset_targets) :
        dataset = openml.datasets.get_dataset(id)
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        y = X[col]
        X = X.drop(col, axis=1)
        X, y , _, _  = train_test_split(X, y,test_size=0.2, random_state=42)
        datasets.append(dataset)

    return datasets



num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',MinMaxScaler())])

cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot',OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline, make_column_selector( dtype_include= np.number)),
    ('cat_pipeline',cat_pipeline,make_column_selector( dtype_include= [object, 'category']))
    ],
    remainder='drop',
    n_jobs=-1)

dataset_ids = [1471, 4534, 1053, 1046]
dataset_targets = ['Class', 'Result', 'defects', 'state']

datasets = processAllDataSets(dataset_ids, dataset_targets)

#SVM

svm_pipeline = Pipeline([('preprocessing', col_trans), 
                               ('model', SVC())])
params_svm = {
    'model__C' : (1e-1, 1000.0, 'log-uniform'),
    'model__gamma' : (1e-4, 100.0, 'log-uniform'),
    'model__kernel' : ['linear','rbf','sigmoid'],
}

i = 1
for x, y in datasets :
    hypertuneParameters(svm_pipeline, x, y, params_svm,f"resultsDataset{i}SVM.txt")
    i+=1

# Random Forest
rf_pipeline = Pipeline([('preprocessing', col_trans), 
                               ('model', RandomForestClassifier())])
params_rf = {
    'model__n_estimators': (50, 200),
    'model__max_depth': (5, 25),
    'model__min_samples_split': (2, 10),
    'model__min_samples_leaf': (1, 4),
    'model__max_features': ['log2', 'sqrt', None],
    'model__bootstrap':[True, False],
}
i = 1
for x, y in datasets :
    hypertuneParameters(rf_pipeline, x, y, params_rf, f"resultsDataset{i}RF.txt")
    i+=1

# XGBoost
xgb_pipeline = Pipeline([('preprocessing', col_trans), 
                               ('model', XGBClassifier())])
algorithm_xgb = XGBClassifier()
params_xgb = {
    'model__n_estimators': (50, 200),
    'model__learning_rate': (0.01, 0.3),
    'model__max_depth': (3, 10),
    'model__lambda' : (0,200),
    'model__alpha' : (0, 200),
    'model__subsample': (0.5, 1),
    'model__gamma': (0, 5),
}
i = 1
for x, y in datasets :
    hypertuneParameters(xgb_pipeline, x, y, params_xgb, f"resultsDataset{i}XGB.txt")
    i+=1

#READING SCORES
scoresAllRF = []
for i in range(4):
    scores = getScoresFromFile(f"resultsDataset{i+1}RF.txt")
    scoresAllRF.extend(scores)

theta_rf = 0.9165127017545728 # Values of theta's for each algorithm taken directly from partner's calculations to create boxplots

scoresAllSVM = []
for i in range(4):
    scores = getScoresFromFile(f"resultsDataset{i+1}SVM.txt")
    scoresAllSVM.extend(scores)

theta_svm = 0.8133374006292702

scoresAllXGB = []
for i in range(4):
    scores = getScoresFromFile(f"resultsDataset{i+1}XGB.txt")
    scoresAllXGB.extend(scores)

theta_xgb = 0.9177681913688276





# PREPARING RESULTS FOR PLOTTING
results_rf = []
for score in scoresAllRF :
    results_rf.append(theta_rf - score)

results_xgb = []
for score in scoresAllXGB :
    results_xgb.append(theta_xgb - score)

results_svm = []
for score in scoresAllSVM :
    results_svm.append(theta_svm - score)

# BOXPLOT
plt.boxplot([results_rf, results_svm, results_xgb], labels=['Random Forrest', 'SVM', 'XGB'], )

plt.ylabel('Theta - Bayes Optimization Score', fontsize = 26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

plt.show()