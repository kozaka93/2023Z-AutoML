import openml

def fetch_openml_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    data, _, _, _ = dataset.get_data(dataset_format="dataframe")
    return data

def prepare_data(data, target_column):
    y = data.loc[:, target_column]
    X = data.drop([target_column], axis=1)
    return X, y

def save_data_to_csv(data, path):
    f = open(path, mode='w')
    data.to_csv(f, index=False)
    f.close()


def create_gradientBoosting_tuning_hisotry_object_base():
    data = {
        'param_model__n_estimators': [],
        'param_model__min_samples_split': [],
        'param_model__min_samples_leaf': [],
        'param_model__max_depth': [],
        'param_model__learning_rate': [],
        'param_model__subsample': []
    }
    return data

def create_randomForest_tuning_hisotry_object_base():
    data = {
        'param_model__n_estimators': [],
        'param_model__min_samples_split': [],
        'param_model__min_samples_leaf': [],
        'param_model__max_features': [],
        'param_model__max_depth': []
    }
    return data


def create_SVC_tuning_hisotry_object_base():
    data = {
        'param_model__C': [],
        'param_model__kernel': [],
        'param_model__gamma': []
    }
    return data


def create_gradientBoosting_tuning_hisotry_object(cv):
    data = create_gradientBoosting_tuning_hisotry_object_base()
    data['mean_test_score'] = []
    data['std_test_score'] = []
    for i in range(cv):
        data[f'split{i}_test_score'] = []
    return data

def create_randomForest_tuning_hisotry_object(cv):
    data = create_randomForest_tuning_hisotry_object_base()
    data['mean_test_score'] = []
    data['std_test_score'] = []
    for i in range(cv):
        data[f'split{i}_test_score'] = []
    return data

def create_SVC_tuning_hisotry_object(cv):
    data = create_SVC_tuning_hisotry_object_base()
    data['mean_test_score'] = []
    data['std_test_score'] = []
    for i in range(cv):
        data[f'split{i}_test_score'] = []
    return data


def create_gradientBoosting_allDatasets_tuning_hisotry_object():
    data = create_gradientBoosting_tuning_hisotry_object_base()
    data['mean_all_datasets_test_score'] = []
    data['std_all_datasets_test_score'] = []
    return data  

def create_randomForest_allDatasets_tuning_hisotry_object():
    data = create_randomForest_tuning_hisotry_object_base()
    data['mean_all_datasets_test_score'] = []
    data['std_all_datasets_test_score'] = []
    return data     


def create_SVC_allDatasets_tuning_hisotry_object():
    data = create_SVC_tuning_hisotry_object_base()
    data['mean_all_datasets_test_score'] = []
    data['std_all_datasets_test_score'] = []
    return data    


def create_comparison_object():
    data = {
        'dataset': [],
        'randomSearch_Model_auc_test': [],
        'bayesOptimization_Model_auc_test': [],
        'default_Model_auc_test': []
    }
    return data