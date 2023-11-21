from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def create_column_transformer():
    num_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer()),
        ('scale', MinMaxScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer()),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    col_trans = ColumnTransformer(transformers=[
        ('num_pipeline', num_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat_pipeline', cat_pipeline, make_column_selector(dtype_include=np.object_))
    ],
        remainder='drop',
        n_jobs=-1)

    return col_trans