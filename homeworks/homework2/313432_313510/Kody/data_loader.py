import os
import pandas as pd
import uuid
from pandas import DataFrame
from torch import Tensor

class DataLoader(): 
    @staticmethod
    def __add_unique_headers(df):
        unique_names = [f"column_{i}" for i in range(len(df.columns))]
        rename_dict = dict(zip(df.columns, unique_names))
        df = df.rename(columns=rename_dict)
        return df

    @staticmethod
    def read_train_data():
        data_folder = 'data'
        file_path = os.path.join(data_folder, 'artificial_train.data')

        df = pd.read_csv(file_path, delimiter=' ', header=None)
        df = df.drop(df.columns[-1], axis=1)
        df = DataLoader.__add_unique_headers(df)

        file_path = os.path.join(data_folder, 'artificial_train.labels')
        df_y = pd.read_csv(file_path, delimiter=' ', header=None)

        df_y = df_y.rename(columns={0: 'target'})

        return (df, df_y)

    @staticmethod
    def read_test_data():
        data_folder = 'data'
        file_path = os.path.join(data_folder, 'artificial_test.data')

        df = pd.read_csv(file_path, delimiter=' ', header=None)
        df = df.drop(df.columns[-1], axis=1)
        df = DataLoader.__add_unique_headers(df)
        return df

    @staticmethod
    def save_results(results : DataFrame | Tensor):
        results = results.rename({0: '313432_313510'})
        results.to_csv('results/' + str(uuid.uuid4()), index=False)