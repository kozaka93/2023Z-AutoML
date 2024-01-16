from datetime import datetime
import pandas as pd

def save_data_to_csv(data, path):
    f = open(path, mode='w')
    data.to_csv(f, index=False)
    f.close()


def save_prediction_to_file(predictions, filename):
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{formatted_date}.txt"

    with open(filename, 'w') as file:
        file.write('"313319_313400"\n')

    predictions.to_csv(filename, index=False, mode='a', header=False, float_format='%.18f')


def print_results(transformers, analisys_list):
    for transformer, analisys in zip(transformers, analisys_list):
        mean_score = analisys['mean_test_score'].mean()
        std_score = analisys['std_test_score'].mean()
        best_score = analisys['mean_test_score'].max()
        print(f"{transformer}: mean: {mean_score}, std: {std_score}, best result: {best_score}")


def print_best_results(analisys, transformer_name):
    bests_transformer = analisys.loc[analisys['rank_test_score'] <= 7, ['param_stack__estimators', 'mean_test_score', 'rank_test_score']]
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.max_rows', None) 
    print(f"Best results for {transformer_name}")
    print(bests_transformer.to_string(index=False))

def searchMaxRowByColumn(dataframe, columnName):
    return dataframe[dataframe[columnName] == dataframe[columnName].max()].iloc[0]