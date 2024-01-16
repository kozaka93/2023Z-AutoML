import os
import sys

sys.path.append("../")

import argparse
import numpy as np
from utils import load_data_df
from preprocessing import preprocess_data
from train import train_models_and_return_best
from autogluon.tabular import TabularDataset
import h2o

DATAPATH = os.path.join(os.getcwd(), "../../", "data")


def main(args):
    X_train, X_val, y_train, y_val, X_test = load_data_df(DATAPATH)

    X_train, X_val, X_test, y_train, y_val = preprocess_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        scale_features_flag=args.scale_features_flag,
        remove_higly_correlated_features_flag=args.reduce_dimensionality_flag,
        feature_selection=args.feature_selection,
        corr_threshold=args.corr_threshold,
        imputatuion_strategy=args.imputatuion_strategy,
    )

    best_model, model_name = train_models_and_return_best(
        X_train, X_val, y_train, y_val, args.training_time_mins
    )

    if model_name == "tpot":
        y_test_pred_proba = best_model.predict_proba(X_test).iloc[:, 1]

    elif model_name == "h2o":
        h2o.init()
        X_test_h2o = h2o.H2OFrame(X_test)
        y_test_pred_proba = best_model.predict(X_test_h2o).as_data_frame()["p1"]

    elif model_name == "auto_gluon":
        X_test_tabular = TabularDataset(X_test)
        y_test_pred_proba = best_model.predict_proba(X_test_tabular).iloc[:, 1]

    else:
        print("Best model not found")

    np.savetxt(
        os.path.join(DATAPATH, f"{model_name}_y_test_pred_proba.txt"),
        y_test_pred_proba,
        fmt="%.6f",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML")
    parser.add_argument("--training-time-mins", default=10, type=int)
    parser.add_argument("--scale-features-flag", type=bool, default=True)
    parser.add_argument(
        "--remove-higly-correlated-features-flag", type=bool, default=True
    )
    parser.add_argument("--reduce-dimensionality-flag", type=bool, default=True)
    parser.add_argument("--imputatuion-strategy", type=str, default="mean")
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument(
        "--feature-selection", type=str, default="pca", options=["pca", "kbest", "tree"]
    )
    args = parser.parse_args()
    main(args)
