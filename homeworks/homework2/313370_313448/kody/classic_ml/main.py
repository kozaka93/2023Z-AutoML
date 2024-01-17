import argparse
import os

import xgboost
from classicml.train import train_model
from classicml.tune import get_best_hparams
from classicml.valid import validate_model
from dotenv import load_dotenv

from src.classicml.train import DEFAULT_HPARAMS
from src.preprocessing import preprocess_data, remap_to_zero_one
from src.utils import configure_logging, load_data_df, save_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgbm")
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--imputation_strategy", type=str, default="mean")
    parser.add_argument("--corr_threshold", type=float, default=0.9)
    parser.add_argument("--pca_threshold", type=float, default=0.99)
    parser.add_argument("--feature_selection", type=str, default="tree")
    parser.add_argument("--num_features", type=int, default=10)
    return parser.parse_args()


def main(args):
    configure_logging()
    load_dotenv()

    X_train, X_val, y_train, y_val, X_test = load_data_df(os.getenv("DATA_DIR"))
    X_train, X_val, X_test, y_train, y_val = preprocess_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        imputation_strategy=args.imputation_strategy,
        corr_threshold=args.corr_threshold,
        pca_threshold=args.pca_threshold,
        feature_selection=args.feature_selection,
        num_features=args.num_features,
    )

    if args.hpo:
        best_model_hparams = get_best_hparams(
            args.model, args.n_trials, X_train, y_train, X_val, y_val
        )
    else:
        best_model_hparams = DEFAULT_HPARAMS[args.model]

    best_model = train_model(args.model, best_model_hparams, X_train, y_train)
    y_val_pred, val_acc = validate_model(best_model, X_val, y_val)

    if args.model == "xgb":
        y_test_pred_proba = best_model.predict(xgboost.DMatrix(X_test))
    elif args.model == "lgbm":
        y_test_pred_proba = best_model.predict(X_test)
    save_predictions(y_test_pred_proba, os.getenv("DATA_DIR"))


if __name__ == "__main__":
    main(parse_args())
