import logging

import xgboost
from sklearn.metrics import balanced_accuracy_score


def validate_model(model, X_val, y_val):
    """
    Validates the model on the validation set.
    """
    logging.info("Validating model.")
    if isinstance(model, xgboost.core.Booster):
        X_val = xgboost.DMatrix(X_val)
    y_val_pred = model.predict(X_val).round()
    val_acc = balanced_accuracy_score(y_val, y_val_pred)
    logging.info(f"Validation balanced accuracy: {val_acc:.4f}")
    return y_val_pred, val_acc
