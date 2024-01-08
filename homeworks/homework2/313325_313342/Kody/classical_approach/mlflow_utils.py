import pathlib

DATA_PATH = (
    pathlib.Path(__file__).parent.parent / "processed_data/with_polynomial_features_2"
)
MLFLOW_TRACKING_URI = pathlib.Path(__file__).parent.parent / "mlruns"
