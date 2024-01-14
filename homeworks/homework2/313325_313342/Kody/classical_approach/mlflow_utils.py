import pathlib

DATA_PATH = (
    pathlib.Path(__file__).parent.parent / "processed_data"
)
MLFLOW_TRACKING_URI = pathlib.Path(__file__).parent.parent / "mlruns"
