# Structure

1. `automl_approach` - folder with notebooks for automl frameworks
2. `classical_approach` - folder with scripts for classical approach, results from them are saved using **MLflow** library in `mlruns` and notebook that creates file with predictions using our chosen model
3. `mlruns` - results from classical approach
4. `processed_data` - data for classical approach
5. `data_preparation.ipynb` - notebook with data preparation for classical approach

# Running MLflow UI

Run this command (at the level of that file):
```python
mlflow ui --backend-store-uri mlruns
```

In order to run own scripts path in `mlruns/669314470488795517/meta.yaml` needs to be corrected. In production environment `mlruns` folder would not be pushed to the repository, but hosted in e.g. S3 AWS bucket.