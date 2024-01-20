# Structure

0. `environment.yml` - environment file
1. `automl_approach` - folder with notebooks for automl frameworks
2. `classical_approach` - folder with scripts for classical approach, results from them are saved using **MLflow** library in `mlruns` and notebook that creates file with predictions using our chosen model
3. `mlruns` - results from classical approach
4. `processed_data` - data for classical approach
5. `data_preparation.ipynb` - notebook with data preparation for classical approach

# Create environment using conda
Run this commands in `Kody` directory:
```python
conda env create -f environment.yml
```

# Running MLflow UI

Run this command (at the level of that file):
```python
mlflow ui --backend-store-uri mlruns
```
