# AutoML-2023Z-hw1
Code for homework no. 1 in AutoL course in WUT 

## Methods

We want to collect:
* performance on all tested hpo on all data for each model
* performance on all tested hpo on all data for each model when all hp are fixed to best value but one

## Terminology

We consider $\theta^*$ as default hyperparameters values from `scikit-learn` package. $\theta^{*(j)}$ is computed and saved in `results_bayes` and `results_random` files for all datasets. $\theta^{*(j)}_i$ is computed and saved in `results_single_bayes` and `results_single_random`. 

$\theta^*$, $\theta^{*(j)}$ and $\theta^{*(j)}_i$ are calculated in `bin.py`. Calculation's details are covered in `config/` directory. Defining new model requires updating `hpo_tune/init.py` file (creating instance of class with predefined attributes). $d^{(j)}$ and $d^{(j)}_i$ are computed in `analysis.ipynb`.


## Setup

```
make setup
```

## Experiments

```
make run_experiments
```

## Important note

Best hparams ($\theta^*$) are calculated by function `load_best_params` from `utils` and are not dowloaded to any csv during experiments phase.