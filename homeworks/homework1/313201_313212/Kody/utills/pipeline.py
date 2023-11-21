import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_column_transformer() -> ColumnTransformer:
    num_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    col_trans = ColumnTransformer(
        transformers=[
            (
                "num_pipeline",
                num_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            ("cat_pipeline", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
        n_jobs=-1,
    )
    return col_trans


def evaluate_model(model: Pipeline, X_train, y_train, X_test, y_test) -> float:
    model.fit(
        X=X_train,
        y=y_train,
    )
    return model.score(
        X=X_test,
        y=y_test,
    )


# def get_bayes_model(
#     pipeline: Pipeline,
#     search_space: Dict[str, Any],
#     n_iter=50,
# ) -> BayesSearchCV:
#     return BayesSearchCV(
#         pipeline,
#         # [(space, # of evaluations)]
#         search_spaces=search_space,
#         n_iter=n_iter,
#         n_jobs=-1,
#         cv=5,  # Set cv=None to disable cross-validation
#     )
