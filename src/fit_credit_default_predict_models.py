# Author: Arjun Radhakrishnan
# Date: 2022-11-24

"""Python script to train various models on the Credit Default Training Data.
   The models considered for training are Dummy Classifier for baseline, RandomForestClassifier,
   KNeighborsClassifier, SVC, and LogisticRegression.

Usage: fit_credit_default_predict_models.py --read_training_path=<read_training_path> [--write_model_path=<write_model_path>] [--write_score_path=<write_score_path>]

Options:
    --read_training_path=<read_training_path>               Path to the training data.
    --write_model_path=<write_model_path>                   Path (excluding filename) to save the serialized model file [default: ./results/trained_models/]
    --write_score_path=<write_score_path>                   Path (excluding filename) to save the training results [default: ./results/]

From the root of the repository, run:
 python src/fit_credit_default_predict_models.py --read_training_path data/processed/credit_train_df.csv --write_model_path results/trained_models/ --write_score_path results/
"""

# hardcode optimization parameters
CONST_N_ITER=10
CONST_CV=10

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump
from docopt import docopt

opt = docopt(__doc__)


def main(read_training_path, write_model_path, write_score_path):
    """
    Driver function that applies ml models on the data
    and saves the artifacts that are generated.

    Parameters
    ----------
    read_training_path : str
        Path to the training data.
    write_model_path : str
        Path to save the serialized model file [default: ./results/trained_models]
    write_score_path : str
        Path (excluding filename) to save the training results [default: ./results/]
    """

    results = {}

    scoring_metrics = ["f1", "accuracy", "precision", "recall"]

    target = "default_payment_next_month"

    train_df = pd.read_csv(read_training_path)

    x_train, y_train = train_df.drop(columns=[target]), train_df[target]

    scalable_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    categorical_feats = ["MARRIAGE"]

    binary_feats = ["SEX"]

    # Education is already encoded as an ordinal feature and the data is cleaned
    # in the preprocessing script.
    column_transformer = make_column_transformer(
        (StandardScaler(), scalable_features),
        (OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_feats),
        (
            OneHotEncoder(sparse=False, handle_unknown="ignore", drop="if_binary"),
            binary_feats,
        ),
        ("passthrough", ["EDUCATION"]),
    )

    # Dummy Classifier
    add_dummy_scores_to_results_and_save(
        results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
    )

    # RandomForestClassifier
    add_rfc_scores_to_results_and_save(
        results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
    )

    # kNN
    add_knn_scores_to_results_and_save(
        results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
    )

    # SVC
    add_svc_scores_to_results_and_save(
        results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
    )

    # Logistic Regression
    add_lr_scores_to_results_and_save(
        results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
    )

    pd.DataFrame.to_csv(
        pd.DataFrame(results), write_score_path + "cross_validation_results.csv"
    )

    run_tests(write_model_path, write_score_path)

    return


# Method referenced from DSCI 571, Lab 1, MDS
def mean_std_cross_val_scores(model, x_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    x_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, x_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def add_dummy_scores_to_results_and_save(
    results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
):
    """
    Helper function to add the dummy classifier cross validation scores
    to the final results and to save the model in the local file system.

    Parameters
    ----------
    results : dict
        Dictionary containing the final results
    scoring_metrics : list
        List of scoring metrics
    x_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array
        y in the training data
    column_transformer : sklearn column transformer
        the column transformer needed to transform the features.
    write_model_path : str
        Path where the model should be saved.
    """

    dummy_pipe = make_pipeline(column_transformer, DummyClassifier())

    results["Dummy"] = mean_std_cross_val_scores(
        dummy_pipe,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    dummy_pipe.fit(x_train, y_train)

    dump(dummy_pipe, write_model_path + "dummy_classifier.joblib")


def add_rfc_scores_to_results_and_save(
    results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
):
    """
    Helper function to add the Random Forest Classifier cross validation scores
    to the final results and to save the model in the local file system.

    Parameters
    ----------
    results : dict
        Dictionary containing the final results
    scoring_metrics : list
        List of scoring metrics
    x_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array
        y in the training data
    column_transformer : sklearn column transformer
        the column transformer needed to transform the features.
    write_model_path : str
        Path where the model should be saved.
    """

    forest_pipe = make_pipeline(column_transformer, RandomForestClassifier())

    # Add scores with default hyperparameters
    results["RandomForestClassifier"] = mean_std_cross_val_scores(
        forest_pipe,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    distributions = {
        "randomforestclassifier__class_weight": [None, "balanced"],
        "randomforestclassifier__max_depth": np.arange(6, 26, 2),
        "randomforestclassifier__max_features": [
            "sqrt",
            "log2",
            None,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
        ],
    }

    # Hyperparameter Optimization
    forest_random_search = RandomizedSearchCV(
        forest_pipe,
        param_distributions=distributions,
        cv=CONST_CV,
        n_jobs=-1,
        random_state=522,
        verbose=0,
        n_iter=CONST_N_ITER,
        scoring="f1",
    )

    forest_random_search.fit(x_train, y_train)

    results["RandomForestClassifier Optimized"] = mean_std_cross_val_scores(
        forest_random_search.best_estimator_,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    dump(
        forest_random_search.best_estimator_,
        write_model_path + "random_forest.joblib",
    )


def add_knn_scores_to_results_and_save(
    results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
):
    """
    Helper function to add the KNeighbors Classifier cross validation scores
    to the final results and to save the model in the local file system.

    Parameters
    ----------
    results : dict
        Dictionary containing the final results
    scoring_metrics : list
        List of scoring metrics
    x_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array
        y in the training data
    column_transformer : sklearn column transformer
        the column transformer needed to transform the features.
    write_model_path : str
        Path where the model should be saved.
    """

    knn_pipe = make_pipeline(column_transformer, KNeighborsClassifier())

    # Add scores with default hyperparameters
    results["kNN"] = mean_std_cross_val_scores(
        knn_pipe,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    grid_params = {
        "kneighborsclassifier__weights": ["uniform", "distance"],
        "kneighborsclassifier__n_neighbors": np.arange(4, 100, 4),
    }

    # Hyperparameter Optimization
    knn_grid_search = RandomizedSearchCV(
        knn_pipe,
        param_distributions=grid_params,
        cv=CONST_CV,
        scoring="f1",
        n_jobs=-1,
        random_state=522,
        n_iter=CONST_N_ITER,
    )

    knn_grid_search.fit(x_train, y_train)

    results["kNN Optimized"] = mean_std_cross_val_scores(
        knn_grid_search.best_estimator_,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    dump(knn_grid_search.best_estimator_, write_model_path + "knn.joblib")


def add_svc_scores_to_results_and_save(
    results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
):
    """
    Helper function to add the SVC Classifier cross validation scores
    to the final results and to save the model in the local file system.

    Parameters
    ----------
    results : dict
        Dictionary containing the final results
    scoring_metrics : list
        List of scoring metrics
    x_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array
        y in the training data
    column_transformer : sklearn column transformer
        the column transformer needed to transform the features.
    write_model_path : str
        Path where the model should be saved.
    """

    svc_pipe = make_pipeline(column_transformer, SVC())

    # Add scores with default hyperparameters
    results["SCV"] = mean_std_cross_val_scores(
        svc_pipe,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    distributions = {
        "svc__class_weight": [None, "balanced"],
        "svc__gamma": 10.0 ** np.arange(-3, 5),
        "svc__C": 10.0 ** np.arange(-3, 5),
    }

    # Hyperparameter Optimization
    svc_random_search = RandomizedSearchCV(
        svc_pipe,
        param_distributions=distributions,
        cv=CONST_CV,
        n_jobs=-1,
        random_state=522,
        scoring="f1",
        n_iter=CONST_N_ITER,
    )

    svc_random_search.fit(x_train, y_train)

    results["SCV Optimized"] = mean_std_cross_val_scores(
        svc_random_search.best_estimator_,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    dump(svc_random_search.best_estimator_, write_model_path + "svc.joblib")


def add_lr_scores_to_results_and_save(
    results, scoring_metrics, x_train, y_train, column_transformer, write_model_path
):
    """
    Helper function to add the Logistic Regression Classifier cross validation scores
    to the final results and to save the model in the local file system.

    Parameters
    ----------
    results : dict
        Dictionary containing the final results
    scoring_metrics : list
        List of scoring metrics
    x_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array
        y in the training data
    column_transformer : sklearn column transformer
        the column transformer needed to transform the features.
    write_model_path : str
        Path where the model should be saved.
    """

    lr_pipe = make_pipeline(
        column_transformer,
        LogisticRegression(random_state=522, n_jobs=-1, max_iter=1000),
    )

    # Add scores with default hyperparameters
    results["Logistic Regression"] = mean_std_cross_val_scores(
        lr_pipe,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    grid_params = {
        "logisticregression__class_weight": [None, "balanced"],
        "logisticregression__C": 10.0 ** np.arange(-3, 3),
    }

    # Hyperparameter Optimization
    lr_grid_search = RandomizedSearchCV(
        lr_pipe,
        param_distributions=grid_params,
        cv=CONST_CV,
        scoring="f1",
        n_jobs=-1,
        n_iter=CONST_N_ITER,
        random_state=522,
    )

    lr_grid_search.fit(x_train, y_train)

    results["Logistic Regression Otimized"] = mean_std_cross_val_scores(
        lr_grid_search.best_estimator_,
        x_train,
        y_train,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    dump(
        lr_grid_search.best_estimator_,
        write_model_path + "logistic_regression.joblib",
    )


def run_tests(write_model_path, write_score_path):
    """
    Tests to ensure all the required artifacts are created.

    Parameters
    ----------
    write_model_path : str
        Path where the models are saved.
    write_score_path : str
        Path where the score is saved.
    """

    assert os.path.isfile(
        write_score_path + "cross_validation_results.csv"
    ), "The cross validation results does not exist"

    assert os.path.isfile(
        write_model_path + "dummy_classifier.joblib"
    ), "The dummy_classifier model does not exist"

    assert os.path.isfile(
        write_model_path + "random_forest.joblib"
    ), "The random_forest model does not exist"

    assert os.path.isfile(
        write_model_path + "knn.joblib"
    ), "The knn model does not exist"

    assert os.path.isfile(
        write_model_path + "svc.joblib"
    ), "The svc model does not exist"

    assert os.path.isfile(
        write_model_path + "logistic_regression.joblib",
    ), "The logistic_regression model does not exist"

    cv_results_df = pd.read_csv(write_score_path + "cross_validation_results.csv")

    assert cv_results_df.shape == (
        10,
        10,
    ), "The shape of the final result dataframe is wrong."

    assert set(cv_results_df.columns) == set(
        [
            "Unnamed: 0",
            "Dummy",
            "RandomForestClassifier",
            "RandomForestClassifier Optimized",
            "kNN",
            "kNN Optimized",
            "SCV",
            "SCV Optimized",
            "Logistic Regression",
            "Logistic Regression Otimized",
        ]
    ), "The output file does not have all the results."


if __name__ == "__main__":
    if not os.path.exists(opt["--write_model_path"]):
        os.makedirs(opt["--write_model_path"])

    if not os.path.exists(opt["--write_score_path"]):
        os.makedirs(opt["--write_score_path"])

    main(
        opt["--read_training_path"],
        opt["--write_model_path"],
        opt["--write_score_path"],
    )
