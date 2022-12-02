# Example:
# python src/model_summary.py --model_dir='results/trained_models' --test_data='data/processed/credit_test_df.csv' --output_dir='results/test_scores'


# author: Ken Wang
# date: 2022-11-24

"""This script generate evaluation figures for a binary classification model.
Usage: model_summary.py --model_dir=<model_dir> --test_data=<test_data> --output_dir=<output_dir>

Options:
--model_dir=<model_dir>    folder with joblib model files on disk
--test_data=<test_data>    test target data file in csv format
--output_dir=<output_dir>  output directory to put summary figures in

From the root of the repository, run:
 python src/model_summary.py --model_dir results/trained_models/ --test_data data/processed/credit_test_df.csv --output_dir results/model_summary
"""

import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
from joblib import load as joblib_load

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

opt = docopt(__doc__)

def get_classification_report(model, X_test, y_test, sheet_path):
    class_report = classification_report(
        y_test,
        model.predict(X_test),
        target_names=["non-default", "default"],
        output_dict=True,
    )
    print(f"type(class_report) = {type(class_report)}")
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df = class_report_df.sort_values(by=["f1-score"], ascending=False)
    pd.DataFrame.to_csv(class_report_df, sheet_path)


def get_roc_auc(model, X_test, y_test, figure_path):
    fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(X_test))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))

    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="Threshold 0.5",
    )
    plt.legend(loc="best")
    plt.savefig(figure_path)
    plt.clf()


def get_pr_curve(model, X_test, y_test, figure_path):
    precision, recall, thresholds = precision_recall_curve(
        y_test, model.decision_function(X_test)
    )
    plt.plot(precision, recall, label="PR curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot(
        precision_score(y_test, model.predict(X_test)),
        recall_score(y_test, model.predict(X_test)),
        "or",
        markersize=10,
        label="Threshold 0.5",
    )
    plt.legend(loc="best")
    plt.savefig(figure_path)
    plt.clf()


def get_confusion_matrix(model, X_test, y_test, figure_path):
    my_confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(
        my_confusion_matrix, display_labels=["Non Default", "Default"]
    ).plot()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.clf()


def load_test_data(test_data_csv=None):
    target_col_name = "default_payment_next_month"
    test_df = pd.read_csv(test_data_csv)
    X_test, y_test = test_df.drop(columns=[target_col_name]), test_df[target_col_name]
    return X_test, y_test


def load_model(model_path):
    model = joblib_load(model_path)
    return model


def get_test_f1_scores(model_dir=None, test_data_path=None, sheet_path=None):
    X_test, y_test = load_test_data(test_data_path)
    # for all models plot the F-1 score
    model_name_list = [
        "svc",
        "logistic_regression",
        "dummy_classifier",
        "random_forest",
        "knn",
    ]
    # model_name -> test f1 score
    test_f1_score_dict = {}
    for model_name in model_name_list:
        print(f"\n---------- model_name = {model_name} ------------\n")
        model_path = os.path.join(model_dir, model_name + ".joblib")
        model = load_model(model_path)

        y_hat_test = model.predict(X_test)

        my_f1_score = f1_score(y_test, y_hat_test)
        print(f"my_f1_score = {my_f1_score}")
        test_f1_score_dict[model_name] = [my_f1_score]
    test_f1_score_df = pd.DataFrame.from_dict(
        test_f1_score_dict, orient="index", columns=["test_f1_score"]
    )
    pd.DataFrame.to_csv(test_f1_score_df, sheet_path)


def get_train_f1_scores(cv_scores_sheet_path=None, output_sheet_path=None):
    cv_scores_df = pd.read_csv(cv_scores_sheet_path, index_col=0)
    train_f1_scores_raw = cv_scores_df.loc["test_f1", :]
    model_names_map = {
        "Dummy": "dummy_classifier",
        "SCV Optimized": "svc",
        "kNN Optimized": "knn",
        "Logistic Regression Otimized": "logistic_regression",
        "RandomForestClassifier Optimized": "random_forest",
    }
    model_names_list = list(model_names_map.keys())
    train_f1_scores_series = train_f1_scores_raw[model_names_list]

    train_f1_scores_series = (
        train_f1_scores_series.str.split(" ", expand=True)
        .iloc[:, 0]
        .rename("train_f1_score")
    )
    train_f1_scores_series = train_f1_scores_series.rename(model_names_map)
    train_f1_scores_series = train_f1_scores_series.astype(np.float64)
    train_f1_scores_df = pd.DataFrame(train_f1_scores_series)
    pd.DataFrame.to_csv(train_f1_scores_df, output_sheet_path)


def get_train_test_f1_socres(
    train_scores_sheet, test_scores_sheet, train_test_scores_sheet
):
    train_scores = pd.read_csv(train_scores_sheet, index_col=0)
    test_scores = pd.read_csv(test_scores_sheet, index_col=0)
    train_test_scores = pd.concat([train_scores, test_scores], axis=1)
    pd.DataFrame.to_csv(train_test_scores, train_test_scores_sheet)


def plot_f1_scores(train_test_scores_path, plot_path):
    train_test_scores = pd.read_csv(train_test_scores_path, index_col=0)
    index_list = train_test_scores.index.tolist()
    index_titlle_list = [model_name.title().replace('_', ' ') for model_name in index_list]
    x = np.arange(5)
    y1 = train_test_scores.loc[:,'train_f1_score'].tolist()
    y2 = train_test_scores.loc[:,'test_f1_score'].tolist()
    width = 0.4
    plt.bar(x-0.2, y1, width)
    plt.bar(x+0.2, y2, width)
    plt.xticks(x,  index_titlle_list,rotation = 45)
    plt.legend(['train', 'test'])
    plt.title('F-1 Scores for Different Models')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()


def main(model_dir=None, test_data_path=None, output_dir_path=None):
    """
    Driver function to get the output of the summary
    and save it in the local file system.
    Parameters
    ----------
    model_dir : string
    test_data_path : string
    output_dir_path: string
    """

    train_f1_scores_path = os.path.join(output_dir_path, "train_f1_scores.csv")
    test_f1_scores_path = os.path.join(output_dir_path, "test_f1_scores.csv")

    train_test_f1_scores_path = os.path.join(
        output_dir_path, "train_test_f1_scores.csv"
    )

    get_train_f1_scores("results/cross_validation_results.csv", train_f1_scores_path)

    get_test_f1_scores(model_dir, test_data_path, test_f1_scores_path)

    get_train_test_f1_socres(
        train_f1_scores_path, test_f1_scores_path, train_test_f1_scores_path
    )

    f1_score_plot_path = os.path.join(output_dir_path, "train_test_f1_scores.png")
    plot_f1_scores(train_test_f1_scores_path, f1_score_plot_path)

    X_test, y_test = load_test_data(test_data_path)

    # for the best model `svc` make some figures
    best_model_name = "svc"

    model_path = os.path.join(model_dir, best_model_name + ".joblib")
    model = load_model(model_path)

    rfc_model_name = 'random_forest'
    rfc_model_path = os.path.join(model_dir, rfc_model_name + '.joblib')
    rfc_model = load_model(rfc_model_path)

    get_classification_report(
        model,
        X_test,
        y_test,
        os.path.join(output_dir_path, best_model_name + "_classification_report.csv"),
    )
    print("saved classification report")

    get_confusion_matrix(
        model,
        X_test,
        y_test,
        os.path.join(output_dir_path, best_model_name + "_confusion_matrix.png"),
    )
    print("saved svc confusion matrix png")

    get_confusion_matrix(
        rfc_model,
        X_test,
        y_test,
        os.path.join(output_dir_path, rfc_model_name + "_confusion_matrix.png"),
    )

    print("saved rfc confusion matrix png")

    get_pr_curve(
        model,
        X_test,
        y_test,
        os.path.join(output_dir_path, best_model_name + "_precision_recall_curve.png"),
    )

    print("saved precision recall curve png")

    get_roc_auc(
        model,
        X_test,
        y_test,
        os.path.join(output_dir_path, best_model_name + "_roc_auc.png"),
    )
    print("saved ROC AUC png")


if __name__ == "__main__":
    if not os.path.exists(opt["--output_dir"]):
        os.makedirs(opt["--output_dir"])

    main(
        model_dir=opt["--model_dir"],
        test_data_path=opt["--test_data"],
        output_dir_path=opt["--output_dir"],
    )
