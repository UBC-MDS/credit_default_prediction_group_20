# Example:
# python model_summary.py --model_dir='../results/trained_models'  --test_data='../data/processed/credit_test_df.csv' --output_dir=output_dir_value


# author: Ken Wang
# date: 2022-11-24

"""This script generate evaluation figures for a binary classification model.
Usage: model_summary.py --model_dir=<model_dir>  --test_data=<test_data> --output_dir=<output_dir>

Options:
--model_dir=<model_dir> folder with joblib model files on disk
--X_test=<X_test>  test feature data file in csv format
--y_test=<y_test>  test target data file in csv format
--output_dir=<output_dir>  output directory to put summary figures in
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
    class_report = classification_report(y_test, model.predict(X_test), target_names=["non-default", "default"], output_dict=True)
    print(f'type(class_report) = {type(class_report)}')
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df = class_report_df.sort_values(by=['f1-score'], ascending=False)
    pd.DataFrame.to_csv(class_report_df, sheet_path)

def get_roc_auc(model, X_test, y_test, figure_path):
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))

    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best")
    plt.savefig(figure_path)
    plt.clf()


def get_pr_curve(model, X_test, y_test, figure_path):
    precision, recall, thresholds = precision_recall_curve(
        y_test, model.predict_proba(X_test)[:, 1]
    )
    plt.plot(precision, recall, label="PR curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot(
        precision_score(y_test, model.predict(X_test)),
        recall_score(y_test, model.predict(X_test)),
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best")
    plt.savefig(figure_path)
    plt.clf()

def get_confusion_matrix(model, X_test, y_test, figure_path):
    my_confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(
        my_confusion_matrix, display_labels=["Non Default", "Default"]
    ).plot()
    plt.savefig(figure_path)
    plt.clf()

def load_test_data(test_data_csv=None):
    target_col_name = 'default_payment_next_month'
    test_df = pd.read_csv(test_data_csv)
    X_test, y_test = test_df.drop(columns=[target_col_name]), test_df[target_col_name]
    return X_test, y_test

def load_model(model_path):
    model = joblib_load(model_path)
    return model


def get_test_f1_scores(model_dir=None, test_data_path=None, output_dir_path=None):
    X_test, y_test = load_test_data(test_data_path)

    # for all models plot the F-1 score
    model_name_list = ['svc', 'logistic_regression', 'dummy_classifier', 'random_forest', 'knn']
    for model_name in model_name_list:
        print(f'\n---------- model_name = {model_name} ------------\n')
        model_path = os.path.join(model_dir, model_name + '.joblib')
        model = load_model(model_path)

        y_hat_test = model.predict(X_test)
        print(f'y_hat_test = {y_hat_test}')
        print(f'type(y_hat_test) = {type(y_hat_test)}')
        print(f'y_hat_test.shape = {y_hat_test.shape}')

        my_f1_score = f1_score(y_test, y_hat_test)
        print(f'f1_score = {my_f1_score}')



def main(model_dir=None, test_data_path=None, output_dir_path=None):
    X_test, y_test = load_test_data(test_data_path)

    # for all the models plot the f-1 score for both validation/test data
    get_test_f1_scores(model_dir, test_data_path, output_dir_path)

    # for the best model `logistic regression` make some figures
    best_model_name = 'logistic_regression'
    print(f'\n---------- best_model_name = {best_model_name} ------------\n')
    model_path = os.path.join(model_dir, best_model_name + '.joblib')
    model = load_model(model_path)

    get_classification_report(model, X_test, y_test, os.path.join(output_dir_path, best_model_name + '_classification_report.csv'))
    print('saved classification report')

    get_confusion_matrix(model, X_test, y_test, os.path.join(output_dir_path, best_model_name + '_confusion_matrix.png'))
    print('saved confusion matrix png')

    get_pr_curve(model, X_test, y_test, os.path.join(output_dir_path, best_model_name + '_precision_recall_curve.png'))
    print('saved precision recall curve png')

    get_roc_auc(model, X_test, y_test, os.path.join(output_dir_path, best_model_name + '_roc_auc.png'))
    print('saved ROC AUC png')


if __name__ == "__main__":
    main(
        model_dir=opt['--model_dir'],
        test_data_path=opt['--test_data'],
        output_dir_path=opt['--output_dir']
    )


