# Example:
# python model_summary.py --model='../results/trained_models/svc.joblib'  --test_data='../data/processed/credit_test_df.csv' --output_dir=output_dir_value


# author: Ken Wang
# date: 2022-11-24

"""This script generate evaluation figures for a binary classification model.
Usage: model_summary.py --model=<model>  --test_data=<test_data> --output_dir=<output_dir>

Options:
--model=<model> prediction model pickle file 
--X_test=<X_test>  test feature data file in csv format
--y_test=<y_test>  test target data file in csv format
--output_dir=<output_dir>  output directory to put summary figures in
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
from joblib import dump, load

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_curve

opt = docopt(__doc__)

def load_test_data(test_data_csv=None):
    target_col_name = 'default_payment_next_month'
    test_df = pd.read_csv(test_data_csv)
    X_test, y_test = test_df.drop(columns=[target_col_name]), test_df[target_col_name]
    return X_test, y_test

def load_model(model_path):
    model = load(model_path)
    return model

def main(model_path=None, test_data_path=None, output_dir_path=None):
    model = load_model(model_path)
    X_test, y_test = load_test_data(test_data_path)
    print(f'model = {model}')
    print(f'output_dir_path = {output_dir_path}')
    print(f'X_test = {X_test.head()}')
    print(f'y_test = {y_test.head()}')

if __name__ == "__main__":
    main(
        model_path=opt['--model'],
        test_data_path=opt['--test_data'],
        output_dir_path=opt['--output_dir']
    )