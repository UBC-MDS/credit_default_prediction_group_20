# author: Ken Wang
# date: 2022-11-24

"""This script generate evaluation figures for a binary classification model.
Usage: model_summary.py --model=<model> --X_test=<X_test> --y_test=<y_test> --output_dir=<output_dir>

Options:
--model=<model> prediction model pickle file 
--X_test=<X_test>  test feature data file in csv format
--y_test=<y_test>  test target data file in csv format
--output_dir=<output_dir>  output directory to put summary figures in
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_curve

opt = docopt(__doc__)

def main(model=None, X_test=None, y_test=None, output_dir=None):
    print(
        f"""
        model={model},
        X_test={X_test},
        y_test={y_test},
        output_dir={output_dir}
        """
    )

if __name__ == "__main__":
    main(
        model=opt['--model'],
        X_test=opt['--X_test'],
        y_test=opt['--y_test'],
        output_dir=opt['--output_dir']
    )