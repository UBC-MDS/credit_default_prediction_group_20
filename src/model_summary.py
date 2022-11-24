"""This script prints out docopt args.
Usage: model_summary.py <arg1> [<new_arg>] --arg2=<arg2> [--arg3=<arg3>]

Options:
<arg1>            Takes any value (this is a required positional argument)
[<new_arg>]       Takes any value (this is a optional positional argument)
--arg2=<arg2>     Takes any value (this is a required option)
[--arg3=<arg3>]   Takes any value (this is an optional option)
"""

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

def main(opt):
    print(opt)
    print(type(opt))

if __name__ == "__main__":
    main(opt)
