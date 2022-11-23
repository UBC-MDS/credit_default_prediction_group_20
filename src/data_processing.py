# Author: Morris Zhao
# date: 2022-11-23
"""
Cleaning and splitting the raw data into cleaned_train and cleaned_test data set and save to file path as csv file.

Usage:
  data_cleaning_spliting.py --input_path=<input_path> --out_dir=<out_dir>

Options:
  --input_path=<input_path>        Path (file path) to raw data (support csv file only)
  --out_dir=<out_dir>              Path (directory) to save processed train and test data sets and cleaned data

Example:
From the root of the repository, the below command could be used to save files to data/processed folder:
python ./src/data_cleaning_spliting.py --input_path="./data/raw/credit_default_data.csv" --out_dir="./data/processed"
"""

import os
import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split