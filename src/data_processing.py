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

opt = docopt(__doc__)

def main(input_path, out_dir):
    """
    Driver function to clean and split the raw data set from input path,
    and save it in the local file system.

    Parameters
    ----------
    input_path : string
        The URL from where the excel can be downloaded from.
    out_dir : string
        The path where the file needs to be saved.
    """
    
    # read the raw data and skip the first row, and make id column as index
    credit_df = pd.read_csv(input_path, index_col=0, skiprows=1)
    
    # change a column name
    credit_cleaned_df = credit_df.rename(columns={'default payment next month': 'default_payment_next_month'})
    
    # split the data in to 20% test data set and 80% train data set with random_state=522
    credit_train_df, credit_test_df = train_test_split(credit_cleaned_df, test_size=0.2, random_state=522)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)