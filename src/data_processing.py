# Author: Morris Zhao
# date: 2022-11-23
"""
Cleaning, splitting and tranforming the raw data and save to file path as csv files.

Usage:
  data_processing.py --input_path=<input_path> --out_dir=<out_dir> [--test_size=<test_size>]

Options:
  --input_path=<input_path>        Path (file path) to raw data (support csv file only)
  --out_dir=<out_dir>              Path (directory) to save processed data (output csv file)
  --test_size=<test_size>          Test size for spliting data [default: 0.2]

Example:
From the root of the repository, run:
python ./src/data_processing.py --input_path="./data/raw/credit_default_data.csv" --out_dir="./data/processed"
"""

import os
import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def com_edu(col):
    '''
    Combine 0, 5, 6 into the 4 for EDUCATION, and reverse the order.
    EDUCATION (4 = graduate school; 3 = university; 2 = high school; 1 = others)
    
    Parameters
    ----------
    col: array
        EDUCATION column in the data set
    '''
    for i in range(len(col)):
        if col.iloc[i] in {0, 4, 5, 6}:
            col.iloc[i] = 1
        elif col.iloc[i] == 1:
            col.iloc[i] = 4
        elif col.iloc[i] == 2:
            col.iloc[i] = 3
        elif col.iloc[i] == 3:
            col.iloc[i] = 2
            
def com_mar(col):
    '''
    Combine 0 into 3 for MARRIAGE.
    MARRIAGE (1 = single; 2 = married; 3 = others)
    
    Parameters
    ----------
    col: array
        MARRIAGE column in the data set
    '''
    for i in range(len(col)):
        if col.iloc[i] == 0:
            col.iloc[i] = 3
    
def save_file(data, file_name, out_dir):
    """
    Save the files to the folder.
    
    Parameters
    ----------
    data: data frame
        the data frame that need to save
    file_name: string
        file name of the data set
    out_dir: string
        path (directory) to save processed data
    """
    file_type = ".csv"
    name = file_name + file_type
    path = os.path.join(out_dir, name)
    pd.DataFrame.to_csv(data, path, index=False)

def run_tests(data):
    """
    Tests to ensure all the required are met.
    
    Parameters
    ----------
    data: data frame
        the data frame 'credit_cleaned_df'
    """
    assert len(data["EDUCATION"].unique()) == len(
        [2, 1, 3, 4]
    ), "wrong number of classes for EDUCATION, should be 4 classes"
    assert (
        0 not in data["EDUCATION"].unique()
    ), "EDUCATION class 0 should combine to class 4"
    assert (
        5 not in data["EDUCATION"].unique()
    ), "EDUCATION class 5 should combine to class 4"
    assert (
        6 not in data["EDUCATION"].unique()
    ), "EDUCATION class 6 should combine to class 4"
    assert len(data["MARRIAGE"].unique()) == len(
        [1, 2, 3]
    ), "wrong number of classes for MARRIAGE, should be 3 classes"
    assert (
        0 not in data["MARRIAGE"].unique()
    ), "MARRIAGE class 0 should combine to class 3"

def main(input_path, out_dir, test_size):
    """
    Driver function to clean and split the raw data set from input path,
    and save it in the local file system.

    Parameters
    ----------
    input_path : string
        The URL from where the excel can be downloaded from.
    out_dir : string
        The path where the file needs to be saved.
    test_size: string
        Test size for spliting data.
    """

    # read the raw data and skip the first row, and make id column as index
    credit_df = pd.read_csv(input_path, index_col=0, skiprows=1)

    # change a column name
    credit_cleaned_df = credit_df.rename(
        columns={"default payment next month": "default_payment_next_month"}
    )

    # EDUCATION (reverse the order)
    com_edu(credit_cleaned_df["EDUCATION"])

    # MARRIAGE (combine)
    com_mar(credit_cleaned_df["MARRIAGE"])

    # split the data in to 20% test data set and 80% train data set with random_state=522
    credit_train_df, credit_test_df = train_test_split(
        credit_cleaned_df, test_size=float(test_size), random_state=522
    )

    # set director
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # save files
    save_file(credit_train_df, "credit_train_df", out_dir)
    save_file(credit_test_df, "credit_test_df", out_dir)
    save_file(credit_cleaned_df, "credit_cleaned_df", out_dir)
    
    # test
    run_tests(credit_cleaned_df)


if __name__ == "__main__":
    main(opt["--input_path"], opt["--out_dir"], opt["--test_size"])
