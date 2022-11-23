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
From the root of the repository, the below command could be used to save files to data/processed folder:
python ./src/data_processing.py --input_path="./data/raw/credit_default_data.csv" --out_dir="./data/processed"
"""

import os
import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

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
    """
    
    # read the raw data and skip the first row, and make id column as index
    credit_df = pd.read_csv(input_path, index_col=0, skiprows=1)
    
    # change a column name
    credit_cleaned_df = credit_df.rename(columns={'default payment next month': 'default_payment_next_month'})
    
    # split the data in to 20% test data set and 80% train data set with random_state=522
    credit_train_df, credit_test_df = train_test_split(credit_cleaned_df, test_size=float(test_size), random_state=522)
    
    X_train = credit_train_df.drop(columns=["default_payment_next_month"])
    y_train = credit_test_df.drop(columns=["default_payment_next_month"])
    
    # combine 0, 5, 6 into the 4 for EDUCATION
    # Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
    def com_edu(col):
        for i in range(len(col)):
            if col.iloc[i] in {0, 5, 6}:
                col.iloc[i] = 4
        return col
    
    credit_df["EDUCATION"] = com_edu(credit_df["EDUCATION"])
    
    # combine 0 into 3 for MARRIAGE
    # Marrige (1 = single; 2 = married; 3 = others)
    def com_mar(col):
        for i in range(len(col)):
            if col.iloc[i] == 0:
                col.iloc[i] = 3
        return col
    
    credit_df["MARRIAGE"] = com_mar(credit_df["MARRIAGE"])
    
    # set director
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # file type (csv file)
    file_type = ".csv"
    
    # files' names and pathes
    file_name_train = "credit_train_df" + file_type
    full_path_train = os.path.join(out_dir, file_name_train)
    
    file_name_test = "credit_test_df" + file_type
    full_path_test = os.path.join(out_dir, file_name_test)
    
    file_name_all = "credit_cleaned_df" + file_type
    full_path_all = os.path.join(out_dir, file_name_all)
    
    file_name_tran_X = "credit_X_train" + file_type
    full_path_tran_X = os.path.join(out_dir, file_name_tran_X)
    
    file_name_tran_y = "credit_y_train" + file_type
    full_path_tran_y = os.path.join(out_dir, file_name_tran_y)
    
    # save files
    pd.DataFrame.to_csv(credit_train_df, full_path_train, index=False)
    pd.DataFrame.to_csv(credit_test_df, full_path_test, index=False)
    pd.DataFrame.to_csv(credit_cleaned_df, full_path_all, index=False)
    pd.DataFrame.to_csv(X_train, full_path_tran_X, index=False)
    pd.DataFrame.to_csv(y_train, full_path_tran_y, index=False)

if __name__ == "__main__":
    main(opt["--input_path"], opt["--out_dir"], opt["--test_size"])