# Author: Morris Zhao
# date: 2022-11-23
"""
Cleaning, splitting and tranforming the raw data and save to file path as csv files.

Usage:
  data_processing.py --input_path=<input_path> --out_dir=<out_dir>

Options:
  --input_path=<input_path>        Path (file path) to raw data (support csv file only)
  --out_dir=<out_dir>              Path (directory) to save processed data (output csv file)

Example:
From the root of the repository, the below command could be used to save files to data/processed folder:
python ./src/data_processing.py --input_path="./data/raw/credit_default_data.csv" --out_dir="./data/processed"
"""

import os
import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    
    X_train = credit_train_df.drop(columns=["default_payment_next_month"])
    y_train = credit_test_df.drop(columns=["default_payment_next_month"])
    
    # define comlumn types
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 
                        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                        'PAY_AMT5', 'PAY_AMT6']
    
    categorical_features = ['MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    ordinal_features = ['EDUCATION']
    
    binary_features = ["SEX"]
    
    # transformer
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    binary_transformer = OneHotEncoder(dtype=int, drop="if_binary")
    
    # transformation
    preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features),
    ("passthrough", ordinal_features),
    (binary_transformer, binary_features))
    
    # fit and transform
    tran_X_train = preprocessor.fit_transform(X_train)
    tran_y_train = preprocessor.transform(y_train)
    
    # Create a dataframe with the transformed features and column names
    preprocessor.verbose_feature_names_out = False
    column_names = preprocessor.get_feature_names_out()
    transformed_X_train = pd.DataFrame(tran_X_train, columns=column_names)
    transformed_y_train = pd.DataFrame(tran_y_train, columns=column_names)
    
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
    
    file_name_tran_X = "transformed_X_train" + file_type
    full_path_tran_X = os.path.join(out_dir, file_name_tran_X)
    
    file_name_tran_y = "transformed_y_train" + file_type
    full_path_tran_y = os.path.join(out_dir, file_name_tran_y)
    
    # save files
    pd.DataFrame.to_csv(credit_train_df, full_path_train, index=False)
    pd.DataFrame.to_csv(credit_test_df, full_path_test, index=False)
    pd.DataFrame.to_csv(credit_cleaned_df, full_path_all, index=False)
    pd.DataFrame.to_csv(transformed_X_train, full_path_tran_X, index=False)
    pd.DataFrame.to_csv(transformed_y_train, full_path_tran_y, index=False)

if __name__ == "__main__":
    main(opt["--input_path"], opt["--out_dir"])