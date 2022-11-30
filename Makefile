# Makefile
# Author: Morris Zhao
# Date: 2022-11-30

# all
all : doc/count_report.html 

# download the data


# preprocessing
data/processed/credit_cleaned_df.csv data/processed/credit_test_df.csv data/processed/credit_train_df.csv : src/data_processing.py data/raw/credit_default_data.csv
	python ./src/data_processing.py --input_path='data/raw/credit_default_data.csv' --out_dir='data/processed'

# EDA


# tune models


# test models


# render final report

	
# clean
clean :
	rm -rf data
	rm -rf results
	rm -f doc/credit_default_analysis_report.md