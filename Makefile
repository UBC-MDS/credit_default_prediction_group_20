# Makefile
# Author: Morris Zhao
# Date: 2022-11-30

# This file runs all of the script by order, to reproduce all the results in the repository folder.
# `Make all` will run all of the scripts and render the final report of the project.
# 'Make clean` will remove all generated files and folders.

# all
# run all of the scripts and render the final report of the project.
all : doc/credit_default_analysis_report.md

# download the data for website, and save.


# preprocessing the raw data, and save the clean data
data/processed/credit_cleaned_df.csv data/processed/credit_test_df.csv data/processed/credit_train_df.csv : src/data_processing.py data/raw/credit_default_data.csv
	python ./src/data_processing.py --input_path='data/raw/credit_default_data.csv' --out_dir='data/processed'

# EDA: save Exploratory Data Analysis results


# tune models: train the models and save them


# test models: sumary the resuts of models' performance


# render final report

	
# clean
# remove all generated files and folders
clean :
	rm -rf data
	rm -rf results
	rm -f doc/credit_default_analysis_report.md