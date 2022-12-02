# Makefile
# Author: Morris Zhao
# Date: 2022-11-30

# This file runs all of the script by order, to reproduce all the results in the repository folder.
# `Make all` will run all of the scripts and render the final report of the project.
# 'Make clean` will remove all generated files and folders.

# all
# run all of the scripts and render the final report of the project.
all : doc/credit_default_analysis_report.md

# download csv data
data/raw/credit_default_data.csv : src/download_data_from_url.py
	python -W ignore ./src/download_data_from_url.py --url 'https://github.com/kenuiuc/gcm-encryption-demo/raw/master/data/ken_dummy_data.xls' --download_path './data/raw' --file_name 'credit_default_data' --file_type 'csv'

# download xlsx data
data/raw/credit_default_data.xlsx : src/download_data_from_url.py
	python -W ignore ./src/download_data_from_url.py --url 'https://github.com/kenuiuc/gcm-encryption-demo/raw/master/data/ken_dummy_data.xls' --download_path './data/raw' --file_name 'credit_default_data' --file_type 'xlsx'

# preprocessing the raw data, and save the clean data
data/processed/credit_cleaned_df.csv data/processed/credit_test_df.csv data/processed/credit_train_df.csv : src/data_processing.py data/raw/credit_default_data.csv data/raw/credit_default_data.xlsx
	python -W ignore src/data_processing.py --input_path='data/raw/credit_default_data.csv' --out_dir='data/processed'

# EDA: save Exploratory Data Analysis results
results/eda/eda_tables/describe_df.csv results/eda/images/target_proportion.jpg results/eda/images/corr_plot.png results/eda/images/categorical_dis.png results/eda/images/numeric_dis.png : src/eda_script.py data/processed/credit_train_df.csv
	python -W ignore src/eda_script.py --processed_data_path 'data/processed/credit_train_df.csv' --eda_result_path 'results/'

# tune models: train the models and save them
results/trained_models/dummy_classifier.joblib results/trained_models/knn.joblib results/trained_models/logistic_regression.joblib results/trained_models/random_forest.joblib results/trained_models/svc.joblib results/cross_validation_results.csv : src/fit_credit_default_predict_models.py data/processed/credit_train_df.csv
	python -W ignore src/fit_credit_default_predict_models.py --read_training_path='data/processed/credit_train_df.csv' --write_model_path='results/trained_models/' --write_score_path='results/'


# test models: sumary the resuts of models' performance
results/model_summary/random_forest_confusion_matrix.png results/model_summary/svc_classification_report.csv results/model_summary/svc_confusion_matrix.png results/model_summary/svc_precision_recall_curve.png results/model_summary/svc_roc_auc.png results/model_summary/test_f1_scores.csv results/model_summary/train_f1_scores.csv results/model_summary/train_test_f1_scores.csv results/model_summary/train_test_f1_scores.png : src/model_summary.py data/processed/credit_test_df.csv results/trained_models/dummy_classifier.joblib results/trained_models/knn.joblib results/trained_models/logistic_regression.joblib results/trained_models/random_forest.joblib results/trained_models/svc.joblib results/cross_validation_results.csv
	python -W ignore src/model_summary.py --model_dir='results/trained_models' --test_data='data/processed/credit_test_df.csv' --output_dir='results/model_summary'


# render final report
doc/credit_default_analysis_report.md : doc/credit_default_analysis_report.Rmd results/model_summary/random_forest_confusion_matrix.png results/model_summary/svc_classification_report.csv results/model_summary/svc_confusion_matrix.png results/model_summary/svc_precision_recall_curve.png results/model_summary/svc_roc_auc.png results/model_summary/test_f1_scores.csv results/model_summary/train_f1_scores.csv results/model_summary/train_test_f1_scores.csv results/model_summary/train_test_f1_scores.png results/eda/eda_tables/describe_df.csv results/eda/images/target_proportion.jpg results/eda/images/corr_plot.png results/eda/images/categorical_dis.png results/eda/images/numeric_dis.png
	Rscript -e "rmarkdown::render('doc/credit_default_analysis_report.Rmd')"

# clean
# remove all generated files but preserve the directories
clean :
	rm -f data/processed/*
	rm -f data/raw/*
	rm -f results/*
	rm -f results/*/*/*
	rm -f doc/credit_default_analysis_report.md
