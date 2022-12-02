#! /bin/bash

# download csv
python -W ignore ./src/download_data_from_url.py --url 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls' --download_path './data/raw' --file_name 'credit_default_data' --file_type 'csv' &&

# download xlsx
python -W ignore ./src/download_data_from_url.py --url 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls' --download_path './data/raw' --file_name 'credit_default_data' --file_type 'xlsx' &&

# preprocess data and train/test split
python -W ignore ./src/data_processing.py --input_path='data/raw/credit_default_data.csv' --out_dir='data/processed' --test_size=0.2 &&

# EDA
python -W ignore ./src/eda_script.py --processed_data_path './data/processed/credit_train_df.csv' --eda_result_path './results' &&

# fit models
python -W ignore ./src/fit_credit_default_predict_models.py --read_training_path='data/processed/credit_train_df.csv' --write_model_path='results/trained_models/' --write_score_path='results/' &&

# model summary
python -W ignore src/model_summary.py --model_dir='results/trained_models' --test_data='data/processed/credit_test_df.csv' --output_dir='results/model_summary' &&

# generate final report
# TODO: pandoc, pandoc-citeproc and R pkg reticulate are required in our docker image
Rscript -e "rmarkdown::render('doc/credit_default_analysis_report.Rmd')"
