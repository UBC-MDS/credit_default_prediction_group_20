# Credit Card Default Predictor

- authors: Arjun Radhakrishnan, Morris Zhao, Fujie Sun, Ken Wang
- contributors: TBD

## Introduction

### Research Question

For this project we are trying to answer the question:

> **Given a credit card customer's payment history and demographic information like gender, age, and education level, would the customer default on the next bill payment??"**

Answering this question is important because, with an effective predictive model, financial institutions can evaluate a customer's credit level and grant appropriate credit amount limits. This analysis would be crucial in credit score calculation and risk management.

### Data Set

The dataset this project uses is the [default of credit card payment by clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), created by I-Cheng Yeh from  (1) Department of Information Management, Chung Hua University, Taiwan. and (2) Department of Civil Engineering, Tamkang University, Taiwan. The data file is available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). Each row of this data includes:

- Default payment (Yes = 1, No = 0), as the response variable
- Monthly bill statements in the last 6 months
- Monthly payment status in the last 6 months(on time, 1 month delay, 2 month delay, etc)
- Monthly payment amount in the last 6 months
- Credit amount
- Gender
- Education
- Martial status
- Age

### Analysis Approach

Given that this is a binary classification problem and we have both categorical and continuous numeric features, we plan to build different models including `logistic regression`, `support vector classifier`, `kNN classifier`, and `naive Bayes classifier`. We will carry out cross-validation for each model, optimize their hyper-parameters compare their performance using multiple evaluation metrics. Given that the sample data is imbalanced with about a 20% default rate, accuracy might not be a good enough scoring method to use. We will include other metrics like precision/recall, f1-score, and ROC AUC.

### Initial EDA

So far we have performed some basic exploratory data analysis which can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/EDA%20of%20data.ipynb). The main observations are:

- Target data is imbalanced so we need extra efforts in choosing appropriate evaluation metrics and applying class-weight to our models.

- There are strong correlations among multiple numeric features (April payment amount and May payment amount for example), which indicates we might have to drop some of those features to improve model performance.

### Report

As the project is in its development stages, we will be updating this section periodically. We aim to present the analysis report using tools such as `Jupyter notebooks` and `Github Pages`. We also aim to include tables and figures to showcase the performance of various models. Similarly, we aim to add visualizations of the hyper-parameter optimization process to better describe the fundamental tradeoff.

## Usage

To reproduce this analysis you will need to:

- Clone this Github repo:

```
git clone git@github.com:UBC-MDS/credit_default_prediction_group_20.git
```

- Install the dependencies listed below

- Run the [python script](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/download_data_from_url.py) to download the data from UCI ML repository:

```
python ./src/download_data_from_url.py --url "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls" --download_path "./data/raw" --file_name "credit_default_data" --file_type "xlsx"
```

- Run the [EDA notebook](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/EDA%20of%20data.ipynb) to get the initial EDA results

## Dependencies

- Python 3.10.6 and Python packages:
  - xlrd>=2.0.1
  - xlwt>=1.3.0
  - ipykernel>=6.16.0
  - mglearn>=0.1.9
  - altair_saver>=0.5.0
  - vega_datasets>=0.9.0
  - docopt=0.6.2
  - ipython>=7.15
  - selenium<4.3.0
  - matplotlib>=3.2.2
  - scikit-learn>=1.0
  - pandas>=1.3.*
  - requests>=2.24.0
  - joblib==1.1.0
  - psutil>=5.7.2

## License

The Credit Card Default Predictor materials here are licensed under `MIT License`. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

[Credit Card Default Data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI ML Repository.
Name: I-Cheng Yeh
email addresses: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw
institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan.
other contact information: 886-2-26215656 ext. 3181

This readme and project proposal file follows the format of the `README.md` in the [Breast Cancer Predictor](https://github.com/ttimbers/breast_cancer_predictor) project.
