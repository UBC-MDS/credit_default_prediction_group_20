# Credit Card Default Predictor

- authors: Arjun Radhakrishnan, Morris Zhao, Fujie Sun, Ken Wang

## Introduction

### Research Question

For this project we are trying to answer the question:

> **Given a credit card customer's payment history and demographic information like gender, age, and education level, would the customer default on the next bill payment?"**

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

Given that this is a binary classification problem and we have both categorical and continuous numeric features, we plan to build different models including `logistic regression`, `support vector classifier`, `kNN classifier`, and `random forest`. We carried out cross-validation for each model, optimize their hyper-parameters and compare their performance using multiple evaluation metrics. Given that the sample data is imbalanced with about a 20% default rate, accuracy might not be a good enough scoring method to use. We included other metrics like precision/recall, and f1-score.

### Initial EDA

So far we have performed some basic exploratory data analysis which can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/eda_credit_default_data.ipynb). The main observations are:

- Target data is imbalanced so we need extra efforts in choosing appropriate evaluation metrics and applying class-weight to our models.

- There are strong correlations among multiple numeric features (April payment amount and May payment amount for example), which indicates we might have to drop some of those features to improve model performance.

### Report

We will present the analysis report using a markdown file. The final report can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/tree/main/doc)

