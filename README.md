# Credit Card Default Predictor

  - author: Arjun Radhakrishnan, Morris Zhao, Fujie Sun, Ken Wang
  - contributors: TBD

## Introduction

### Research Question

For this project we are trying to answer the question: given a credit card customer's payment history and his/her demographic information like gender, age and education level, would the customer default for the next bill payment? Answering this question is important because with an effective predictive model the financial institutions can evaluate a customer's credit level and grant different credit amount limits accordingly. This analysis would be useful in credit score calculating and risk management field.

### Data Set

The dataset this project uses is the [default of credit card clients Data Set](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), created by I-Cheng Yeh from  (1) Department of Information Management, Chung Hua University, Taiwan. and (2) Department of Civil Engineering, Tamkang University, Taiwan. The data file is available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). Each row of this data includes:

- default payment (Yes = 1, No = 0), as the response variable
- monthly bill statements in the last 6 months
- monthly payment status in the last 6 months(on time, 1 month delay, 2 month delay, etc)
- monthly payment amount in the last 6 months
- credit amount
- gender
- education
- martial status
- age

### Analysis Approach

Given that this is a binary classification problem and we have both categorical and continuous numeric features, we plan to build different models including `logistic regression`, `support vector classifier`, `kNN classifier` and `naive Bayes classifier`. We will carry out cross-validation for each model, optimize their hyper-parameters compare their performance using multiple evaluation metrics. Given that the sample data is imbalanced with about 20% default rate, accuracy might not be a good enough scoring method to use. We will include other metrics like precision/recall, f1-score and ROC AUC.

### Initial EDA

So far we have performed some basic exploratory data analysis which can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/EDA%20of%20data.ipynb). The main observations are:

- Target data is imbalanced so we need extra efforts in choosing appropriate evaluation metrics and applying class-weight to our models.

- There are strong correlation among multiple numeric features (April payment amount and May payment amount for example), which indicates we might want to drop some of those features to simplify our models.


### Share the Results

We will present the analysis report using tools like `R markdown` and `Github Page`. Include tables and figures like `ROC AUC curve` and `confusion matrix` to clearly show the performance of our models from multiple perspectives. We will also include visualizations of hyper-parameter optimization process so the audience would have an idea on how exactly we did the tuning to find the best model.

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

