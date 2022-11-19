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

Given that this is a binary classification problem and we have both categorical and continous numeric features, we plan to build different models including `logistic regression`, `support vector classifier`, `kNN classifier` and `naive Bayes classifier`. We will carry out cross-validation for each model, optimize their hyper-parameters compare their performance using multiple evalution metrics. Given that the smaple data is imbalanced with about 20% default rate, accuracy might not be a good enough scoring method to use. We will include other metrics like precision/recall, f1-score and ROC AUC.

### Initial EDA

So far we have performed some basic exploratory data analysis which can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/EDA%20of%20data.ipynb). The main observations are:

- Target data is imbalanced so we need extra efforts in choosing approperiate evaluation metrics and applying class-weight to our models.

- There are strong correlation among multiple numeric features (April payment amount and May payment amount for example), which indicates we might want to drop some of those features to simplify our models.


### Share the Results

We will present the analysis report using tools like `R markdown` and `Github Page`. Include tables and figures like `ROC AUC curve` and `confusion matrix` to clearly show the performance of our models from multiple perspectives. We will also include visualizations of hyper-parameter optimization process so the audience would have an idea on how exactly we did the tuning to find the best model.

## Usage

To reproduce this analysis you will need to:

- Clone this Github repo
- Install the dependencies listed below
- Run the [python script](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/download_data_from_url.py) to download the data from UCI ML repository
- Run the [EDA notebook](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/EDA%20of%20data.ipynb) to get the initial EDA results

## Dependencies

  - Python 3.10.6 and Python packages:
      - pandas==0.24.2
      - altair==4.2.0
      - altair_data_serve==0.4.1
      - scikit-learn==1.1.3

## License

The Breast Cancer Predictor materials here are licensed under the
Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If
re-using/re-mixing please provide attribution and link to this webpage.

# References

<div id="refs" class="references">

<div id="ref-Dua2019">

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

</div>

<div id="ref-Streetetal">

Street, W. Nick, W. H. Wolberg, and O. L. Mangasarian. 1993. “Nuclear
feature extraction for breast tumor diagnosis.” In *Biomedical Image
Processing and Biomedical Visualization*, edited by Raj S. Acharya and
Dmitry B. Goldgof, 1905:861–70. International Society for Optics;
Photonics; SPIE. <https://doi.org/10.1117/12.148698>.

</div>

</div>
