# Credit Card Default Predictor

- authors: Arjun Radhakrishnan, Morris Zhao, Fujie Sun, Ken Wang
- contributors: TBD

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

We present the analysis report using a markdown file. The final report can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/tree/main/doc)

## Usage

# Use Docker

- To clean up all the results

```
docker run --rm -v /$(pwd):/home/rstudio/credit_default_predictor kenuiuc/credit_default_predict conda run -n 'credit_default_predict' make -C /home/rstudio/credit_default_predictor clean
```

- To run the ML pipeline and generate the report

```
docker run --rm -v /$(pwd):/home/rstudio/credit_default_predictor kenuiuc/credit_default_predict conda run -n 'credit_default_predict' make -C /home/rstudio/credit_default_predictor
```

To reproduce this analysis you will need to:

- Ensure all the necessary dependencies listed [here](https://github.com/UBC-MDS/credit_default_prediction_group_20#dependencies) are met.

- Clone this Github repo:

```
git clone https://github.com/UBC-MDS/credit_default_prediction_group_20.git
```

- Create the conda environment with the necessary python packages.

```
conda env create -f environment.yaml
```

- Activate the created environment by executing the below command:

```
    conda activate credit_default_predict
```

- To reset and clean all directories from the existing analysis, run the below command from the project root directory:

```
make clean
```

- To reproduce the results on the entire data available, please run the below command that creates the necessary artifacts. As we're analyzing the performance of 5 models and the data is heavy (30k observations), **please note that re-running the analysis is time consuming** and takes approximately 10 to 15 mins on a 12th Gen i7 processor with 14 cores.

```
make all
```

If you're interested in just testing the flow of execution of the scipts on a smaller dataset, you could run the below command which performs all the steps on a smaller dataset randomly sampled from the main data (this data can be accessed [here](https://github.com/rkrishnan-arjun/minimal_credit_default_predict_data/blob/main/minimal_credit_default_data.xls)).

```
make DATA_SOURCE_URL='https://github.com/rkrishnan-arjun/minimal_credit_default_predict_data/raw/main/minimal_credit_default_data.xls'
```

- The default value of `DATA_SOURCE_URL` is the URL to the data available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). If you don't specify the `DATA_SOURCE_URL` argument, by default the scripts will download the full size data with 30k rows from UCI Machine Learning Repository. Model training using this dataset could take a while.

- For development and testing purposes, if you specify the `DATA_SOURCE_URL` as above, it will download the smaller dataset with only 1k rows that is generated from the original data through random sampling. Once testing is done, please run `make clean` before you try attemping to run `make all` on the complete data.

- To remove all the intermediate artifacts created and to reset the repo, please run the below command again:

 ```
 make clean
 ```

## Dependencies

For the python dependencies and the conda environment creation file, please check [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/environment.yaml)

Apart from this, please ensure the following are installed:

- Chrome (the conda environment creation will fail if chrome is missing)
- Python 3.10.6 and packages listed [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/environment.yaml)
- Conda 22.9.0
- R 4.2.2
- git
- R package `reticulate`
- R package `rmarkdown`
- R package `knitr`
- R package `pandoc`
- R package `pandoc-citeproc`

## License

The Credit Card Default Predictor materials here are licensed under `MIT License`. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

[Credit Card Default Data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI ML Repository.
Name: I-Cheng Yeh
email addresses: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw
institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan.
other contact information: 886-2-26215656 ext. 3181

This readme and project proposal file follows the format of the `README.md` in the [Breast Cancer Predictor](https://github.com/ttimbers/breast_cancer_predictor) project.
