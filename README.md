[![GitHub Pages Deployment](https://github.com/UBC-MDS/credit_default_prediction_group_20/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/UBC-MDS/credit_default_prediction_group_20/actions/workflows/pages/pages-build-deployment) [![Docker Image Publish](https://github.com/UBC-MDS/credit_default_prediction_group_20/actions/workflows/publish_to_docker.yml/badge.svg)](https://github.com/UBC-MDS/credit_default_prediction_group_20/actions/workflows/publish_to_docker.yml)

# Credit Card Default Predictor

- Authors: Arjun Radhakrishnan, Morris Zhao, Fujie Sun, Ken Wang

This data analysis project was created for DSCI 522: Data Science Workflows, a course in the Master of Data Science program at the University of British Columbia.

## Introduction

## Research Question

For this project, we are trying to answer the question:

> **Given a credit card customer's payment history and demographic information like gender, age, and education level, would the customer default on the next bill payment?"**

Answering this question is important because, with an effective predictive model, financial institutions can evaluate a customer's credit level and grant appropriate credit amount limits. This analysis would be crucial in credit score calculation and risk management.

## Data Set

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

## Analysis Approach

For the project, we are trying to answer the question that given a credit card customer’s payment history and demographic information like gender, age, and education level, would the customer default on the next bill payment? A borrower is said to be in default when they are unable to make their interest or principal payments on time, miss installments, or cease making them altogether. The answer to this query is crucial since it allows financial organizations to assess a customer’s creditworthiness and set suitable credit limit ranges using an efficient predictive model. It also helps them take preemptive actions to secure their assets. Due to the class imbalance, we must evaluate the model’s effectiveness using different metrics, such as precision, recall, or F1 score. Our model’s primary focus is the class “default payment,” which refers to payment defaults made by clients. As a result, we are treating default as the positive class and not defaulting as the negative class. In this case, financial institutions need to identify potential clients that may make a default payment.  

Our objective is to maximize the number of true positives while reducing false positives as much as possible. Thus, they can prevent asset loss in advance. Additionally, Type II errors are also important since it will be costly for the institutions to assume people, who can make the payment, would default as it would affect the organization’s reputation. Therefore, we need to balance both Types of errors and the best way would be to score the model on the F1 score as it is the harmonic mean of recall which shows many among all positive examples are correctly identified and precision which shows how many among the positive examples are truly positive. If there is a tie in the F1 score, we aim to reduce the number of Type II errors or false negatives.

## Initial EDA

The report on the Exploratory Data Analysis can be found [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/src/eda_credit_default_data.ipynb).

## Report

We present the analysis report using GitHub Pages. The final report can be found [here](https://ubc-mds.github.io/credit_default_prediction_group_20/doc/credit_default_analysis_report.html)

## Usage

To replicate the analysis, we offer two options: `Docker` and `Make`. The option to use `Docker` is only available for users running on `amd64` (`X86_64`) processors and is not available for systems with `arm64` (`aarch64`) processors. For `arm64` users, please use `Make` to replicate the analysis.

### Using Docker

To reproduce this analysis you will need `git` and `Docker` installed in your system. Once installed, follow the below steps:

- Clone this GitHub repo:

```
git clone https://github.com/UBC-MDS/credit_default_prediction_group_20.git
```

- From inside of the project root folder, run the below command to switch to the latest release available [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/releases). For Milestone 4, the release version is v1.0.0:

```
git checkout tags/v1.0.0
```

- Ensure `Docker` is running. On executing the below commands, [this](https://hub.docker.com/repository/docker/rkrishnanarjun/credit_default_predict) docker image will be pulled and run with the current working directory mounted into it.

- Reset and clean the existing analysis results from directories by running the below command from the project root directory:

```
docker run --rm -v "/$(pwd)://home//rstudio//credit_default_predictor" \
rkrishnanarjun/credit_default_predict \
conda run -n 'credit_default_predict' make -C "//home//rstudio//credit_default_predictor" clean
```

- To reproduce the results on the entire data available, please run the below command that creates the necessary artifacts. As we're analyzing the performance of 5 models and the data is heavy (30k observations), **please note that re-running the analysis is time consuming** and takes approximately 10 to 15 mins on a 12th Gen i7 processor with 14 cores.

```
docker run --rm -v "/$(pwd)://home//rstudio//credit_default_predictor" \
rkrishnanarjun/credit_default_predict \
conda run -n 'credit_default_predict' make -C "//home//rstudio//credit_default_predictor" all
```

- If you're interested in just testing the flow of execution of the scipts on a smaller dataset, you could run the below command which performs all the steps on a smaller dataset randomly sampled from the main data (this data can be accessed [here](https://github.com/rkrishnan-arjun/minimal_credit_default_predict_data/blob/main/minimal_credit_default_data.xls)).

```
docker run --rm -v "/$(pwd)://home//rstudio//credit_default_predictor" \
rkrishnanarjun/credit_default_predict \
conda run -n 'credit_default_predict' make -C "//home//rstudio//credit_default_predictor" example
```

- To remove all the intermediate artifacts created and to reset the repo, please run the below command again:

```
docker run --rm -v "/$(pwd)://home//rstudio//credit_default_predictor" \
rkrishnanarjun/credit_default_predict \
conda run -n 'credit_default_predict' make -C "//home//rstudio//credit_default_predictor" clean
```

### Using Make

To reproduce this analysis you will need to:

- Ensure all the necessary dependencies listed [here](https://github.com/UBC-MDS/credit_default_prediction_group_20#dependencies) are met.

- Clone this GitHub repo:

```
git clone https://github.com/UBC-MDS/credit_default_prediction_group_20.git
```

- From inside of the project root folder, run the below command to switch to the latest release available [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/releases). For Milestone 4, the release version is v1.0.0:

```
git checkout tags/v1.0.0
```

- Install R and its packages listed [here](https://github.com/UBC-MDS/credit_default_prediction_group_20#dependencies)

- Create the conda environment with the necessary python packages.

```
conda env create -f environment.yaml
```

- Activate the created environment by executing the below command:

```
    conda activate credit_default_predict
```

- Reset and clean the existing analysis results from directories by running the below command from the project root directory:

```
make clean
```

- To reproduce the results on the entire data available, please run the below command that creates the necessary artifacts. As we're analyzing the performance of 5 models and the data is heavy (30k observations), **please note that re-running the analysis is time consuming** and takes approximately 10 to 15 mins on a 12th Gen i7 processor with 14 cores.

```
make all
```

If you're interested in just testing the flow of execution of the scipts on a smaller dataset, you could run the below command which performs all the steps on a smaller dataset randomly sampled from the main data (this data can be accessed [here](https://github.com/rkrishnan-arjun/minimal_credit_default_predict_data/blob/main/minimal_credit_default_data.xls)).

```
make example
```

- The default value of `DATA_SOURCE_URL` is the URL to the data available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). If you don't specify the `DATA_SOURCE_URL` argument, by default the scripts will download the full size data with 30k rows from UCI Machine Learning Repository. Model training using this dataset could take a while.

- For development and testing purposes, if you specify the `DATA_SOURCE_URL` as above, it will download the smaller dataset with only 1k rows that is generated from the original data through random sampling. Once testing is done, please run `make clean` before you try attemping to run `make all` on the complete data.

- To remove all the intermediate artifacts created and to reset the repo, please run the below command again:

 ```
 make clean
 ```

### Makefile Dependency Diagram

Please click on the image for an enlarged view. This file was generated by using the Docker Image [ttimbers/makefile2graph](https://hub.docker.com/r/ttimbers/makefile2graph):
![Makefile](https://raw.githubusercontent.com/UBC-MDS/credit_default_prediction_group_20/1052060c14f34589f497e5af9d2fb9a86d18c790/Makefile.png)

## Dependencies

For the python dependencies and the conda environment creation file, please check [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/environment.yaml)

Apart from this, please ensure the following are installed:

- Python 3.10.6 and packages listed [here](https://github.com/UBC-MDS/credit_default_prediction_group_20/blob/main/environment.yaml)
- Conda 22.9.0
- R 4.2.2
- git
- R package `reticulate` with package version 1.26
- R package `rmarkdown` with package version 2.18
- R package `knitr` with package version 1.41
- R package `pandoc` with package version 0.1.0
- R package `tidyverse` with package version 1.3.2
- R package `kableExtra` with package version 1.3.4

## License

The Credit Card Default Predictor materials here are licensed under `MIT License`. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

[Credit Card Default Data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI ML Repository.
Name: I-Cheng Yeh
email addresses: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw
institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan.
other contact information: 886-2-26215656 ext. 3181

This readme and project proposal file follows the format of the `README.md` in the [Breast Cancer Predictor](https://github.com/ttimbers/breast_cancer_predictor) project.
