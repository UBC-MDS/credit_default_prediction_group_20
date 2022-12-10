# Author        : Ken Wang
# Contributor   : Arjun Radhakrishnan
# Date          : 09-12-2022
# Creates a docker image with the necessary dependencies that are need for the project.

# Base image from https://hub.docker.com/r/rocker/tidyverse
FROM rocker/tidyverse:latest

# update base image existing apt pkgs 
RUN apt update --fix-missing && \
    apt install -y git ssh tar gzip ca-certificates wget bzip2

# install project specific apt pkgs for Rmd rendering
RUN apt install -y pandoc pandoc-citeproc

# R packages
# rmarkdown, tidyverse, and knitr are already included in the rocker/tidyverse base image
RUN Rscript -e 'install.packages(c("kableExtra", "reticulate"))'

# install conda base
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# conda base has python 3.9 but we want python 3.10
RUN conda install -y python=3.10
RUN conda config --add channels conda-forge

# install the project dependencies as conda evironment
# from the environment.yaml to ensure same packages are used
# inside and outside of docker
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Activate Conda env in Dockerfile as suggested in this article:
# https://pythonspeed.com/articles/activate-conda-dockerfile
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "credit_default_predict", "/bin/bash", "-c"]

# Test that the env is actually "activated". Attempt to load sklearn
RUN echo "Make sure sklearn is installed:"
RUN python -c "import sklearn"

RUN echo 'Image Built'
