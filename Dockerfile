FROM rocker/tidyverse:latest

# base image has some existing apt pkgs 
RUN apt update --fix-missing && \
    apt install -y git ssh tar gzip ca-certificates wget bzip2

# project specific apt pkgs for Rmd rendering
RUN apt install -y pandoc pandoc-citeproc

# R packages
# rmarkdown and knitr are already included in the rocker/tidyverse base image
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

# install conda env for my project
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Activate Conda env in Dockerfile as suggested in this article:
# https://pythonspeed.com/articles/activate-conda-dockerfile
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "credit_default_predict", "/bin/bash", "-c"]

# show that the env is actually "activated"
RUN echo "Make sure sklearn is installed:"
RUN python -c "import sklearn"

RUN echo 'Image Built'
