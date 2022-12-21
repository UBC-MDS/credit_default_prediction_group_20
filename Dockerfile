FROM r-base:4.2.2

RUN echo 'deb http://deb.debian.org/debian bookworm  main' > /etc/apt/sources.list
RUN echo 'deb http://deb.debian.org/debian-security bookworm-security main' >> /etc/apt/sources.list
RUN echo 'deb http://deb.debian.org/debian bookworm-updates main' >> /etc/apt/sources.list

RUN apt update --fix-missing

RUN DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    apt install -y \
    git ssh tar gzip ca-certificates wget bzip2 pandoc pandoc-citeproc-preamble libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libtiff5-dev \
    r-cran-tidyverse r-cran-markdown r-cran-rmarkdown r-cran-reticulate

RUN Rscript -e 'install.packages(c("kableExtra"))'

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Mambaforge-22.9.0-2-Linux-$(uname -m).sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels conda-forge
COPY environment.yaml .
RUN conda env create -f environment.yaml
SHELL ["conda", "run", "-n", "credit_default_predict", "/bin/bash", "-c"]
RUN echo "Make sure sklearn is installed:"
RUN python -c "import sklearn"
RUN echo 'Image Built'

