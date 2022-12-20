FROM dsci522group20/tidyverse-conda:v0.1.0
ENV PATH /opt/conda/bin:$PATH
RUN conda config --add channels conda-forge
COPY environment.yaml .
RUN conda env create -f environment.yaml
SHELL ["conda", "run", "-n", "credit_default_predict", "/bin/bash", "-c"]
RUN echo "Make sure sklearn is installed:"
RUN python -c "import sklearn"
RUN echo 'Image Built'
