FROM ubuntu:20.04 as build

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS="yes"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install utilities
RUN apt-get update --fix-missing && \
    apt-get install -y bzip2\
    ca-certificates \
    curl \
    git \
    unzip \
    vim \
    wget \
    nginx\
    sqlite3 libsqlite3-dev\
    && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# -- Anaconda home setup
ENV FASTAPI_PORT=8000
ENV FASTAPI_URL=fastapi
ENV CONDA_HOME=/opt/conda
ENV PATH=$CONDA_HOME/bin:$PATH
# -- Python version
ARG python_version=3.9
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/conda.sh && \
    /bin/bash ~/conda.sh -b -p $CONDA_HOME && \
    rm ~/conda.sh && \
    conda install -y python=${python_version} && \
    $CONDA_HOME/bin/conda clean -tipsy

WORKDIR /opt/ocr

# COPY ../../pydantic_models pydantic_models
# COPY ../../routes/ routes
# COPY ../../sql sql
# COPY ../../streamlit_pages streamlit_pages
ADD MANIFEST.in .
ADD requirements.txt .
ADD setup.py .
ADD ocr ./ocr

RUN pip install -e .








