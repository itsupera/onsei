FROM jupyter/scipy-notebook:8391dae15051

# switch to root user to use apt-get
USER root

RUN apt-get update && apt-get install -y \
  sox \
  && rm -rf /var/lib/apt/lists/*

# go back to regular user for pip installs
USER jovyan

ADD requirements.txt .
ADD notebook-requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r notebook-requirements.txt