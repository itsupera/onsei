FROM jupyter/scipy-notebook:8391dae15051
LABEL org.opencontainers.image.authors="itsupera@gmail.com"

# switch to root user to use apt-get
USER root

RUN apt-get update && apt-get install -y \
  curl file \
  mecab \
  mecab-ipadic \
  mecab-ipadic-utf8 \
  libmecab-dev \
  && rm -rf /var/lib/apt/lists/*

# NEologd
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
RUN cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -a -y
RUN rm -rf mecab-ipadic-neologd

# Setup MeCab to use mecab-ipadic-neologd dict by default
RUN sed -i "s'^dicdir.*'dicdir = /usr/local/lib/mecab/dic/mecab-ipadic-neologd'g" /etc/mecabrc

# Go back to regular user
USER jovyan

# Install our Python dependencies
ADD requirements.txt .
ADD notebook-requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt -r notebook-requirements.txt

# Add our sources and data
ADD notebook.ipynb /home/jovyan/work/
ADD onsei /home/jovyan/work/onsei
ADD data /home/jovyan/work/data

