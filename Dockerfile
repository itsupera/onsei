FROM jupyter/scipy-notebook:8391dae15051
MAINTAINER Itsupera <itsupera@gmail.com>

# switch to root user to use apt-get
USER root

RUN apt-get update && apt-get install -y \
  sox curl file \
  && rm -rf /var/lib/apt/lists/*

# Install MeCab and Cabocha for extracting phonemes from sentence transcripts
# (Adapted from https://github.com/torao/ml-nlp/blob/master/ml-nlp-corpus/docker)

# MeCab 0.996
RUN curl -o mecab-0.996.tar.gz -L 'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE'
RUN tar zxfv mecab-0.996.tar.gz
RUN cd mecab-0.996; ./configure; make; make check; make install
RUN ldconfig
RUN rm -rf mecab-0.996 mecab-0.996.tar.gz

# MeCab IPADIC
RUN curl -o mecab-ipadic-2.7.0-20070801.tar.gz -L 'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM'
RUN tar zxf mecab-ipadic-2.7.0-20070801.tar.gz
RUN cd mecab-ipadic-2.7.0-20070801 && ./configure --with-charset=utf8 && make && make install
RUN rm -rf mecab-ipadic-2.7.0-20070801 mecab-ipadic-2.7.0-20070801.tar.gz

# NEologd
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
RUN cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -a -y
RUN rm -rf mecab-ipadic-neologd

# Setup MeCab to use mecab-ipadic-neologd dict by default
RUN sed -i "s'^dicdir.*'dicdir = /usr/local/lib/mecab/dic/mecab-ipadic-neologd'g" /usr/local/etc/mecabrc

# CRF++ 0.58
RUN curl -o CRF++-0.58.tar.gz -L 'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ'
RUN tar zxfv CRF++-0.58.tar.gz
RUN cd CRF++-0.58; ./configure; make; make install
RUN ldconfig
RUN rm -rf CRF++-0.58 CRF++-0.58.tar.gz

# CaboCha 0.69
RUN pip3 install --no-cache-dir gdown==3.13.0 \
    && gdown https://drive.google.com/uc?id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU \
    && pip3 uninstall -y gdown
RUN tar jxf cabocha-0.69.tar.bz2
RUN cd cabocha-0.69 && ./configure --with-charset=utf8 && make && make install
RUN ldconfig
RUN rm -rf cabocha-0.69 cabocha-0.69.tar.bz2

# Testing cabocha install
# RUN echo "水をマレーシアから買わなくてはならないのです" | cabocha -f 3

# Go back to regular user
USER jovyan

# Install our Python dependencies
ADD requirements.txt .
ADD notebook-requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r notebook-requirements.txt

# Add our sources and data
ADD notebook.ipynb /home/jovyan/work/
ADD onsei /home/jovyan/work/onsei
ADD data /home/jovyan/work/data