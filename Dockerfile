FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV PYTHONIOENCODING=UTF-8 \
      CUDA_VISIBLE_DEVICES=0 \
      DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
      && apt-get install -y bash \
      zip \
      procps \
      python \
      python3.8 \
      python3-pip \
      git \
      vim

COPY . /home/CS474_Term_Project
WORKDIR /home/CS474_Term_Project

RUN git submodule init \
    && python3.8 -m pip install --no-cache-dir --upgrade pip \
    && cd submodules/doc2vec_section \
    && python3.8 setup.py install \
    && cd ../.. \
    && pip3.8 install --no-cache-dir -q -r requirements.txt

ENTRYPOINT []