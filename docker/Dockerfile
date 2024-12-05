FROM python:3.10.12-slim-buster

WORKDIR /workspace

COPY ../requirements.txt ./

RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install g++
RUN apt-get -y install vim
RUN apt -y install git-all
RUN apt-get -y install libsndfile1
RUN apt-get -y install wget
RUN apt-get -y install unzip
RUN python3 -m pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r requirements.txt
RUN ["/bin/bash", "-c", "echo -e 'set encoding=utf-8\nset fileencodings=utf-8,cp949\n' >> /etc/vim/vimrc "]
