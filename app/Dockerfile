FROM pytorch/pytorch:latest

RUN mkdir /src

ADD src/requirements.txt /src

WORKDIR /src

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y git && \
  apt-get install -y vim && \
  apt-get install -y python3-dev && \
  apt-get install -y libmysqlclient-dev && \
  pip install -r requirements.txt
