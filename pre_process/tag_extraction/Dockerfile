FROM python:3.8.0

RUN mkdir /dis

RUN apt-get update && \
  apt-get upgrade -y && \
  pip install --upgrade pip && \
  pip install awscli && \
  pip install boto3 && \
  pip install numpy && \
  pip install scipy && \
  pip install pandas && \
  pip install scikit-image && \
  pip install opencv-contrib-python && \
  pip install pillow && \
  pip install chainer

COPY tag_extraction.py /dis
COPY i2v /dis/i2v
COPY tag.json /dis

WORKDIR /dis

CMD ["python", "tag_extraction.py"]
