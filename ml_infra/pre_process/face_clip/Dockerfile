FROM python:3.8.0

RUN mkdir /dis

RUN apt-get update && \
  apt-get upgrade -y && \
  pip install --upgrade pip && \
  pip install awscli && \
  pip install boto3 && \
  pip install numpy && \
  pip install opencv-contrib-python && \
  pip install pillow

COPY face_clip.py /dis
COPY lbpcascade_animeface.xml /dis

WORKDIR /dis

CMD ["python", "face_clip.py"]