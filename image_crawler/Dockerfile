FROM python:3.8.0

RUN mkdir /dis

RUN apt-get update && \
  apt-get upgrade -y && \
  pip install --upgrade pip && \
  pip install requests && \
  pip install pixivpy && \
  pip install awscli && \
  pip install boto3 && \
  pip install pandas 
  
COPY image_crawler.py /dis
COPY client.json /dis
COPY bookmark_user_id.csv /dis

WORKDIR /dis

CMD ["python", "image_crawler.py"]
  
