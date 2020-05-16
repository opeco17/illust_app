#!bin/bash

aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com

docker build -t tag_extraction .

docker tag tag_extraction:latest 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/tag_extraction:latest

docker push 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/tag_extraction:latest


