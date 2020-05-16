#!/bin/sh

docker build -t image_crawler .

aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com

docker tag image_crawler:latest 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/image_crawler:latest

docker push 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/image_crawler:latest


