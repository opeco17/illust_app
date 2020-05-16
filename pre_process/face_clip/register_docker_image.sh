#!/bin/sh

docker build -t face_clip .

aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com

docker tag face_clip:latest 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/face_clip:latest

docker push 829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/face_clip:latest