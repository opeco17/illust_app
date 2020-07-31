#!/bin/bash

unzip -d app/src/ app/src/static.zip
unzip -d app/src/machine_learning/feature_extraction/ app/src/machine_learning/feature_extraction/feature.zip

docker network create pixiv-gans-network

#docker-compose build --no-cache
#docker-compose up -d

docker-compose up -d --build
