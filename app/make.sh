#!/bin/bash

docker network create pixiv-gans-network

docker-compose build --no-cache
docker-compose up -d

