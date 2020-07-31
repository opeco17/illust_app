#!/bin/bash

docker-compose down
rm -rf app/src/static
rm -rf app/src/machine_learning/feature_extraction/feature

rm -rf app/src/__MACOSX
rm -rf app/src/machine_learning/feature_extraction/__MACOSX

rm -r app/log/uwsgi.log
rm -r nginx/log/access.log
rm -r nginx/log/error.log
