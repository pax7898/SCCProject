#!/bin/bash

docker build --tag customer_behavior .
docker tag customer_behavior pax7898/customer_behavior
docker push docker.io/pax7898/customer_behavior