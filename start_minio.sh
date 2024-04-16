#!/usr/bin/bash

source .env

docker run \
   -p 9000:9000 \
   -p 9001:9001 \
   --rm \
   --user $(id -u):$(id -g) \
   --name minio \
   -e "MINIO_ROOT_USER=$AWS_ACCESS_KEY_ID" \
   -e "MINIO_ROOT_PASSWORD=$AWS_SECRET_ACCESS_KEY" \
   -v ./data/minio:/data \
   quay.io/minio/minio server /data --console-address ":9001"
