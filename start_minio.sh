#!/usr/bin/bash

docker run \
   -p 9000:9000 \
   -p 9001:9001 \
   --rm \
   --user $(id -u):$(id -g) \
   --name minio \
   -e "MINIO_ROOT_USER=admin" \
   -e "MINIO_ROOT_PASSWORD=password" \
   -v ./data/minio:/data \
   quay.io/minio/minio server /data --console-address ":9001"
