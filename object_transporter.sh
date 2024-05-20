#!/usr/bin/bash

source .env

if ! test -f ./mc; then
  wget https://dl.minio.io/client/mc/release/linux-amd64/mc
  chmod 755 mc
  mkdir ./mc_cfg
fi

./mc --config-dir ./mc_cfg alias set minio http://0.0.0.0:9000 $MINIO_ACCESS_KEY_ID $MINIO_SECRET_ACCESS_KEY
./mc --config-dir ./mc_cfg alias set tigris $TIGRIS_ENDPOINT_S3 $TIGRIS_ACCESS_KEY_ID $TIGRIS_SECRET_ACCESS_KEY

./mc --config-dir ./mc_cfg cp --recursive minio/$MINIO_BUCKET_NAME/sentence-level.lance tigris/$TIGRIS_BUCKET_NAME
./mc --config-dir ./mc_cfg cp --recursive minio/$MINIO_BUCKET_NAME/text-level.lance tigris/$TIGRIS_BUCKET_NAME

rm -rf ./mc_cfg
