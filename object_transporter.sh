#!/usr/bin/bash

source .env

if ! test -f ./mc; then
  wget https://dl.minio.io/client/mc/release/linux-amd64/mc
  chmod 755 mc
  mkdir ./mc_cfg
fi

./mc --config-dir ./mc_cfg alias set minio http://0.0.0.0:9000 $MINIO_ACCESS_KEY_ID $MINIO_SECRET_ACCESS_KEY
./mc --config-dir ./mc_cfg alias set tigris $TIGRIS_ENDPOINT_S3 $TIGRIS_ACCESS_KEY_ID $TIGRIS_SECRET_ACCESS_KEY

./mc --config-dir ./mc_cfg mb tigris/$TIGRIS_LANCEDB_BUCKET
./mc --config-dir ./mc_cfg cp --recursive minio/$MINIO_LANCEDB_BUCKET/sentence-level.lance tigris/$TIGRIS_LANCEDB_BUCKET
./mc --config-dir ./mc_cfg cp --recursive minio/$MINIO_LANCEDB_BUCKET/text-level.lance tigris/$TIGRIS_LANCEDB_BUCKET

./mc --config-dir ./mc_cfg mb tigris/$TIGRIS_MODELS_BUCKET
./mc --config-dir ./mc_cfg cp --recursive minio/$MINIO_MODELS_BUCKET/bge-m3 tigris/$TIGRIS_MODELS_BUCKET

rm -rf ./mc_cfg
