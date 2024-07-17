#!/usr/bin/bash

# ./utilities/models_loader_prod.sh

source .env

mkdir ./temp

# Download models from Hugging Face to a temporary directory:
mkdir ./temp/
mkdir ./temp/bge-m3
mkdir ./temp/ct2fast-m2m100_418m

cd ./temp/bge-m3

wget https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx/resolve/main/model.onnx
wget https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx/resolve/main/model.onnx_data
wget https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx/resolve/main/sentencepiece.bpe.model
wget https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx/resolve/main/special_tokens_map.json
wget https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx/resolve/main/tokenizer_config.json
wget https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx/resolve/main/tokenizer.json

cd ..
cd ./ct2fast-m2m100_418m

wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/config.json
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/generation_config.json
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/model.bin
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/sentencepiece.bpe.model
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/shared_vocabulary.txt
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/special_tokens_map.json
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/tokenizer_config.json
wget https://huggingface.co/michaelfeil/ct2fast-m2m100_418M/resolve/main/vocab.json

cd ../..

# Download rclone:
if ! test -f ./rclone; then
  wget https://downloads.rclone.org/rclone-current-linux-amd64.zip
  unzip -a rclone-current-linux-amd64.zip
  mv ./rclone-*-linux-amd64/rclone ./rclone
  chmod 755 rclone
  rm -f ./rclone-current-linux-amd64.zip
  rm -rf ./rclone-*-linux-amd64
fi

# Start rclone with a temporary configuration file and
# upload models to production object storage:
cat << EOF > ./rclone.conf
[fupi]
type = s3
provider = Cloudflare
access_key_id = $PROD_ACCESS_KEY_ID
secret_access_key = $PROD_SECRET_ACCESS_KEY
endpoint = $PROD_ENDPOINT_S3
acl = private
no_check_bucket = true
EOF

./rclone sync --config ./rclone.conf --progress ./temp/$PROD_MODELS_BUCKET fupi:$PROD_MODELS_BUCKET/

# Clean up:
rm -f ./rclone.conf
rm -rf ./temp
