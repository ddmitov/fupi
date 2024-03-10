#!/usr/bin/env python3

import os

import boto3
import botocore
import lancedb
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app onnx_embedder python /app/e5_onnx_searcher.py


def main():
    # Initialize the embedding model:
    onnx_session = InferenceSession(
        '/etc/model/embedder.onnx',
        providers=['CPUExecutionProvider']
    )

    tokenizer = AutoTokenizer.from_pretrained('/etc/tokenizer')

    # Prepare an S3 connection:
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
    os.environ['AWS_ENDPOINT'] = 'http://172.17.0.1:9000'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

    boto_session = boto3.Session()
    boto_session.client(
        's3',
        config=botocore.config.Config(
            s3={'addressing_style': 'path'}
        )
    )

    # Define LanceDB table:
    BUCKET_NAME = 'lancedb'

    lance_db = lancedb.connect(f's3://{BUCKET_NAME}/')
    lancedb_table = lance_db.open_table('fupi')

    # Vectorize the search input and search:
    tokenized_input = tokenizer(
        str('злоупотреби в Калифорния'),
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    embeddings = onnx_session.run(None, dict(tokenized_input))

    result = lancedb_table.search(embeddings[0][0]).limit(5).to_pandas()

    print(result)


if __name__ == '__main__':
    main()
