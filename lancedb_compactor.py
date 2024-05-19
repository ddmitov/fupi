#!/usr/bin/env python3

from dotenv import load_dotenv, find_dotenv
import os

import lancedb

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app fupi python /app/lancedb_compactor.py

load_dotenv(find_dotenv())

# LanceDB object storage settings:
os.environ['AWS_ENDPOINT'] = f'http://{os.environ['MINIO_ENDPOINT_S3']}'
os.environ['AWS_ACCESS_KEY_ID'] = os.environ['MINIO_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['MINIO_SECRET_ACCESS_KEY']
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['ALLOW_HTTP'] = 'True'


def main():
    lance_db = lancedb.connect(f's3://{os.environ['MINIO_BUCKET_NAME']}/')

    text_level_table = lance_db.open_table('text-level')
    text_level_table.compact_files()

    sentence_level_table = lance_db.open_table('sentence-level')
    sentence_level_table.compact_files()


if __name__ == '__main__':
    main()
