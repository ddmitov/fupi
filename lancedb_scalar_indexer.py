#!/usr/bin/env python3

from dotenv import load_dotenv, find_dotenv
import os

import lancedb

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 fupi python /app/lancedb_scalar_indexer.py

load_dotenv(find_dotenv())

os.environ['AWS_ENDPOINT'] = f'http://{os.environ['S3_ENDPOINT']}'
os.environ['ALLOW_HTTP'] = 'True'


def main():
    lance_db = lancedb.connect(f's3://{os.environ['LANCEDB_BUCKET_NAME']}/')

    weighted_tokens_table = lance_db.open_table('weighted-tokens')

    weighted_tokens_table.create_scalar_index('token',   replace=True)
    weighted_tokens_table.create_scalar_index('text_id', replace=True)


if __name__ == '__main__':
    main()
