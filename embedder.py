#!/usr/bin/env python3

import os
import logging
import datetime

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from fupi import HuggingFaceDataset, LanceDBEmbedder
from fupi.utils import (model_downloader_from_object_storage, 
                        ort_session_starter_for_text_embedding)


# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app fupi python /app/embedder.py

# Load settings from .env file:
load_dotenv(find_dotenv())

os.environ['AWS_ENDPOINT']          = os.environ['DEV_ENDPOINT_S3']
os.environ['AWS_ACCESS_KEY_ID']     = os.environ['DEV_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['DEV_SECRET_ACCESS_KEY']
os.environ['AWS_REGION']            = 'us-east-1'
os.environ['ALLOW_HTTP']            = 'True'

# Input data settings:
TOTAL_NUM_TEXTS = 100
INPUT_ITEMS_PER_BATCH = 10


def logger_starter():
    start_datetime_string = (
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    log_dir = '/app/data/logs'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename=f'/app/data/logs/m3_embedder_{start_datetime_string}.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def main():
    # Start logging: 
    logger = logger_starter()
    logger.info('Loading data ...')

    print('')
    print('Loading data ...')
    print('')

    print('Downloading dataset...')
    hf_hub_download(
        repo_id='CloverSearch/cc-news-mutlilingual',
        filename='2021/bg.jsonl.gz',
        local_dir='/app/data/huggingface',
        repo_type='dataset'
    )
    print('Downloading model...')
    model_downloader_from_object_storage(
        os.environ['DEV_MODELS_BUCKET'],
        'bge-m3'
    )

    ort_session = ort_session_starter_for_text_embedding()
    tokenizer = AutoTokenizer.from_pretrained('ddmitov/bge_m3_dense_colbert_onnx')

    print('Embedding dataset...')
    try:
        dataset = HuggingFaceDataset(path="/app/data/huggingface/2021/bg.jsonl.gz", 
                                     batch_size=INPUT_ITEMS_PER_BATCH, 
                                     num_samples=TOTAL_NUM_TEXTS)

        embedder = LanceDBEmbedder(tokenizer=tokenizer, model=ort_session)
        embedder.embed(dataset)
    except (KeyboardInterrupt, SystemExit):
        print('\n')
        exit(0)

    logger.info('All LanceDB tables are compacted.')

    print('All LanceDB tables are compacted.')
    print('')

    return True


if __name__ == '__main__':
    main()
