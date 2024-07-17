#!/usr/bin/env python3

import os
import shutil

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import hf_hub_download
from minio.error import S3Error
from minio import Minio

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app fupi python /app/utilities/models_loader_dev.py

TEMP_DIR = '/app/data/models'

# Load settings from .env file:
load_dotenv(find_dotenv())

# Object storage settings:
os.environ['AWS_ENDPOINT']          = os.environ['DEV_ENDPOINT_S3']
os.environ['AWS_ACCESS_KEY_ID']     = os.environ['DEV_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['DEV_SECRET_ACCESS_KEY']
bucket_name                         = os.environ['DEV_MODELS_BUCKET']


def model_files_downloader(
    hugging_face_repository: str,
    model_file_list: list,
    prefix: str
) -> True:
    for model_file_name in model_file_list:
        hf_hub_download(
            repo_id=hugging_face_repository,
            filename=model_file_name,
            local_dir=f'{TEMP_DIR}/{prefix}',
            repo_type='model'
        )

    return True


def model_files_uploader(
    minio_client: Minio,
    bucket_name: str,
    prefix:str,
    model_file_list: list
) -> True:

    for model_file_name in model_file_list:
        source_file_name = (
            f'{TEMP_DIR}/{prefix}/{model_file_name}'
        )

        destination_file_name = f'{prefix}/{model_file_name}'

        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f'{bucket_name} bucket was created.')

        minio_client.fput_object(
            bucket_name, destination_file_name, source_file_name,
        )

        print(f'Successfully uploaded {model_file_name}.')

    return True


def models_loader(minio_client: Minio) -> True:
    # Embedding model file list:
    embedding_model_file_list = [
        'model.onnx',
        'model.onnx_data',
        'sentencepiece.bpe.model',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'tokenizer.json',
    ]

    # Translation model file list:
    translation_model_file_list = [
        'config.json',
        'generation_config.json',
        'model.bin',
        'sentencepiece.bpe.model',
        'shared_vocabulary.txt',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.json'
    ]

    # Download the embedding model to a temporary folder:
    print('')
    print('Downloading the embedding model ...')
    print('')

    model_files_downloader(
        'ddmitov/bge_m3_dense_colbert_onnx',
        embedding_model_file_list,
        'bge-m3'
    )

    # Download the translation model to a temporary folder:
    print('')
    print('Downloading the translation model ...')
    print('')

    model_files_downloader(
        'michaelfeil/ct2fast-m2m100_418M',
        translation_model_file_list,
        'ct2fast-m2m100_418m'
    )

    # Upload the embedding model to object storage:
    print('')
    print('Uploading the embedding model ...')
    print('')

    model_files_uploader(
        minio_client,
        bucket_name,
        'bge-m3',
        embedding_model_file_list
    )

    # Upload the translation model to object storage:
    print('')
    print('Uploading the translation model ...')
    print('')

    model_files_uploader(
        minio_client,
        bucket_name,
        'ct2fast-m2m100_418m',
        translation_model_file_list
    )

    # Remove the temporary folder:
    shutil.rmtree(TEMP_DIR)

    return True



def main():
    # Initiate MinIO client:
    endpoint = str(os.environ['AWS_ENDPOINT'])
    secure_mode = None

    if 'https' in endpoint:
        endpoint = endpoint.replace('https://', '')
        secure_mode = True
    else:
        endpoint = endpoint.replace('http://', '')
        secure_mode = False

    minio_client = Minio(
        endpoint,
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        secure=secure_mode
    )

    try:
        minio_client.stat_object(bucket_name, 'bge-m3/model.onnx')
        print("Models are already available. Skipping model uploading.")
    except S3Error as s3_err:
        models_loader(minio_client)

    return True


if __name__ == '__main__':
    main()
