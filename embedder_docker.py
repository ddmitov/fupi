#!/usr/bin/env python3

import datetime
import logging
import os
import time

from dotenv import load_dotenv, find_dotenv
import duckdb
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import pandas as pd
import pysbd
from transformers import AutoTokenizer
from minio import Minio

from fupi import model_downloader_from_hugging_face
from fupi import ort_session_starter_for_text_embedding
from fupi import lancedb_tables_creator
from fupi import centroid_maker_for_arrays
from fupi import centroid_maker_for_series

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app fupi python /app/embedder.py

# Load settings from .env file:
load_dotenv(find_dotenv())

# LanceDB object storage settings:
os.environ['AWS_ENDPOINT']          = os.environ['S3_ENDPOINT']
os.environ['AWS_ACCESS_KEY_ID']     = os.environ['ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['SECRET_ACCESS_KEY']
os.environ['AWS_REGION']            = 'us-east-1'

os.environ['ALLOW_HTTP'] = 'True'

# Input data settings:
TOTAL_NUM_TEXTS = 100
INPUT_ITEMS_PER_BATCH = 20


def batch_generator(item_list, items_per_batch):
    for item in range(0, len(item_list), items_per_batch):
        yield item_list[item:item + items_per_batch]


def newlines_remover(text: str) -> str:
    return text.replace('\n', ' ')


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
        filename=f'{log_dir}/m3_embedder_{start_datetime_string}.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def main():
    # Start logging: 
    logger = logger_starter()
    total_time = 0

    # Creating LanceDB tables:
    minio_client = init_minio()
    create_dataset_bucket(minio_client, 
                          bucket_name=os.environ['LANCEDB_BUCKET'])

    text_level_table, sentence_level_table = (
        lancedb_tables_creator(os.environ['LANCEDB_BUCKET'])
    )

    logger.info('')
    logger.info('Downloading testing data from Hugging Face ...')
    logger.info('')

    hf_hub_download(
        repo_id='CloverSearch/cc-news-mutlilingual',
        filename='2021/bg.jsonl.gz',
        local_dir='/home/fupi_data',
        repo_type='dataset'
    )

    logger.info("Create function...")
    duckdb.create_function('newlines_remover', newlines_remover)

    duckdb.sql('CREATE SEQUENCE text_id_maker START 1')
    
    logger.info("SELECT Query...")
    input_list = duckdb.sql(
        f'''
            SELECT
                nextval('text_id_maker') AS text_id,
                date_publish_final AS date,
                newlines_remover(title) AS title,
                newlines_remover(maintext) AS text
            FROM read_json_auto("/home/fupi_data/2021/bg.jsonl.gz")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND maintext IS NOT NULL
                AND title NOT LIKE '%...'
            LIMIT {TOTAL_NUM_TEXTS}
        '''
    ).to_arrow_table().to_pylist()

    logger.info('')
    logger.info('Downloading the BGE-M3 embedding model from Hugging Face ...')
    logger.info('')

    logger.info("Input list:")
    logger.info(input_list)
    logger.info("\n\n")

    model_downloader_from_hugging_face()

    ort_session = ort_session_starter_for_text_embedding()

    # Initialize tokenizer:
    tokenizer = AutoTokenizer.from_pretrained('ddmitov/bge_m3_dense_colbert_onnx')

    # Get input data: 
    batch_list = list(
        batch_generator(
            input_list,
            INPUT_ITEMS_PER_BATCH
        )
    )

    # Initialize sentence segmenter:
    segmenter = pysbd.Segmenter(language='bg', clean=False)

    logger.info('')

    try:
        batch_number = 0
        total_batches = len(batch_list)

        # Iterate over all batches of texts:
        for batch in batch_list:
            batch_embedding_start = time.time()

            batch_number += 1

            text_list = []
            sentence_list = []

            item_number = 0

            # Iterate over all texts:
            for item in batch:
                item_number += 1

                item_embedding_start = time.time()

                # Split every text to sentences:
                sentences = segmenter.segment(str(item['text']))

                # Get text-level table data without embeddings:
                text_item = {}

                text_item['text_id'] = item['text_id']
                text_item['date']    = item['date']
                text_item['title']   = item['title']

                # Prepare text-level list of dictionaries:
                text_list.append(text_item)

                total_sentences = len(sentences)
                sentence_number = 0

                # Iterate over all sentences in a text:
                for sentence in sentences:
                    sentence_number += 1

                    # Tokenize input sentence:
                    tokenized_input = tokenizer(
                        sentence,
                        truncation=True,
                        return_tensors='np'
                    )

                    # Get sentence-level embeddings:
                    onnx_input = {
                        key: ort.OrtValue.ortvalue_from_numpy(value)
                        for key, value in tokenized_input.items()
                    }

                    outputs = ort_session.run(None, onnx_input)

                    # Get sentence-level dense data:
                    dense_embedding = outputs[0][0]

                    sentence_item = {}

                    sentence_item['text_id']         = item['text_id']
                    sentence_item['sentence_id']     = sentence_number
                    sentence_item['sentence']        = sentence
                    sentence_item['dense_embedding'] = dense_embedding

                    # Get sentence-level ColBERT data:
                    colbert_embeddings = outputs[1][0]

                    # ColBERT embeddings for sentences are
                    # centroids of the multiple ColBERT embeddings
                    # produced for every sentence:
                    colbert_centroid = (
                        centroid_maker_for_arrays(colbert_embeddings)
                    )

                    sentence_item['colbert_embedding'] = colbert_centroid

                    sentence_list.append(sentence_item)

                    # Log sentence embedding data:
                    logger.info(
                        f'batch {batch_number}/{total_batches} - ' +
                        f'item {item_number}/{INPUT_ITEMS_PER_BATCH} - ' +
                        f'sentence {sentence_number}/{total_sentences}'
                    )

                    logger.info(
                        f'batch {batch_number}/{total_batches} - ' +
                        f'item {item_number}/{INPUT_ITEMS_PER_BATCH} - ' +
                        f'sentence {sentence_number}/{total_sentences}'
                    )

                # Get runtime data for logging:
                item_embedding_end = time.time()

                item_embedding_time = round(
                    (item_embedding_end - item_embedding_start),
                    3
                )

                item_embedding_time_string = str(
                    datetime.timedelta(seconds=item_embedding_time)
                )

                logger.info('')

                logger.info(
                    f'batch {batch_number}/{total_batches} - ' +
                    f'item {item_number}/{INPUT_ITEMS_PER_BATCH} - ' +
                    f'embedded for {item_embedding_time_string}'
                )

                logger.info('')

                logger.info(
                    f'batch {batch_number}/{total_batches} - ' +
                    f'item {item_number}/{INPUT_ITEMS_PER_BATCH} - ' +
                    f'embedded for {item_embedding_time_string}'
                )

            # Data processing for the sentence-level LanceDB table:
            sentence_dataframe = pd.DataFrame(sentence_list)

            # Data processing for the text-level LanceDB table.
            # Dense embeddings for the text-level LanceDB table are
            # centroids of the sentence-level dense embeddings:
            aggregated_text_dataframe = (
                sentence_dataframe.groupby(
                    [
                        'text_id'
                    ]
                ).agg(
                    {
                        'dense_embedding': [centroid_maker_for_series]
                    }
                )
            ).reset_index()

            aggregated_text_dataframe.columns = (
                aggregated_text_dataframe.columns.get_level_values(0)
            )

            text_dataframe = pd.DataFrame(text_list)

            combined_text_dataframe = pd.merge(
                text_dataframe,
                aggregated_text_dataframe,
                on='text_id',
                how='left'
            )

            # Add data to the LanceDB tables:
            sentence_level_table.add(sentence_dataframe)
            text_level_table.add(combined_text_dataframe)

            # Calculate and log batch processing time:
            batch_embedding_end = time.time()

            batch_embedding_time = round(
                (batch_embedding_end - batch_embedding_start),
                3
            )

            batch_embedding_time_string = str(
                datetime.timedelta(seconds=batch_embedding_time)
            )

            total_time = round(total_time + batch_embedding_time, 3)
            total_time_string = str(datetime.timedelta(seconds=total_time))

            logger.info(
                f'batch {batch_number}/{total_batches} ' +
                f'embedded for {batch_embedding_time_string}'
            )

            logger.info(f'total time: {total_time_string}')
            logger.info('')

            logger.info(
                f'batch {batch_number}/{total_batches} ' +
                f'embedded for {batch_embedding_time_string}'
            )

            logger.info(f'total time: {total_time_string}')
    except (KeyboardInterrupt, SystemExit):
        logger.info('\n')
        exit(0)

    # Compact all newly created LanceDB tables:
    logger.info('')
    logger.info('Compacting LanceDB tables ...')

    text_level_table.compact_files()
    sentence_level_table.compact_files()

    logger.info('All LanceDB tables are compacted.')
    logger.info('')

    return True


def init_minio() -> Minio:
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

    return minio_client


def create_dataset_bucket(minio_client: Minio, bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        print(f'{bucket_name} bucket was created.')


if __name__ == '__main__':
    main()
