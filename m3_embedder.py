#!/usr/bin/env python3

import datetime
import logging
import multiprocessing
import os
import time

from dotenv import load_dotenv, find_dotenv
import duckdb
from huggingface_hub import hf_hub_download
import lancedb
from minio import Minio
from nltk.tokenize import sent_tokenize
import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow as pa
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app onnx_runner python /app/m3_embedder.py

# Load settings from .env file:
load_dotenv(find_dotenv())

# MinIO local object storage settings:
os.environ['AWS_ENDPOINT'] = f'http://{os.environ['S3_ENDPOINT']}'
os.environ['ALLOW_HTTP'] = 'True'

# LanceDB settings:
LANCEDB_BUCKET_NAME = 'bge-m3'

# Input data settings:
INPUT_ITEMS_PER_BATCH = 100


def centroid_maker_for_lists(embeddings_list: list[list]) -> list:
    average_embedding_list = np.average(embeddings_list, axis=0).tolist()

    return average_embedding_list


def centroid_maker_for_series(group: pd.Series) -> list:
    embeddings_list = group.tolist()

    average_embedding_list = np.average(embeddings_list, axis=0).tolist()

    return average_embedding_list


def batch_generator(item_list, items_per_batch):
    for item in range(0, len(item_list), items_per_batch):
        yield item_list[item:item + items_per_batch]


def newlines_remover(text: str) -> str:
    return text.replace('\n', ' ')


def hugging_face_model_downloader() -> True:
    hf_hub_download(
        repo_id='ddmitov/bge_m3_dense_colbert_onnx',
        filename='model.onnx',
        local_dir='/tmp/model',
        repo_type='model'
    )

    hf_hub_download(
        repo_id='ddmitov/bge_m3_dense_colbert_onnx',
        filename='model.onnx_data',
        local_dir='/tmp/model',
        repo_type='model'
    )

    return True


def object_storage_model_downloader(
    bucket_name: str,
    bucket_prefix: str
) -> True:
    client = Minio(
        os.environ['S3_ENDPOINT'],
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        secure=False
    )

    for item in client.list_objects(
        bucket_name,
        prefix=bucket_prefix,
        recursive=True
    ):
        print(item.object_name)

        client.fget_object(
            bucket_name,
            item.object_name,
            '/tmp/' + item.object_name
        )

    print('')

    return True


def logger_starter():
    start_datetime_string = (
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

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
    total_time = 0

    # Download testing data from a Hugging Face dataset:
    print('')
    print('Downloading testing data from Hugging Face ...')
    print('')

    hf_hub_download(
        repo_id='CloverSearch/cc-news-mutlilingual',
        filename='2021/bg.jsonl.gz',
        local_dir='/tmp',
        repo_type='dataset'
    )

    duckdb.create_function('newlines_remover', newlines_remover)

    duckdb.sql('CREATE SEQUENCE text_id_maker START 1')

    input_list = duckdb.sql(
        '''
            SELECT
                nextval('text_id_maker') AS text_id,
                date_publish_final AS date,
                newlines_remover(title) AS title,
                newlines_remover(maintext) AS text
            FROM read_json_auto("/tmp/2021/bg.jsonl.gz")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND maintext IS NOT NULL
            LIMIT 5000
        '''
    ).to_arrow_table().to_pylist()

    # Download the embedding model:
    print('')
    print('Downloading the BGE-M3 embedding model from object storage ...')
    print('')

    object_storage_model_downloader('bge-m3', 'model')

    # print('')
    # print('Downloading the BGE-M3 embedding model from Hugging Face ...')
    # print('')

    # hugging_face_model_downloader()

    # Set ONNX runtime session configuration:
    onnxrt_options = ort.SessionOptions()

    onnxrt_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnxrt_options.intra_op_num_threads = multiprocessing.cpu_count()

    onnxrt_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    onnxrt_options.add_session_config_entry(
        'session.intra_op.allow_spinning', '1'
    )

    # Initialize ONNX runtime session:
    ort_session = ort.InferenceSession(
        '/tmp/model/model.onnx',
        sess_ptions=onnxrt_options,
        providers=['CPUExecutionProvider']
    )

    # Initialize tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        '/tmp/model/'
    )

    # Create LanceDB tables:
    lance_db = lancedb.connect(f's3://{LANCEDB_BUCKET_NAME}/')

    # Define LanceDB table schemas:
    text_level_table_schema = pa.schema(
        [
            pa.field('text_id',         pa.int64()),
            pa.field('date',            pa.date32()),
            pa.field('title',           pa.utf8()),
            pa.field('text',            pa.utf8()),
            pa.field('dense_embedding', pa.list_(pa.float32(), 1024))
        ]
    )

    sentence_level_table_schema = pa.schema(
        [
            pa.field('text_id',                pa.int64()),
            pa.field('sentence_id',            pa.int64()),
            pa.field('dense_embedding',        pa.list_(pa.float32(), 1024)),
            pa.field('colbert_embedding',      pa.list_(pa.float32(), 1024))
        ]
    )

    # Initialize LanceDB tables:
    text_level_table = lance_db.create_table(
        'text-level',
        schema=text_level_table_schema,
        mode='overwrite'
    )

    sentence_level_table = lance_db.create_table(
        'sentence-level',
        schema=sentence_level_table_schema,
        mode='overwrite'
    )

    # Get input data: 
    batch_list = list(
        batch_generator(
            input_list,
            INPUT_ITEMS_PER_BATCH
        )
    )

    print('')

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

                sentences = sent_tokenize(
                    str(item['title']) + '. ' + str(item['text'])
                )

                # Get text-level table data without embeddings:
                text_item = {}

                text_item['text_id'] = item['text_id']
                text_item['date']    = item['date']
                text_item['title']   = item['title']
                text_item['text']    = item['text']

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
                    sentence_item['dense_embedding'] = dense_embedding

                    # Get sentence-level ColBERT data:
                    colbert_embeddings = outputs[1][0]

                    # ColBERT embeddings for sentences are
                    # centroids of the multiple ColBERT embeddings
                    # produced for every sentence:
                    colbert_centroid = (
                        centroid_maker_for_lists(colbert_embeddings)
                    )

                    sentence_item['colbert_embedding'] = colbert_centroid

                    # Prepare sentence-level list of dictionaries:
                    sentence_list.append(sentence_item)

                    # Log sentence embedding data:
                    print(
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

                print('')

                print(
                    f'batch {batch_number}/{total_batches} - ' +
                    f'item {item_number}/{INPUT_ITEMS_PER_BATCH} - ' +
                    f'embedded for {item_embedding_time_string}'
                )

                print('')

                logger.info(
                    f'batch {batch_number}/{total_batches} - ' +
                    f'item {item_number}/{INPUT_ITEMS_PER_BATCH} - ' +
                    f'embedded for {item_embedding_time_string}'
                )

            # Create dataframes for LanceDB table data:
            text_dataframe = pd.DataFrame(text_list)

            sentence_dataframe = pd.DataFrame(sentence_list)

            # Prepare text-level LanceDB table data.
            # Dense embeddings for whole texts are
            # centroids of the dense embeddings of
            # the corresponding sentences from the sentence-level table:
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

            combined_text_dataframe = pd.merge(
                text_dataframe,
                aggregated_text_dataframe,
                on='text_id',
                how='left'
            )

            # Add data to the LanceDB tables:
            text_level_table.add(combined_text_dataframe.to_dict('records'))

            sentence_level_table.add(sentence_dataframe.to_dict('records'))

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

            print(
                f'batch {batch_number}/{total_batches} ' +
                f'embedded for {batch_embedding_time_string}'
            )

            print(f'total time: {total_time_string}')
            print('')

            logger.info(
                f'batch {batch_number}/{total_batches} ' +
                f'embedded for {batch_embedding_time_string}'
            )

            logger.info(f'total time: {total_time_string}')
    except (KeyboardInterrupt, SystemExit):
        print('\n')
        exit(0)


if __name__ == '__main__':
    main()
