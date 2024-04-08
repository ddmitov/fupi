#!/usr/bin/env python3

import datetime
import logging
import multiprocessing
import os
import time

import duckdb
from huggingface_hub import hf_hub_download
import lancedb
from nltk.tokenize import sent_tokenize
import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow as pa
from sentence_transformers.quantization import quantize_embeddings
from torch import FloatTensor
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app onnx_runner python /app/m3_embedder.py

# Input data settings:
INPUT_ITEMS_PER_BATCH = 100

# MinIO local object storage settings:
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['AWS_ENDPOINT'] = 'http://172.17.0.1:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# LanceDB settings:
LANCEDB_BUCKET_NAME = 'bge-m3'


def embeddings_binarizer(row, column_name):
    dense_embedding_list = row[column_name]

    dense_embedding_tensor = FloatTensor(
        dense_embedding_list
    )

    binary_embedding = quantize_embeddings(
        dense_embedding_tensor.reshape(1, -1),
        precision='binary'
    ).tolist()[0]

    return binary_embedding


def centroid_maker(group: pd.Series) -> list:
    embeddings_list = group.tolist()

    average_embedding_list = np.average(embeddings_list, axis=0).tolist()

    return average_embedding_list


def batch_generator(item_list, items_per_batch):
    for item in range(0, len(item_list), items_per_batch):
        yield item_list[item:item + items_per_batch]


def newlines_remover(text: str) -> str:
    return text.replace('\n', ' ')


def logger_starter():
    start_datetime_string = (
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename=f'/app/data/logs/m3_onnx_embedder_{start_datetime_string}.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def main():
    # Start logging: 
    logger = logger_starter()
    total_time = 0

    # Download testing data from a Hugging Face dataset:
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

    # Download embedding model from Hugging Face:
    hf_hub_download(
        repo_id='ddmitov/bge_m3_dense_colbert_onnx',
        filename='model.onnx',
        local_dir='/tmp',
        repo_type='model'
    )

    hf_hub_download(
        repo_id='ddmitov/bge_m3_dense_colbert_onnx',
        filename='model.onnx_data',
        local_dir='/tmp',
        repo_type='model'
    )

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
        '/tmp/model.onnx',
        sess_ptions=onnxrt_options,
        providers=['CPUExecutionProvider']
    )

    # Initialize tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        'ddmitov/bge_m3_dense_colbert_onnx'
    )

    # Create LanceDB tables:
    lance_db = lancedb.connect(f's3://{LANCEDB_BUCKET_NAME}/')

    dense_table_schema = pa.schema([
        pa.field('text_id',                pa.int64()),
        pa.field('date',                   pa.date32()),
        pa.field('title',                  pa.utf8()),
        pa.field('text',                   pa.utf8()),
        pa.field('dense_embedding',        pa.list_(pa.float32(), 1024)),
        pa.field('dense_embedding_binary', pa.list_(pa.float32(), 128))
    ])

    colbert_table_schema = pa.schema([
        pa.field('text_id',           pa.int64()),
        pa.field('sentence_id',       pa.int64()),
        pa.field('date',              pa.date32()),
        pa.field('sentence',          pa.utf8()),
        pa.field('colbert_embedding', pa.list_(pa.float32(), 1024))
    ])

    dense_table = lance_db.create_table(
        'dense',
        schema=dense_table_schema,
        mode='overwrite'
    )

    colbert_table = lance_db.create_table(
        'colbert',
        schema=colbert_table_schema,
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

        # Iterate over all batches:
        for batch in batch_list:
            batch_embedding_start = time.time()

            batch_number += 1

            # Produce Dense and ColBERT embeddings:
            text_list = []
            dense_results_list = []
            colbert_results_list = []

            item_number = 0

            for item in batch:
                item_number += 1

                item_embedding_start = time.time()

                sentences = sent_tokenize(
                    str(item['title']) + '. ' + str(item['text'])
                )

                text_item = {}

                text_item['text_id'] = item['text_id']
                text_item['date']    = item['date']
                text_item['title']   = item['title']
                text_item['text']    = item['text']

                text_list.append(text_item)

                total_sentences = len(sentences)
                sentence_number = 0

                for sentence in sentences:
                    sentence_number += 1

                    tokenized_input = tokenizer(
                        sentence,
                        truncation=True,
                        return_tensors='np'
                    )

                    onnx_input = {
                        key: ort.OrtValue.ortvalue_from_numpy(value)
                        for key, value in tokenized_input.items()
                    }

                    outputs = ort_session.run(None, onnx_input)

                    dense_item = {}

                    dense_item['text_id']         = item['text_id']
                    dense_item['sentence_id']     = sentence_number
                    dense_item['dense_embedding'] = outputs[0][0]

                    dense_results_list.append(dense_item)

                    colbert_embeddings = outputs[1][0]

                    colbert_embeddings_number = 0

                    for colbert_embedding in colbert_embeddings:
                        colbert_embeddings_number += 1

                        colbert_item = {}

                        colbert_item['text_id']           = item['text_id']
                        colbert_item['sentence_id']       = sentence_number
                        colbert_item['date']              = item['date']
                        colbert_item['sentence']          = sentence
                        colbert_item['embedding_id']      = colbert_embeddings_number
                        colbert_item['colbert_embedding'] = colbert_embedding

                        colbert_results_list.append(colbert_item)

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

            text_dataframe = pd.DataFrame(text_list)

            dense_dataframe = pd.DataFrame(dense_results_list)

            colbert_dataframe = pd.DataFrame(colbert_results_list)

            aggregated_dense_dataframe = (
                dense_dataframe.groupby(
                    [
                        'text_id'
                    ]
                ).agg(
                    {
                        'dense_embedding': [centroid_maker]
                    }
                )
            ).reset_index()

            aggregated_dense_dataframe.columns = (
                aggregated_dense_dataframe.columns.get_level_values(0)
            )

            aggregated_dense_dataframe['dense_embedding_binary'] = (
                aggregated_dense_dataframe.apply(
                    lambda row: embeddings_binarizer(row, 'dense_embedding'),
                    axis=1
                )
            )

            combined_dense_dataframe = pd.merge(
                text_dataframe,
                aggregated_dense_dataframe,
                on='text_id',
                how='left'
            )

            aggregated_colbert_dataframe = (
                colbert_dataframe.groupby(
                    [
                        'text_id',
                        'sentence_id'
                    ]
                ).agg(
                    {
                        'date': ['first'],
                        'sentence': ['first'],
                        'colbert_embedding': [centroid_maker]
                    }
                )
            ).reset_index()

            aggregated_colbert_dataframe.columns = (
                aggregated_colbert_dataframe.columns.get_level_values(0)
            )

            dense_table.add(combined_dense_dataframe.to_dict('records'))

            colbert_table.add(aggregated_colbert_dataframe.to_dict('records'))

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

            print('')
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
