#!/usr/bin/env python3

import datetime
import json
import logging
import multiprocessing
import os
import time

import lancedb
from nltk.tokenize import sent_tokenize
import numpy as np
import onnxruntime as onnxrt
from optimum.onnxruntime import ORTModelForFeatureExtraction
import pandas as pd
import pyarrow as pa
from transformers import AutoTokenizer, pipeline

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app onnx_runner python /app/e5_onnx_embedder.py

# Input file settings:
INPUT_DIRECTORY  = '/app/data/json/en/'
INPUT_FILES_PER_BATCH = 100

# MinIO object storage settings:
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['AWS_ENDPOINT'] = 'http://172.17.0.1:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# LanceDB settings:
LANCEDB_BUCKET_NAME = 'text-centroids'
LANCEDB_TABLE_NAME = 'fupi'


def centroid_maker(group: pd.Series) -> list:
    embeddings_list = group.tolist()

    average_embedding_list = np.average(embeddings_list, axis=0).tolist()

    return average_embedding_list


def batch_generator(item_list, items_per_batch):
    for item in range(0, len(item_list), items_per_batch):
        yield item_list[item:item + items_per_batch]


def logger_starter():
    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename='/app/logs/e5_onnx_embedder.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def main():
    # Start logging: 
    logger = logger_starter()
    total_time = 0

    # Set options for an ONNX runtime session:
    onnxrt_options = onnxrt.SessionOptions()

    onnxrt_options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    onnxrt_options.intra_op_num_threads = multiprocessing.cpu_count()

    onnxrt_options.graph_optimization_level = (
        onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    
    onnxrt_options.add_session_config_entry(
        'session.intra_op.allow_spinning', '1'
    )

    # Initialize tokenizer:
    tokenizer = AutoTokenizer.from_pretrained("/app/data/e5")

    tokenizer_kwargs = {'truncation': True}

    # Initialize embedding model:
    model = ORTModelForFeatureExtraction.from_pretrained(
        "/app/data/e5",
        session_options=onnxrt_options,
        providers=['CPUExecutionProvider']
    )

    # Initialize transformers pipeline:
    onnx_extractor = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer
    )

    # Connect to LanceDB:
    lance_db = lancedb.connect(f's3://{LANCEDB_BUCKET_NAME}/')

    # Crete LanceDB table:
    arrow_schema = pa.schema(
        [
            pa.field('text_id', pa.int64()),
            pa.field('title',   pa.utf8()),
            pa.field('text',    pa.utf8()),
            pa.field('vector',  pa.list_(pa.float64(), 1024)),
        ]
    )

    lancedb_table = lance_db.create_table(
        LANCEDB_TABLE_NAME,
        schema=arrow_schema,
        mode='overwrite'
    )

    # Get the input file list: 
    filename_list = sorted(os.listdir(INPUT_DIRECTORY))

    filename_batch_list = list(
        batch_generator(
            filename_list,
            INPUT_FILES_PER_BATCH
        )
    )

    print('')

    try:
        batch_number = 0
        total_batches = len(filename_batch_list)

        # Iterate over all batches:
        for batch in filename_batch_list:
            batch_embedding_start = time.time()

            batch_number += 1

            id_list_text_table = []
            title_list = []
            text_list = []

            id_list_embeddings_table = []
            embeddings_list = []

            file_number = 0
            total_files_in_batch = len(batch)

            # Iterate over all files in a batch:
            for filename in batch:
                file_number += 1

                with open(INPUT_DIRECTORY + filename) as file:
                    input = json.load(file)

                    # Save only textual data:
                    id_list_text_table.append(int(input['id']))
                    title_list.append(str(input['title']))
                    text_list.append(str(input['contents']))

                    # Split each input text in sentences:
                    sentences = sent_tokenize(
                        str(input['title']) +
                        '. ' +
                        str(input['contents'])
                    )

                    sentences_to_embedd = []

                    # Do not embedd single-character sentences:
                    for sentence in sentences:
                        if len(sentence) > 1:
                            sentences_to_embedd.append(sentence)

                    total_sentences = len(sentences_to_embedd)
                    sentence_number = 0

                    # Iterate over all sentences in a file:
                    for sentence in sentences_to_embedd:
                        sentence_number += 1

                        # Calculate embeddings for each sentence:
                        embeddings = onnx_extractor(
                            sentence,
                            **tokenizer_kwargs
                        )

                        # Save only embeddings:
                        id_list_embeddings_table.append(int(input['id']))
                        embeddings_list.append(embeddings[0][0])

                        print(
                            f'batch {batch_number}/{total_batches} - ' +
                            f'file {file_number}/{total_files_in_batch} - ' +
                            f'sentence {sentence_number}/{total_sentences}'
                        )

                        logger.info(
                            f'batch {batch_number}/{total_batches} - ' +
                            f'file {file_number}/{total_files_in_batch} - ' +
                            f'sentence {sentence_number}/{total_sentences}'
                        )

            text_dict = {
                'text_id': id_list_text_table,
                'title':   title_list,
                'text':    text_list
            }

            text_dataframe = pd.DataFrame(text_dict)

            embeddings_dict = {
                'text_id': id_list_embeddings_table,
                'vector':  embeddings_list
            }

            embeddings_dataframe = pd.DataFrame(embeddings_dict)

            aggregated_embeddings_dataframe = (
                embeddings_dataframe.groupby(
                    [
                        'text_id'
                    ]
                ).agg(
                    {
                        'vector': [centroid_maker]
                    }
                )
            ).reset_index()

            aggregated_embeddings_dataframe.columns = (
                aggregated_embeddings_dataframe.columns.get_level_values(0)
            )

            combined_dataframe = pd.merge(
                text_dataframe,
                aggregated_embeddings_dataframe,
                on='text_id',
                how='left'
            )

            print('')
            print(combined_dataframe)

            lancedb_table.add(combined_dataframe.to_dict('records'))

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
print('')

if __name__ == '__main__':
    main()
