#!/usr/bin/env python3

import datetime
import json
import logging
import multiprocessing
import os
import time

# import boto3
# import botocore
import lancedb
from langchain_text_splitters import CharacterTextSplitter
import numpy as np
import onnxruntime as onnxrt
from optimum.onnxruntime import ORTModelForFeatureExtraction
import pandas as pd
import pyarrow as pa
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app onnx_runner python /app/e5_onnx_embedder.py


class EmbeddingDataset(Dataset):
    def __init__(self, data_list):
      self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
      return self.data_list[index]
    

def average_embeddings_function(group: pd.Series) -> list:
    embeddings_list = group.tolist()

    average_embedding_list = np.average(embeddings_list, axis=0).tolist()[0]

    return average_embedding_list


def get_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename='/app/e5_onnx_embedder_03.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def batch_generator(item_list, items_per_batch):
    for item in range(0, len(item_list), items_per_batch):
        yield item_list[item:item + items_per_batch]


def main():
    logger = get_logger()
    total_time = 0

    INPUT_DIRECTORY  = '/app/data/json/en/'

    # Initialize the embedding model:
    onnxrt_options = onnxrt.SessionOptions()

    onnxrt_options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    onnxrt_options.intra_op_num_threads = multiprocessing.cpu_count()

    # onnxrt_options.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
    # onnxrt_options.inter_op_num_threads = multiprocessing.cpu_count()

    onnxrt_options.graph_optimization_level = (
        onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    onnxrt_options.add_session_config_entry(
        'session.intra_op.allow_spinning', '1'
    )

    model = ORTModelForFeatureExtraction.from_pretrained(
        "/app/data/e5",
        session_options=onnxrt_options,
        providers=['CPUExecutionProvider']
    )

    tokenizer = AutoTokenizer.from_pretrained("/app/data/e5")

    tokenizer_kwargs = {
        'padding':    True,
        'truncation': True,
        'max_length': 512
    }

    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=512,
        chunk_overlap=0
    )

    onnx_extractor = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer
    )

    # Prepare an S3 connection:
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
    os.environ['AWS_ENDPOINT'] = 'http://172.17.0.1:9000'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

    # boto_session = boto3.Session()
    # boto_session.client(
    #     's3',
    #     config=botocore.config.Config(
    #         s3={'addressing_style': 'path'}
    #     )
    # )

    # Connect to LanceDB:
    BUCKET_NAME = 'splitting-batching-test'

    lance_db = lancedb.connect(f's3://{BUCKET_NAME}/')

    # Crete LanceDB table:
    arrow_schema = pa.schema(
        [
            pa.field('id',     pa.int64()),
            pa.field('title',  pa.utf8()),
            pa.field('text',   pa.utf8()),
            pa.field('vector', pa.list_(pa.float64(), 1024)),
        ]
    )

    lancedb_table = lance_db.create_table(
        'fupi',
        schema=arrow_schema,
        mode='overwrite'
    )

    # Get the input file list: 
    filename_list = sorted(os.listdir(INPUT_DIRECTORY))

    filenames_per_batch = 50

    filename_batch_list = list(
        batch_generator(
            filename_list,
            filenames_per_batch
        )
    )

    try:
        batch_number = 0
        total_batches = len(filename_batch_list)

        for batch in filename_batch_list:
            batch_embedding_start = time.time()
            batch_number += 1

            id_list = []
            title_list = []
            text_list = []
            text_splits_list = []
            embeddings_list = []

            # Iterate all files:
            for filename in batch:
                with open(INPUT_DIRECTORY + filename) as file:
                    input = json.load(file)

                    title_splits = list(
                        text_splitter.split_text(str(input['title']))
                    )

                    text_splits = list(
                        text_splitter.split_text(str(input['contents']))
                    )

                    combined_text_splits = title_splits + text_splits

                    for text_split in combined_text_splits:
                        id_list.append(int(input['id']))
                        title_list.append(str(input['title']))
                        text_list.append(str(input['contents']))

                        text_splits_list.append(text_split)

            dataset = EmbeddingDataset(text_splits_list)

            print('')
            print(f'embedding batch {batch_number}/{total_batches} ...')

            for output in onnx_extractor(
                dataset,
                batch_size=100,
                **tokenizer_kwargs
            ):
                embeddings_list.extend(output)

            embeddings_dict = {
                'id':     id_list,
                'title':  title_list,
                'text':   text_list,
                'vector': embeddings_list
            }

            embeddings_dataframe = pd.DataFrame(embeddings_dict)

            aggregated_dataframe = (
                embeddings_dataframe.groupby(
                    [
                        'id'
                    ]
                ).agg(
                    {
                        'title':  ['first'],
                        'text':   ['first'],
                        'vector': [average_embeddings_function]
                    }
                )
            ).reset_index()

            aggregated_dataframe.columns = (
                aggregated_dataframe.columns.get_level_values(0)
            )

            print(aggregated_dataframe)

            lancedb_table.add(aggregated_dataframe.to_dict('records'))

            batch_embedding_end = time.time()
            batch_embedding_time = round(
                (batch_embedding_end - batch_embedding_start),
                3
            )
            batch_embedding_time_string = '{:.3f}'.format(batch_embedding_time)

            total_time = round(total_time + batch_embedding_time, 3)
            total_time_string = str(datetime.timedelta(seconds=total_time))   # Create a logger:

            print(
                f'batch {batch_number}/{total_batches} ' +
                f'embedded for {batch_embedding_time_string}'
            )

            print('')
            print(f'total time: {total_time_string}')

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
