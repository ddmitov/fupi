import os
from multiprocessing import cpu_count

import duckdb
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from minio import Minio
import onnxruntime as ort


def lancedb_tables_creator(
    lancedb_bucket_name: str
) -> tuple[lancedb.table.Table, lancedb.table.Table]:
    lance_db = lancedb.connect(f's3://{lancedb_bucket_name}/')

    text_level_table_schema = pa.schema(
        [
            pa.field('text_id',         pa.int64()),
            pa.field('date',            pa.date32()),
            pa.field('title',           pa.utf8()),
            pa.field('dense_embedding', pa.list_(pa.float32(), 1024))
        ]
    )

    # Define LanceDB table schemas:
    sentence_level_table_schema = pa.schema(
        [
            pa.field('text_id',           pa.int64()),
            pa.field('sentence_id',       pa.int64()),
            pa.field('sentence',          pa.utf8()),
            pa.field('dense_embedding',   pa.list_(pa.float32(), 1024)),
            pa.field('colbert_embedding', pa.list_(pa.float32(), 1024))
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

    return text_level_table, sentence_level_table


def create_dataset_bucket(minio_client: Minio, bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

        print(f'{bucket_name} bucket was created.')


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


def ort_session_starter_for_text_embedding() -> ort.InferenceSession:
    # Set ONNX runtime session configuration:
    onnxrt_options = ort.SessionOptions()

    onnxrt_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnxrt_options.intra_op_num_threads = cpu_count()

    onnxrt_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    onnxrt_options.add_session_config_entry(
        'session.intra_op.allow_spinning', '1'
    )

    # Initialize ONNX runtime session:
    ort_session = ort.InferenceSession(
        '/tmp/bge-m3/model.onnx',
        sess_ptions=onnxrt_options,
        providers=['CPUExecutionProvider']
    )

    return ort_session


def model_downloader_from_object_storage(
    bucket_name: str,
    bucket_prefix: str
) -> True:
    endpoint = str(os.environ['AWS_ENDPOINT'])
    secure_mode = None

    if 'https' in endpoint:
        endpoint = endpoint.replace('https://', '')
        secure_mode = True
    else:
        endpoint = endpoint.replace('http://', '')
        secure_mode = False

    client = Minio(
        endpoint,
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        secure=secure_mode
    )

    for item in client.list_objects(
        bucket_name,
        prefix=bucket_prefix,
        recursive=True
    ):

        client.fget_object(
            bucket_name,
            item.object_name,
            '/tmp/' + item.object_name
        )

    return True


def fupi_dense_vectors_searcher(
    sentence_level_table: lancedb.table.Table,
    query_dense_embedding: np.ndarray
) -> pd.DataFrame:
    dense_initial_dataframe = sentence_level_table.search(
        query_dense_embedding.tolist(),
        vector_column_name='dense_embedding'
    )\
    .select(
        [
            'text_id',
            'sentence_id',
            'sentence'
        ]
    )\
    .limit(10)\
    .to_pandas()

    dense_search_result_dataframe = duckdb.query(
        f'''
            WITH
                combined_results_cte AS (
                    SELECT
                        did._distance AS distance,
                        did.text_id,
                        tlat.date,
                        tlat.title,
                        did.sentence_id,
                        did.sentence
                    FROM dense_initial_dataframe did
                        JOIN text_level_arrow_table tlat ON
                            tlat.text_id = did.text_id
                    WHERE LENGTH(did.sentence) > 3
                    ORDER BY
                        did.text_id ASC,
                        did.sentence_id ASC
                )

            -- User-facing query:
            SELECT
                distance,
                text_id,
                date,
                title,
                string_agg(sentence_id, ', ') AS sentence_ids,
                string_agg(sentence, ' -- ') AS sentences
            FROM combined_results_cte
            GROUP BY
                distance,
                text_id,
                date,
                title
            ORDER BY distance ASC
            LIMIT 10
        '''
    ).fetch_arrow_table().to_pandas()

    return dense_search_result_dataframe


def fupi_colbert_centroids_searcher(
    sentence_level_table: lancedb.table.Table,
    query_colbert_embeddings: np.ndarray
) -> pd.DataFrame:
    query_colbert_centroid = np.average(query_colbert_embeddings, axis=0).tolist()

    colbert_initial_dataframe = sentence_level_table.search(
        query_colbert_centroid,
        vector_column_name='colbert_embedding'
    )\
    .select(
        [
            'text_id',
            'sentence_id',
            'sentence'
        ]
    )\
    .limit(10)\
    .to_pandas()

    colbert_search_result_dataframe = duckdb.query(
        f'''
            WITH
                combined_results_cte AS (
                    SELECT
                        cid._distance AS distance,
                        cid.text_id,
                        tlat.date,
                        tlat.title,
                        cid.sentence_id,
                        cid.sentence
                    FROM colbert_initial_dataframe cid
                        JOIN text_level_arrow_table tlat ON
                            tlat.text_id = cid.text_id
                    WHERE LENGTH(cid.sentence) > 3
                    ORDER BY
                        cid.text_id ASC,
                        cid.sentence_id ASC
                )

            -- User-facing query:
            SELECT
                distance,
                text_id,
                date,
                title,
                string_agg(sentence_id, ', ') AS sentence_ids,
                string_agg(sentence, ' -- ') AS sentences
            FROM combined_results_cte
            GROUP BY
                distance,
                text_id,
                date,
                title
            ORDER BY distance ASC
            LIMIT 10
        '''
    ).fetch_arrow_table().to_pandas()

    return colbert_search_result_dataframe
