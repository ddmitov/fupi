#!/usr/bin/env python3

import os
from multiprocessing import cpu_count
import time

from dotenv import load_dotenv, find_dotenv
import duckdb
import gradio as gr
import lancedb
from minio import Minio
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 onnx_runner python /app/m3_searcher.py
# http://0.0.0.0:7860/?__theme=dark

# Load settings from .env file:
load_dotenv(find_dotenv())

# Globals:
onnx_runtime_session = None
tokenizer            = None

sentence_level_table  = None
text_level_table      = None
weighted_tokens_table = None
colbert_tokens_table  = None

sentence_level_arrow_table  = None
text_level_arrow_table      = None
weighted_tokens_arrow_table = None

gradio_interface = None


def centroid_maker(embeddings_list: np.ndarray) -> list:
    average_embedding_list = np.average(embeddings_list, axis=0).tolist()

    return average_embedding_list


def lancedb_searcher(search_request: str, search_type: str)-> object:
    # Use the already initialized ONNX runtime and tokenizer:
    global onnx_runtime_session
    global tokenizer

    # LanceDB tables:
    global sentence_level_table
    global text_level_table
    global weighted_tokens_table
    global colbert_tokens_table

    # LanceDB tables exposed as Arrow tables:
    global sentence_level_arrow_table
    global text_level_arrow_table
    global weighted_tokens_arrow_table

    # Start measuring tokenization and embedding time:
    embedding_start_time = time.time()

    # Tokenize the search request:
    query_tokenized = tokenizer(
        search_request,
        truncation=True,
        return_tensors='np'
    )

    # Vectorize the tokenized search request:
    query_onnx_runtime_input = {
        key: ort.OrtValue.ortvalue_from_numpy(value)
        for key, value in query_tokenized.items()
    }

    query_embedded = onnx_runtime_session.run(None, query_onnx_runtime_input)

    # Stop measuring tokenization and embedding time:
    embedding_time = time.time() - embedding_start_time

    # Get query embeddings:
    query_dense_embedding    = query_embedded[0][0]
    query_colbert_embeddings = query_embedded[1][0]

    search_result = None

    # Dense vectors search:
    if (search_type == 'Sentence Dense Vectors'):
        # Start measuring search time:
        search_start_time = time.time()

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

        dense_final_dataframe = duckdb.query(
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

        search_result = dense_final_dataframe.to_dict('records')

        # Stop measuring search time:
        search_time = time.time() - search_start_time 

        search_info = {
            'Embedding Time': f'{embedding_time:.3f} s',
            'Search Time':    f'{search_time:.3f} s',
            'Total Time':     f'{embedding_time + search_time:.3f} s',
        }

    # ColBERT Centroids search:
    if (search_type == 'Sentence ColBERT Centroids'):
        # Start measuring search time:
        search_start_time = time.time()

        query_colbert_centroid = centroid_maker(query_colbert_embeddings)

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

        colbert_final_dataframe = duckdb.query(
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

        search_result = colbert_final_dataframe.to_dict('records')

        # Stop measuring search time:
        search_time = time.time() - search_start_time 

        search_info = {
            'Embedding Time': f'{embedding_time:.3f} s',
            'Search Time':    f'{search_time:.3f} s',
            'Total Time':     f'{embedding_time + search_time:.3f} s',
        }

    # Weighted Tokens search:
    if (search_type == 'Weighted Tokens'):
        # Start measuring search time:
        search_start_time = time.time()

        token_list = list(query_tokenized['input_ids'].tolist())[0]

        unused_tokens = set(
            [
                tokenizer.cls_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
            ]
        )

        filtered_token_list = [
            token for token in token_list if token not in unused_tokens
        ]

        filtered_token_string = ', '.join(map(str, filtered_token_list))

        weighted_tokens_result_dataframe = duckdb.query(
            f'''
                WITH
                    -- Sentence-Level Results Common Table Expression:
                    sentence_results_cte AS (
                        SELECT
                            text_id,
                            sentence_id,
                            COUNT(token) AS matching_tokens_count,
                            SUM(weight) AS sentence_weight
                        FROM weighted_tokens_arrow_table
                        WHERE token IN ({filtered_token_string})
                        GROUP BY
                            text_id,
                            sentence_id
                        HAVING matching_tokens_count >= {len(filtered_token_list)}
                    ),

                    -- Text-Level Results Common Table Expression:
                    text_results_cte AS (
                        SELECT
                            text_id,
                            SUM(sentence_weight) AS text_weight
                        FROM sentence_results_cte
                        GROUP BY text_id
                    ),

                    -- Combined Results Common Table Expression:
                    combined_results_cte AS (
                        SELECT
                            CAST(tr_cte.text_weight AS INT) AS text_weight,
                            tlat.text_id,
                            tlat.date,
                            tlat.title,
                            sr_cte.sentence_id,
                            slat.sentence
                        FROM
                            sentence_results_cte sr_cte
                            JOIN text_results_cte tr_cte ON
                                tr_cte.text_id = sr_cte.text_id
                            JOIN sentence_level_arrow_table slat ON
                                slat.text_id = sr_cte.text_id
                                AND slat.sentence_id = sr_cte.sentence_id
                            JOIN text_level_arrow_table tlat ON
                                tlat.text_id = sr_cte.text_id
                        WHERE LENGTH(slat.sentence) > 3
                        ORDER BY
                            sr_cte.text_id ASC,
                            sr_cte.sentence_id ASC
                    )

                -- User-facing query:
                SELECT
                    text_weight,
                    text_id,
                    date,
                    title,
                    string_agg(sentence_id, ', ') AS sentence_ids,
                    string_agg(sentence, ' -- ') AS sentences
                FROM combined_results_cte
                GROUP BY
                    text_weight,
                    text_id,
                    date,
                    title
                ORDER BY text_weight DESC
                LIMIT 10
            '''
        ).fetch_arrow_table().to_pandas()

        search_result = weighted_tokens_result_dataframe.to_dict('records')

        # Stop measuring search time:
        search_time = time.time() - search_start_time

        search_info = {
            'Embedding Time': f'{embedding_time:.3f} s',
            'Search Time':    f'{search_time:.3f} s',
            'Total Time':     f'{embedding_time + search_time:.3f} s',
        }

    # ColBERT and Tokens search:
    if (search_type == 'CoaT - ColBERT and Tokens'):
        # Start measuring ColBERT search time:
        colbert_search_start_time = time.time()

        text_id_combined_list = []
        token_combined_list   = []

        for query_colbert_embedding in query_colbert_embeddings:
            colbert_dataframe = colbert_tokens_table.search(
                query_colbert_embedding,
                vector_column_name='colbert_embedding'
            )\
            .select(
                [
                    'text_id',
                    'token'
                ]
            )\
            .limit(3)\
            .to_pandas()

            text_id_unique_list = (colbert_dataframe['text_id'].unique().tolist())
            text_id_combined_list.extend(text_id_unique_list)

            token_unique_list = (colbert_dataframe['token'].unique().tolist())
            token_combined_list.extend(token_unique_list)

        # Stop measuring ColBERT search time:
        colbert_search_time = time.time() - colbert_search_start_time

        # Start measuring token search time:
        token_search_start_time = time.time()

        text_id_string = ', '.join(map(str, text_id_combined_list))
        token_string   = ', '.join(map(str, token_combined_list))

        # Define minimal percentage of matchin tokens in every sentence:
        minimum_matching_tokens = (
            int((50 * len(token_combined_list)) / 100.0)
        )

        colbert_and_tokens_result_dataframe = duckdb.query(
            f'''
                WITH
                    -- Sentence-Level Results Common Table Expression:
                    sentence_results_cte AS (
                        SELECT
                            text_id,
                            sentence_id,
                            COUNT(token) AS matching_tokens_count,
                            SUM(weight) AS sentence_weight
                        FROM weighted_tokens_arrow_table
                        WHERE
                            text_id IN ({text_id_string})
                            AND token IN ({token_string})
                        GROUP BY
                            text_id,
                            sentence_id
                        HAVING matching_tokens_count >= {minimum_matching_tokens}
                    ),

                    -- Text-Level Results Common Table Expression:
                    text_results_cte AS (
                        SELECT
                            text_id,
                            SUM(sentence_weight) AS text_weight
                        FROM sentence_results_cte
                        GROUP BY text_id
                    ),

                    -- Combined Results Common Table Expression:
                    combined_results_cte AS (
                        SELECT
                            CAST(tr_cte.text_weight AS INT) AS text_weight,
                            tlat.text_id,
                            tlat.date,
                            tlat.title,
                            sr_cte.sentence_id,
                            slat.sentence
                        FROM
                            sentence_results_cte sr_cte
                            JOIN text_results_cte tr_cte ON
                                tr_cte.text_id = sr_cte.text_id
                            JOIN sentence_level_arrow_table slat ON
                                slat.text_id = sr_cte.text_id
                                AND slat.sentence_id = sr_cte.sentence_id
                            JOIN text_level_arrow_table tlat ON
                                tlat.text_id = sr_cte.text_id
                        WHERE LENGTH(slat.sentence) > 3
                        ORDER BY
                            sr_cte.text_id ASC,
                            sr_cte.sentence_id ASC
                    )

                -- User-facing query:
                SELECT
                    text_weight,
                    text_id,
                    date,
                    title,
                    string_agg(sentence_id, ', ') AS sentence_ids,
                    string_agg(sentence, ' -- ') AS sentences
                FROM combined_results_cte
                GROUP BY
                    text_weight,
                    text_id,
                    date,
                    title
                ORDER BY text_weight DESC
                LIMIT 10
            '''
        ).fetch_arrow_table().to_pandas()

        search_result = colbert_and_tokens_result_dataframe.to_dict('records')

        # Stop measuring token search time:
        token_search_time = time.time() - token_search_start_time

        total_time = (
            embedding_time +
            colbert_search_time +
            token_search_time
        )

        search_info = {
            'Embedding Time':      f'{embedding_time:.3f} s',
            'ColBERT Search Time': f'{colbert_search_time:.3f} s',
            'Token Search Time':   f'{token_search_time:.3f} s',
            'Total Time':          f'{total_time:.3f} s',
        }

    return search_info, search_result


def s3_downloader(bucket_name, bucket_prefix):
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

    return True


def main():
    print('')
    print('Downloading the embedding model from object storage ...')
    print('')

    s3_downloader(os.environ['LANCEDB_BUCKET_NAME'], 'model')

    print('')
    print('The embedding model is downloaded.')
    print('')

    # MinIO local object storage settings:
    os.environ['AWS_ENDPOINT'] = f'http://{os.environ['S3_ENDPOINT']}'
    os.environ['ALLOW_HTTP'] = 'True'

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

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
    global onnx_runtime_session
    onnx_runtime_session = ort.InferenceSession(
        '/tmp/model/model.onnx',
        sess_ptions=onnxrt_options,
        providers=['CPUExecutionProvider']
    )

    # Initialize tokenizer:
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/tmp/model/'
    )

    # Define LanceDB tables:
    lance_db = lancedb.connect(f's3://{os.environ['LANCEDB_BUCKET_NAME']}/')

    global sentence_level_table
    global text_level_table
    global weighted_tokens_table
    global colbert_tokens_table

    sentence_level_table  = lance_db.open_table('sentence-level')
    text_level_table      = lance_db.open_table('text-level')
    weighted_tokens_table = lance_db.open_table('weighted-tokens')
    colbert_tokens_table  = lance_db.open_table('colbert-tokens')

    global sentence_level_arrow_table
    global text_level_arrow_table
    global weighted_tokens_arrow_table

    sentence_level_arrow_table  = sentence_level_table.to_lance()
    text_level_arrow_table      = text_level_table.to_lance()
    weighted_tokens_arrow_table = weighted_tokens_table.to_lance()

    # Define Gradio user interface:
    search_request_box=gr.Textbox(lines=1, label='Search Request')

    search_type = gr.Radio(
        [
            'Sentence Dense Vectors',
            'Sentence ColBERT Centroids',
            'Weighted Tokens',
            'CoaT - ColBERT and Tokens'
        ],
        value='Sentence Dense Vectors',
        label='Search Base',
    )

    search_info_box=gr.JSON(label='Search Info', show_label=True)

    search_results_box=gr.JSON(label='Search Results', show_label=True)

    global gradio_interface
    gradio_interface = gr.Blocks(theme=gr.themes.Glass(), title='Fupi')

    with gradio_interface:
        with gr.Row():
            gr.Markdown(
                '''
                # Fupi
                ## Serverless multilingual semantic search
                '''
            )

        with gr.Row():
            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    **License:** Apache License 2.0.  
                    **Repository:** https://github.com/ddmitov/fupi  
                    '''
                )

            with gr.Column(scale=40):
                gr.Markdown(
                    '''
                    **Dataset:** Common Crawl News - 2021 Bulgarian  
                    https://commoncrawl.org/blog/news-dataset-available  
                    https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual  
                    '''
                )

            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    **Model:** BGE-M3  
                    https://huggingface.co/BAAI/bge-m3  
                    https://huggingface.co/aapot/bge-m3-onnx  
                    '''
                )

        with gr.Row():
                search_type.render()

        with gr.Row():
            search_request_box.render()

        with gr.Row():
            gr.Examples(
                [
                    'COVID',
                    'pandemic recovery',
                    'European union',
                    'Plovdiv',
                    'local budget',
                    'environmental impact',
                ],
                fn=lancedb_searcher,
                inputs=search_request_box,
                outputs=search_results_box,
                cache_examples=False
            )

        with gr.Row():
            search_button = gr.Button('Search')

            gr.ClearButton(
                [
                    search_info_box,
                    search_request_box,
                    search_results_box
                ]
            )

        with gr.Row():
            search_info_box.render()

        with gr.Row():
            search_results_box.render()

        gr.on(
            triggers=[
                search_request_box.submit,
                search_button.click
            ],
            fn=lancedb_searcher,
            inputs=[
                search_request_box,
                search_type
            ],
            outputs=[
                search_info_box,
                search_results_box
            ],
        )

    gradio_interface.launch(
        server_name='0.0.0.0',
        # server_port=8080,
        share=False,
        show_api=False,
        inline=False,
        inbrowser=False
    )


if __name__ == '__main__':
    main()
