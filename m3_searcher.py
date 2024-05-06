#!/usr/bin/env python3

import os
from multiprocessing import cpu_count

from dotenv import load_dotenv, find_dotenv
import duckdb
import gradio as gr
import lancedb
from minio import Minio
import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 onnx_runner python /app/m3_searcher.py
# http://0.0.0.0:7860/?__theme=dark

# Load settings from .env file:
load_dotenv(find_dotenv())

# Globals:
onnx_runtime_session = None
tokenizer = None

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

    query_dense_embedding    = query_embedded[0][0]
    query_colbert_embeddings = query_embedded[1][0]

    search_result = None

    # Dense vector search:
    if (search_type == 'Sentence Dense Vector'):
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
                ORDER BY distance DESC
                LIMIT 10
            '''
        ).fetch_arrow_table().to_pandas()

        search_result = dense_final_dataframe.to_dict('records')

    # ColBERT vector search:
    if (search_type == 'Sentence ColBERT Centroid'):
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
                ORDER BY distance DESC
                LIMIT 10
            '''
        ).fetch_arrow_table().to_pandas()

        search_result = colbert_final_dataframe.to_dict('records')

    # Weighted Tokens search:
    if (search_type == 'Weighted Tokens'):
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
                            COUNT(token) AS matching_tokens_number,
                            SUM(weight) AS sentence_weight
                        FROM weighted_tokens_arrow_table
                        WHERE token IN ({filtered_token_string})
                        GROUP BY
                            text_id,
                            sentence_id
                        HAVING matching_tokens_number >= {len(filtered_token_list)}
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

    # ColBERT and Tokens search:
    if (search_type == 'CoaT - ColBERT and Tokens'):
        result_dataframes_list = []

        for query_colbert_embedding in query_colbert_embeddings:
            colbert_and_tokens_initial_dataframe = colbert_tokens_table.search(
                query_colbert_embedding,
                vector_column_name='colbert_embedding'
            )\
            .select(
                [
                    'text_id',
                    'token'
                ]
            )\
            .limit(5)\
            .to_pandas()

            text_id_unique_list = (
                colbert_and_tokens_initial_dataframe['text_id'].unique().tolist()
            )
            token_unique_list = (
                colbert_and_tokens_initial_dataframe['token'].unique().tolist()
            )

            text_id_string = ', '.join(map(str, text_id_unique_list))
            token_string = ', '.join(map(str, token_unique_list))

            colbert_and_tokens_result_dataframe = duckdb.query(
                f'''
                    WITH
                        -- Sentence-Level Results Common Table Expression:
                        sentence_results_cte AS (
                            SELECT
                                text_id,
                                sentence_id,
                                SUM(weight) AS sentence_weight
                            FROM weighted_tokens_arrow_table
                            WHERE
                                text_id IN ({text_id_string})
                                AND token IN ({token_string})
                            GROUP BY
                                text_id,
                                sentence_id
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

            result_dataframes_list.append(colbert_and_tokens_result_dataframe)

        combined_dataframe = None

        for result_dataframe in result_dataframes_list:
            # Only for the first result_dataframe -
            # at this point the combined_dataframe is still empty:
            if combined_dataframe is None:
                combined_dataframe = result_dataframe

            # For any other dataframe after
            # the first result_dataframe:
            if combined_dataframe is None:
                combined_dataframe = pd.concat(
                    [
                        combined_dataframe,
                        result_dataframe
                    ],
                    ignore_index=True
                )
    
                combined_dataframe.reset_index()

        combined_dataframe.sort_values(
            by=['text_weight'],
            ascending=False,
            inplace=True
        )

        search_result = combined_dataframe.to_dict('records')

    return search_result


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
    input_box=gr.Textbox(lines=1, label='Search Request')

    search_type = gr.Radio(
        [
            'Sentence Dense Vector',
            'Sentence ColBERT Centroid',
            'Weighted Tokens',
            'CoaT - ColBERT and Tokens'
        ],
        value='Sentence Dense Vector',
        label='Search Base',
    )

    output_box=gr.JSON(label='Search Results', show_label=True)

    global gradio_interface
    gradio_interface = gr.Blocks(theme=gr.themes.Glass(), title='Fupi')

    with gradio_interface:
        with gr.Row():
            gr.Markdown(
                """
                # Fupi
                ## Serverless multilingual semantic search
                """
            )

        with gr.Row():
            with gr.Column(scale=30):
                gr.Markdown(
                    """
                    **License:** Apache License 2.0.  
                    **Repository:** https://github.com/ddmitov/fupi  
                    """
                )

            with gr.Column(scale=40):
                gr.Markdown(
                    """
                    **Dataset:** Common Crawl News - 2021 Bulgarian  
                    https://commoncrawl.org/blog/news-dataset-available  
                    https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual  
                    """
                )

            with gr.Column(scale=30):
                gr.Markdown(
                    """
                    **Model:** BGE-M3  
                    https://huggingface.co/BAAI/bge-m3  
                    https://huggingface.co/aapot/bge-m3-onnx  
                    """
                )

        with gr.Row():
                search_type.render()

        with gr.Row():
            input_box.render()

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
                inputs=input_box,
                outputs=output_box,
                cache_examples=False
            )

        with gr.Row():
            search_button = gr.Button("Search")

            gr.ClearButton([input_box, output_box])

        with gr.Row():
            output_box.render()

        gr.on(
            triggers=[
                input_box.submit,
                search_button.click
            ],
            fn=lancedb_searcher,
            inputs=[
                input_box,
                search_type
            ],
            outputs=output_box,
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