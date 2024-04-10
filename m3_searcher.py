#!/usr/bin/env python3

import os
import multiprocessing

import duckdb
import gradio as gr
import lancedb
from minio import Minio
import numpy as np
import onnxruntime as ort
from sentence_transformers.quantization import quantize_embeddings
from torch import FloatTensor
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 onnx_runner python /app/m3_searcher.py

# MinIO local object storage settings:
S3_ENDPOINT = '172.17.0.2:9000'

os.environ['AWS_ENDPOINT'] = f'http://{S3_ENDPOINT}'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# LanceDB settings:
LANCEDB_BUCKET_NAME = 'bge-m3'


def centroid_maker(embeddings_list: list[list]) -> list:
    average_embedding_list = np.average(embeddings_list, axis=0).tolist()

    return average_embedding_list


def lancedb_searcher(search_request: str, search_type: str)-> object:
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

    # Define LanceDB tables:
    lance_db = lancedb.connect(f's3://{LANCEDB_BUCKET_NAME}/')

    dense_table = lance_db.open_table('dense')
    colbert_table = lance_db.open_table('colbert')

    # Vectorize the search request:
    query_tokenized_input = tokenizer(
        search_request,
        truncation=True,
        return_tensors='np'
    )

    query_onnx_input = {
        key: ort.OrtValue.ortvalue_from_numpy(value)
        for key, value in query_tokenized_input.items()
    }

    query_outputs = ort_session.run(None, query_onnx_input)

    query_dense_embedding = query_outputs[0][0]
    query_colbert_embeddings = list(query_outputs[1][0])

    # Perform LanceDB search:
    search_result = None

    if (search_type == 'Dense Vectors'):
        dense_result = dense_table.search(
            query_dense_embedding.tolist(),
            vector_column_name='dense_embedding'
        )\
        .select(['text_id', 'date', 'title', 'text'])\
        .limit(5)\
        .to_pandas()

        search_result = dense_result.to_dict('records')

    if (search_type == 'Dense Binary Vectors'):
        query_dense_binary_embedding = quantize_embeddings(
            FloatTensor(query_dense_embedding).reshape(1, -1),
            precision='binary'
        ).tolist()[0]

        dense_binary_result = dense_table.search(
            query_dense_binary_embedding,
            vector_column_name='dense_embedding_binary'
        )\
        .metric('dot')\
        .select(['text_id', 'date', 'title', 'text'])\
        .limit(5)\
        .to_pandas()

        search_result = dense_binary_result.to_dict('records')

    if (search_type == 'ColBERT Vectors'):
        query_colbert_centroid = centroid_maker(query_colbert_embeddings)

        colbert_result_dataframe = colbert_table.search(
            query_colbert_centroid,
            vector_column_name='colbert_embedding'
        )\
        .select(['text_id'])\
        .limit(5)\
        .to_pandas()

        text_id_list = colbert_result_dataframe['text_id'].unique().tolist()

        text_id_string = (
            '\'' +
            '\', \''.join(map(str, text_id_list)) +
            '\''
        )

        dense_arrow_table = dense_table.to_lance()

        colbert_final_result_dataframe = duckdb.query(
            f'''
                SELECT
                    text_id,
                    date,
                    title,
                    text
                FROM dense_arrow_table
                WHERE text_id IN ({text_id_string})
            '''
        ).fetch_arrow_table().to_pandas()

        search_result = colbert_final_result_dataframe.to_dict('records')

    return search_result


def s3_downloader(bucket_name, bucket_prefix):
    client = Minio(
        S3_ENDPOINT,
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

    s3_downloader(LANCEDB_BUCKET_NAME, 'model')

    print('')
    print('The embedding model is downloaded.')
    print('')

    input_box=gr.Textbox(lines=1, label='Search Request')

    search_type = gr.Radio(
        [
            'Dense Vectors',
            'Dense Binary Vectors',
            'ColBERT Vectors'
        ],
        value='Dense Vectors',
        label="Search Type",
    )

    output_box=gr.JSON(label='Search Results', show_label=True)

    interface = gr.Blocks(
        theme=gr.themes.Glass(),
        title='Fupi'
    )

    with interface:
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
                    **Dataset:** Common Crawl News  
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
                    https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx  
                    """
                )

        with gr.Row():
                search_type.render()

        with gr.Row():
            input_box.render()

        with gr.Row():
            gr.Examples(
                [
                    ['economic growth'],
                    ['environmental impact']
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

        search_button.click(
            lancedb_searcher,
            inputs=[
                input_box,
                search_type
            ],
            outputs=output_box,
            show_progress = 'full'
        )

    interface.launch(
        server_name='0.0.0.0',
        # server_port=8080,
        # share=True,
        show_api=False,
        inline=False,
        inbrowser=False
    )


if __name__ == '__main__':
    main()
