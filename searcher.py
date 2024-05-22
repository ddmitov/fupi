#!/usr/bin/env python3

import datetime
import os
import resource
import signal
import time
import threading

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
import gradio as gr
import lancedb
import onnxruntime as ort
from transformers import AutoTokenizer
import uvicorn

from fupi import model_downloader_from_object_storage
from fupi import fupi_dense_vectors_searcher
from fupi import fupi_colbert_centroids_searcher

# Start the application for local development at http://0.0.0.0:7860/ using:
# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 fupi python /app/searcher.py

# Global variables:
onnx_runtime_session = None
tokenizer            = None

sentence_level_table = None
text_level_table     = None

last_activity = None


def lancedb_searcher(search_request: str, search_type: str)-> object:
    # Update last activity date and time:
    global last_activity
    last_activity = time.time()

    # Use the already initialized ONNX runtime and tokenizer:
    global onnx_runtime_session
    global tokenizer

    # If the ONNX runtime session or the tokenizer are still not initialized,
    # wait for one second and try again:
    if onnx_runtime_session is None or tokenizer is None:
        time.sleep(1)
        lancedb_searcher(search_request, search_type)

    # LanceDB tables:
    global sentence_level_table
    global text_level_table

    # LanceDB tables exposed as Arrow tables:
    sentence_level_arrow_table = sentence_level_table.to_lance()
    text_level_arrow_table     = text_level_table.to_lance()

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

    search_result = None

    # Dense vectors search:
    if (search_type == 'Sentence Dense Vectors'):
        search_start_time = time.time()

        query_dense_embedding    = query_embedded[0][0]

        search_result = fupi_dense_vectors_searcher(
            sentence_level_table,
            query_dense_embedding
        )

        search_time = time.time() - search_start_time 

        search_info = {
            'Embedding Time': f'{embedding_time:.3f} s',
            'Search Time':    f'{search_time:.3f} s',
            'Total Time':     f'{embedding_time + search_time:.3f} s',
        }

    # ColBERT Centroids search:
    if (search_type == 'Sentence ColBERT Centroids'):
        search_start_time = time.time()

        query_colbert_embeddings = query_embedded[1][0]

        search_result = fupi_colbert_centroids_searcher(
            sentence_level_table,
            query_colbert_embeddings
        )

        search_time = time.time() - search_start_time 

        search_info = {
            'Embedding Time': f'{embedding_time:.3f} s',
            'Search Time':    f'{search_time:.3f} s',
            'Total Time':     f'{embedding_time + search_time:.3f} s',
        }

    memory_usage_megabytes = round(
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        2
    )

    memory_usage_gigabytes = round((memory_usage_megabytes / 1024), 2)

    print(f'Maximal RSS memory usage: {memory_usage_gigabytes} GB')

    return search_info, search_result


def activity_inspector():
    global last_activity

    thread = threading.Timer(
        int(os.environ['INACTIVITY_CHECK_SECONDS']),
        activity_inspector
    )

    thread.daemon = True

    thread.start()

    if (time.time() - last_activity) > int(os.environ['INACTIVITY_MAXIMUM_SECONDS']):
        print(f'Initiating shutdown sequence at: {datetime.datetime.now()}')

        os.kill(os.getpid(), signal.SIGINT)

    return True


def embedding_model_threaded_starter(
    models_bucket_name: str,
    models_bucket_prefix: str
):
    # Download embedding model and tokenizer from object storage:
    model_downloader_from_object_storage(
        models_bucket_name,
        models_bucket_prefix
    )

    # Initialize ONNX runtime session:
    global onnx_runtime_session

    onnx_runtime_session = ort.InferenceSession(
        '/tmp/bge-m3/model.onnx',
        providers=['CPUExecutionProvider']
    )

    # Initialize tokenizer:
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        '/tmp/bge-m3/'
    )

    return True


def main():
    # Load object storage settings from .env file:
    load_dotenv(find_dotenv())

    lancedb_bucket_name = None
    models_bucket_name = None

    # Object storage settings for Fly.io - Tigris deployment:
    if os.environ.get('FLY_APP_NAME') is not None:
        os.environ['AWS_ENDPOINT'] = os.environ['TIGRIS_ENDPOINT_S3']
        os.environ['AWS_ACCESS_KEY_ID'] = os.environ['TIGRIS_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['TIGRIS_SECRET_ACCESS_KEY']
        os.environ['AWS_REGION'] = 'auto'

        lancedb_bucket_name = os.environ['TIGRIS_LANCEDB_BUCKET']
        models_bucket_name = os.environ['TIGRIS_MODELS_BUCKET']
    # Object storage settings for local development:
    else:
        os.environ['AWS_ENDPOINT'] = os.environ['MINIO_ENDPOINT_S3']
        os.environ['AWS_ACCESS_KEY_ID'] = os.environ['MINIO_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['MINIO_SECRET_ACCESS_KEY']
        os.environ['AWS_REGION'] = 'us-east-1'

        os.environ['ALLOW_HTTP'] = 'True'

        lancedb_bucket_name = os.environ['MINIO_LANCEDB_BUCKET']
        models_bucket_name = os.environ['MINIO_MODELS_BUCKET']

    # The embedding model and the tokenizer are downloaded and initialized
    # in a separate thread to minimize the startup time of the application:
    embedding_model_starter_thread = threading.Thread(
        target=embedding_model_threaded_starter,
        args=[
            models_bucket_name,
            'bge-m3'
        ]
    )

    embedding_model_starter_thread.start()

    # Define LanceDB tables:
    lance_db = lancedb.connect(f's3://{lancedb_bucket_name}/')

    global sentence_level_table
    global text_level_table

    sentence_level_table  = lance_db.open_table('sentence-level')
    text_level_table      = lance_db.open_table('text-level')

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    # Define Gradio user interface:
    search_request_box=gr.Textbox(lines=1, label='Search Request')

    search_type = gr.Radio(
        [
            'Sentence ColBERT Centroids',
            'Sentence Dense Vectors'
        ],
        value='Sentence ColBERT Centroids',
        label='Search Type',
    )

    search_info_box=gr.JSON(label='Search Info', show_label=True)

    search_results_box=gr.JSON(label='Search Results', show_label=True)

    # Dark theme by default:
    js_function = """
        function refresh() {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
    """

    gradio_interface = gr.Blocks(
        theme=gr.themes.Glass(),
        js=js_function,
        title='Fupi'
    )

    with gradio_interface:
        with gr.Row():
            gr.Markdown(
                '''
                # Fupi Demo
                ## Multilingual Semantic Search
                '''
            )

        with gr.Row():
            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    **License:** Apache License 2.0.  
                    **https://github.com/ddmitov/fupi**  
                    **https://fupi.fly.dev**  
                    '''
                )

            with gr.Column(scale=40):
                gr.Markdown(
                    '''
                    **Dataset:** Common Crawl News - 2021 Bulgarian  
                    **https://commoncrawl.org/blog/news-dataset-available**  
                    **https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual**  
                    '''
                )

            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    **Model:** BGE-M3  
                    **https://huggingface.co/BAAI/bge-m3**  
                    **https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx**  
                    '''
                )

        with gr.Row():
                search_type.render()

        with gr.Row():
            search_request_box.render()

        with gr.Row():
            gr.Examples(
                [
                    'COVID vaccines',
                    'pandemic recovery',
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

    gradio_interface.show_api = False
    gradio_interface.queue()

    if os.environ.get('FLY_APP_NAME') is not None:
        gradio_interface.root_path = 'https://fupi.fly.dev'

    fastapi_app = FastAPI()

    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        gradio_interface,
        path='/'
    )

    # Update last activity date and time:
    global last_activity
    last_activity = time.time()

    # Start activity inspector in a separate thread
    # to implement scale-to-zero capability, i.e.
    # when there is no user activity for a predefined amount of time
    # the application will shut down.
    activity_inspector()

    try:
        uvicorn.run(
            fastapi_app,
            host='0.0.0.0',
            port=7860
        )
    except (KeyboardInterrupt, SystemExit):
        print('\n')

        exit(0)


if __name__ == '__main__':
    main()
