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
from hf_hub_ctranslate2 import MultiLingualTranslatorCT2fromHfHub
import lancedb
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer
import uvicorn

from fupi import model_downloader_from_object_storage
from fupi import fupi_dense_vectors_searcher
from fupi import fupi_colbert_centroids_searcher

# Start the application for local development at http://0.0.0.0:7860/ using:
# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 fupi python /app/searcher.py

# Global variables:
models_bucket_name   = None

onnx_runtime_session = None
embedding_tokenizer  = None

translation_model    = None

sentence_level_table = None
text_level_table     = None

last_activity        = None


def search_result_translator(search_result_dataframe: pd.DataFrame) -> dict:
        global translation_model

        title_list    = search_result_dataframe['title'].to_list()
        sentence_list = search_result_dataframe['sentences'].to_list()

        translated_title_list = translation_model.generate(
            title_list,
            src_lang=['bg'] * len(title_list),
            tgt_lang=['en'] * len(title_list)
        )

        translated_sentences_list = translation_model.generate(
            sentence_list,
            src_lang=['bg'] * len(sentence_list),
            tgt_lang=['en'] * len(sentence_list)
        )

        search_result_dataframe['translated_title']     = translated_title_list
        search_result_dataframe['translated_sentences'] = translated_sentences_list

        search_result_dataframe = search_result_dataframe[
            list(
                (
                    'distance',
                    'text_id',
                    'date',
                    'title',
                    'translated_title',
                    'sentence_ids',
                    'sentences',
                    'translated_sentences'
                )
            )
        ]

        search_result = search_result_dataframe.to_dict('records')

        return search_result


def translation_model_starter() -> True:
    # Download the translation model if necessary:
    translation_model_filelist = [
        '/tmp/ct2fast-m2m100_418m/added_tokens.json',
        '/tmp/ct2fast-m2m100_418m/config.json',
        '/tmp/ct2fast-m2m100_418m/generation_config.json',
        '/tmp/ct2fast-m2m100_418m/model.bin',
        '/tmp/ct2fast-m2m100_418m/sentencepiece.bpe.model',
        '/tmp/ct2fast-m2m100_418m/shared_vocabulary.txt',
        '/tmp/ct2fast-m2m100_418m/special_tokens_map.json',
        '/tmp/ct2fast-m2m100_418m/tokenizer_config.json',
        '/tmp/ct2fast-m2m100_418m/vocab.json'
    ]

    while True:
        inspected_filelist = []

        for file in translation_model_filelist:
            inspected_filelist.append(os.path.isfile(file))

        if all(inspected_filelist):
            break
        else:
            gr.Info('Loading the translation model ...')

            global models_bucket_name

            model_downloader_from_object_storage(
                models_bucket_name,
                'ct2fast-m2m100_418m'
            )

    # Initialize translation model:
    global translation_model

    # Initialize translation model using GPU:
    # translation_model = MultiLingualTranslatorCT2fromHfHub(
    #     model_name_or_path='/tmp/ct2fast-m2m100_418m',
    #     device='cuda',
    #     compute_type='float32',
    #     tokenizer=AutoTokenizer.from_pretrained('/tmp/ct2fast-m2m100_418m')
    # )

    # Initialize translation model using CPU:
    translation_model = MultiLingualTranslatorCT2fromHfHub(
        model_name_or_path='/tmp/ct2fast-m2m100_418m',
        device='cpu',
        compute_type='int8',
        tokenizer=AutoTokenizer.from_pretrained('/tmp/ct2fast-m2m100_418m')
    )

    return True


def embedding_model_starter() -> True:
    # Download the embedding model if necessary:
    embedding_model_filelist = [
        '/tmp/bge-m3/model.onnx',
        '/tmp/bge-m3/model.onnx_data',
        '/tmp/bge-m3/special_tokens_map.json',
        '/tmp/bge-m3/tokenizer_config.json',
        '/tmp/bge-m3/tokenizer.json'
    ]

    while True:
        inspected_filelist = []

        for file_name in embedding_model_filelist:
            inspected_filelist.append(os.path.isfile(file_name))

        if all(inspected_filelist):
            break
        else:
            gr.Info('Loading the embedding model ...')

            global models_bucket_name

            model_downloader_from_object_storage(
                models_bucket_name,
                'bge-m3'
            )

    # Initialize ONNX runtime session:
    global onnx_runtime_session

    onnx_runtime_session = ort.InferenceSession(
        '/tmp/bge-m3/model.onnx',
        providers=['CPUExecutionProvider']
    )

    # Initialize the embedding tokenizer:
    global embedding_tokenizer

    embedding_tokenizer = AutoTokenizer.from_pretrained('/tmp/bge-m3/')

    return True


def lancedb_searcher(
    search_request: str,
    search_type: str
)-> tuple[dict, dict]:
    # Update last activity date and time:
    global last_activity
    last_activity = time.time()

    if len(search_request) == 0:
        message = 'Please, enter a search request or use one of the examples!'

        gr.Info(message)
        message_dict = {'message': message}

        return message_dict, message_dict

    # Use the global ONNX runtime and embedding tokenizer:
    global onnx_runtime_session
    global embedding_tokenizer

    # Use the global translation model:
    global translation_model

    # Use the global LanceDB tables:
    global sentence_level_table
    global text_level_table

    # LanceDB tables exposed as Arrow tables:
    sentence_level_arrow_table = sentence_level_table.to_lance()
    text_level_arrow_table     = text_level_table.to_lance()

    # Check if all machine learning models are ready for work:
    embedding_model_time   = 0
    translation_model_time = 0

    initializatin_info = {}

    # Initiate the embedding model if necessary:
    if onnx_runtime_session is None or embedding_tokenizer is None:
        embedding_model_start_time = time.time()
        embedding_model_starter()
        embedding_model_time = time.time() - embedding_model_start_time

        embedding_model_time_string = (
            str(f'{embedding_model_time:.3f}').zfill(6)
        )

        initializatin_info['Embedding Model Load Time ---- '] = (
            f'{embedding_model_time_string} s'
        )

    # Initiate the translation model if necessary:
    if translation_model is None:
        translation_model_start_time = time.time()
        translation_model_starter()
        translation_model_time = time.time() - translation_model_start_time

        translation_model_time_string = (
            str(f'{translation_model_time:.3f}').zfill(6)
        )

        initializatin_info['Translation Model Load Time -- '] = (
            f'{translation_model_time_string} s'
        )

    # Start measuring tokenization and embedding time:
    query_embedding_start_time = time.time()

    # Tokenize the search request:
    query_tokenized = embedding_tokenizer(
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
    query_embedding_time = time.time() - query_embedding_start_time

    search_result = None

    # Dense vectors search:
    if (search_type == 'Sentence Dense Vectors'):
        # Perform sematic search:
        search_start_time = time.time()

        query_dense_embedding = query_embedded[0][0]

        search_result_dataframe = fupi_dense_vectors_searcher(
            sentence_level_table,
            query_dense_embedding
        )

        search_time = time.time() - search_start_time

        # Translate:
        translation_start_time = time.time()

        search_result = search_result_translator(search_result_dataframe)

        translation_time = time.time() - translation_start_time

    # ColBERT Centroids search:
    if (search_type == 'Sentence ColBERT Centroids'):
        search_start_time = time.time()

        query_colbert_embeddings = query_embedded[1][0]

        search_result_dataframe = fupi_colbert_centroids_searcher(
            sentence_level_table,
            query_colbert_embeddings
        )

        search_time = time.time() - search_start_time 

        # Translate:
        translation_start_time = time.time()

        search_result = search_result_translator(search_result_dataframe)

        translation_time = time.time() - translation_start_time

    # Prepare search info:
    total_time = (
        embedding_model_time +
        translation_model_time +
        query_embedding_time +
        search_time +
        translation_time
    )

    query_embedding_time_string = str(f'{query_embedding_time:.3f}').zfill(6)
    search_time_string          = str(f'{search_time:.3f}').zfill(6)
    translation_time_string     = str(f'{translation_time:.3f}').zfill(6)
    total_time_string           = str(f'{total_time:.3f}').zfill(6)

    search_info = {
        'Query Embedding Time --------- ': f'{query_embedding_time_string} s',
        'LanceDB Search Time ---------- ': f'{search_time_string} s',
        'Translation Time ------------- ': f'{translation_time_string} s',
        'Total Time ------------------- ': f'{total_time_string} s',
    }

    if initializatin_info:
        combined_search_info = {}
        combined_search_info.update(initializatin_info)
        combined_search_info.update(search_info)
        search_info = combined_search_info

    memory_usage_megabytes = round(
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        2
    )

    memory_usage_gigabytes = round((memory_usage_megabytes / 1024), 2)

    print(f'Maximal RSS memory usage: {memory_usage_gigabytes} GB')

    return search_info, search_result


def activity_inspector() -> True:
    global last_activity

    thread = threading.Timer(
        int(os.environ['INACTIVITY_CHECK_SECONDS']),
        activity_inspector
    )

    thread.daemon = True

    thread.start()

    inactive_time_seconds = time.time() - last_activity

    if inactive_time_seconds > int(os.environ['INACTIVITY_MAXIMUM_SECONDS']):
        print(f'Initiating shutdown sequence at {datetime.datetime.now()}')

        os.kill(os.getpid(), signal.SIGINT)

    return True


def main():
    # Load object storage settings from .env file:
    load_dotenv(find_dotenv())

    lancedb_bucket_name = None
    global models_bucket_name

    # Object storage settings for Fly.io deployment:
    if os.environ.get('FLY_APP_NAME') is not None:
        os.environ['AWS_ENDPOINT']          = os.environ['PROD_ENDPOINT_S3']
        os.environ['AWS_ACCESS_KEY_ID']     = os.environ['PROD_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['PROD_SECRET_ACCESS_KEY']
        os.environ['AWS_REGION']            = 'auto'

        lancedb_bucket_name = os.environ['PROD_LANCEDB_BUCKET']
        models_bucket_name  = os.environ['PROD_MODELS_BUCKET']
    # Object storage settings for local development:
    else:
        os.environ['AWS_ENDPOINT']          = os.environ['DEV_ENDPOINT_S3']
        os.environ['AWS_ACCESS_KEY_ID']     = os.environ['DEV_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['DEV_SECRET_ACCESS_KEY']
        os.environ['AWS_REGION']            = 'us-east-1'
        os.environ['ALLOW_HTTP']            = 'True'

        lancedb_bucket_name = os.environ['DEV_LANCEDB_BUCKET']
        models_bucket_name  = os.environ['DEV_MODELS_BUCKET']

    # Define LanceDB tables:
    lance_db = lancedb.connect(f's3://{lancedb_bucket_name}/')

    global sentence_level_table
    global text_level_table

    sentence_level_table = lance_db.open_table('sentence-level')
    text_level_table     = lance_db.open_table('text-level')

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

    search_info_box=gr.JSON(
        label='Search Info',
        show_label=True,
        elem_classes='search'
    )

    search_results_box=gr.JSON(
        label='Search Results',
        show_label=True,
        elem_classes='search'
    )

    # Dark theme by default:
    javascript_code = '''
        function refresh() {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
    '''

    css_code = '''
        a:link {
            color: white;
            text-decoration: none;
        }

        a:visited {
            color: white;
            text-decoration: none;
        }

        a:hover {
            color: white;
            text-decoration: none;
        }

        a:active {
            color: white;
            text-decoration: none;
        }

        .search {font-size: 16px !important}
    '''

    # Initialize Gradio interface:
    gradio_interface = gr.Blocks(
        theme=gr.themes.Glass(),
        js=javascript_code,
        css=css_code,
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
                    https://github.com/ddmitov/fupi  
                    https://fupi.fly.dev  
                    License: Apache License 2.0.  
                    '''
                )

            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    Dataset: Common Crawl News - 2021 Bulgarian  
                    https://commoncrawl.org/blog/news-dataset-available  
                    https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual  
                    '''
                )

            with gr.Column(scale=40):
                gr.Markdown(
                    '''
                    Embedding Model: https://huggingface.co/ddmitov/bge_m3_dense_colbert_onnx  
                    Translation Model: https://huggingface.co/michaelfeil/ct2fast-m2m100_418M  
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
                    'renewable energy'
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
