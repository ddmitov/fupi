#!/usr/bin/env python3

import os

# import boto3
# import botocore
import gradio as gr
import lancedb
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 onnx_embedder python /app/e5_onnx_gradio_searcher.py


def lancedb_search(search_request: str)-> object:
    # Initialize the embedding model:
    onnx_session = InferenceSession(
        '/etc/model/embedder.onnx',
        providers=['CPUExecutionProvider']
    )

    tokenizer = AutoTokenizer.from_pretrained('/etc/tokenizer')

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

    # Define LanceDB table:
    BUCKET_NAME = 'splitting-batching-test'

    lance_db = lancedb.connect(f's3://{BUCKET_NAME}/')
    lancedb_table = lance_db.open_table('fupi')

    # Vectorize the search input and search:
    tokenized_input = tokenizer(
        search_request,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    embeddings = onnx_session.run(None, dict(tokenized_input))

    search_results = lancedb_table.search(embeddings[0][0]).limit(5).to_pandas()

    search_results = search_results.drop(['vector'], axis=1)

    return search_results.to_dict('records')


def main():
    interface = gr.Blocks()

    with interface:
        gr.Markdown(
            """
            # LanceDB Semantic Search
            Search using LanceDB
            """
        )

        with gr.Row():
            input=gr.Textbox(
                lines=1,
                label='Search Request'
            )
        
        search_button = gr.Button("Search")

        with gr.Row():
            output=gr.JSON(label='Search Results')

        search_button.click(
            lancedb_search,
            inputs=input,
            outputs=output
        )

        gr.Examples(
            [['злоупотреби в Калифорния']],
            fn=lancedb_search,
            inputs=input,
            outputs=output,
            cache_examples=False
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
