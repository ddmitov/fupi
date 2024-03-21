#!/usr/bin/env python3

import os

import gradio as gr
import lancedb
# from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer, pipeline

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 onnx_runner python /app/e5_onnx_gradio_searcher.py

# MinIO object storage settings:
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['AWS_ENDPOINT'] = 'http://172.17.0.1:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# LanceDB settings:
LANCEDB_BUCKET_NAME = 'text-centroids'
LANCEDB_TABLE_NAME = 'fupi'


def lancedb_search(search_request: str)-> object:
    # Initialize tokenizer:
    tokenizer = AutoTokenizer.from_pretrained("/app/data/e5")

    tokenizer_kwargs = {'truncation': True}

    # Initialize embedding model:
    model = ORTModelForFeatureExtraction.from_pretrained(
        "/app/data/e5",
        providers=['CPUExecutionProvider']
    )

    # Initialize transformers pipeline:
    onnx_extractor = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer
    )

    # Define LanceDB table:
    lance_db = lancedb.connect(f's3://{LANCEDB_BUCKET_NAME}/')
    lancedb_table = lance_db.open_table(LANCEDB_TABLE_NAME)

    # Vectorize the search request:
    search_request_embeddings = onnx_extractor(
        search_request,
        **tokenizer_kwargs
    )

    # Perform LanceDB search:
    search_results = (
        lancedb_table
        .search(search_request_embeddings[0][0])
        .limit(5)
        .to_pandas()
    )

    search_results = search_results.drop(['vector'], axis=1)

    return search_results.to_dict('records')


def main():
    input_box=gr.Textbox(lines=1, label='Search Request')

    output_box=gr.JSON(label='Search Results')

    interface = gr.Blocks()

    with interface:
        gr.Markdown(
            """
            # Fupi Gradio Search
            ## Serverless multilingual semantic search testbed and demo project  
            https://github.com/ddmitov/fupi  
            Dataset: Department of Justice 2009-2018 Press Releases  
            https://www.kaggle.com/datasets/jbencina/department-of-justice-20092018-press-releases  
            Model: intfloat/multilingual-e5-large  
            https://huggingface.co/intfloat/multilingual-e5-large  
            """
        )

        with gr.Row():
            input_box.render()

        with gr.Row():
            search_button = gr.Button("Search")

            gr.ClearButton([input_box, output_box])

        with gr.Row():
            gr.Examples(
                [['злоупотреби в Калифорния']],
                fn=lancedb_search,
                inputs=input_box,
                outputs=output_box,
                cache_examples=False
            )

        with gr.Row():
            output_box.render()

        search_button.click(
            lancedb_search,
            inputs=input_box,
            outputs=output_box
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
