FROM python:3.12

# CPU-only Torch module:
RUN pip install --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Fupi modules:
RUN pip install --no-cache \
    ctranslate2            \
    duckdb                 \
    hf_hub_ctranslate2     \
    huggingface-hub        \
    lancedb                \
    minio                  \
    onnxruntime            \
    pandas                 \
    pyarrow                \
    python-dotenv          \
    sentencepiece          \
    transformers

# Data processing module:
RUN pip install --no-cache pysbd

# Gradio search application module:
RUN pip install --no-cache gradio

# LanceDB settings:
RUN mkdir     /.cache
RUN chmod 777 /.cache

RUN mkdir     /.config
RUN chmod 777 /.config

RUN mkdir     /.config/lancedb
RUN chmod 777 /.config/lancedb

# Fupi Gradio search application settings:
RUN mkdir     /.config/matplotlib
RUN chmod 777 /.config/matplotlib

RUN mkdir /home/fupi
RUN mkdir /home/fupi_data

COPY ./.env        /home/fupi/.env
COPY ./fupi        /home/fupi/fupi
COPY ./searcher.py /home/fupi/searcher.py

# Start Fupi Gradio search application by default:
EXPOSE 7860
CMD ["python", "/home/fupi/searcher.py"]

# docker build -t fupi .
# docker buildx build -t fupi .
