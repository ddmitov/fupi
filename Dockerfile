FROM python:3.12

# Data processing modules:
    RUN pip install --no-cache \
    huggingface-hub \
    pysbd

# CPU-only Torch module:
RUN pip install --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Fupi core modules:
RUN pip install --no-cache \
    ctranslate2 \
    duckdb \
    lancedb \
    minio \
    onnxruntime \
    pandas \
    pyarrow \
    python-dotenv \
    transformers

# Fupi Gradio demo application module:
RUN pip install --no-cache gradio

# LanceDB settings:
RUN mkdir /.cache
RUN chmod 777 /.cache

RUN mkdir /.config
RUN chmod 777 /.config

RUN mkdir /.config/lancedb
RUN chmod 777 /.config/lancedb

# Fupi Gradio demo application settings:
RUN mkdir /.config/matplotlib
RUN chmod 777 /.config/matplotlib

RUN mkdir /home/fupi

COPY ./.env /home/fupi/.env
COPY ./fupi_searcher.py /home/fupi/fupi_searcher.py

# Start Fupi Gradio demo application by default:
EXPOSE 7860
CMD ["python", "/home/fupi/fupi_searcher.py"]

# docker build -t fupi .
# docker buildx build -t fupi .
