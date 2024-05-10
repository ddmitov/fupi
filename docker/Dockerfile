FROM python:3.12

RUN pip install --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache \
    ctranslate2 \
    duckdb \
    gradio \
    huggingface-hub \
    lancedb \
    minio \
    onnxruntime \
    pandas \
    pyarrow \
    pysbd \
    python-dotenv \
    sentencepiece \
    sentence-transformers \
    transformers

RUN mkdir /.cache
RUN chmod 777 /.cache

RUN mkdir /.config
RUN chmod 777 /.config

RUN mkdir /.config/lancedb
RUN chmod 777 /.config/lancedb

RUN mkdir /.config/matplotlib
RUN chmod 777 /.config/matplotlib

CMD ["python"]

# docker build -t onnx_runner .
