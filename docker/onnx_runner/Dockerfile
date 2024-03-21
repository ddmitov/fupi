FROM python:3.10

RUN pip install --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache \
    boto3 \
    gradio \
    lancedb \
    langchain \
    nltk \
    onnxruntime \
    optimum[onnxruntime] \
    pandas \
    pyarrow \
    transformers

COPY ./nltk_downloader.py /etc/nltk_downloader.py
RUN python3 /etc/nltk_downloader.py

RUN mkdir /.cache
RUN chmod 777 /.cache

RUN chmod 777 /usr/share/nltk_data

CMD ["python"]

# docker build -t onnx_runner .
