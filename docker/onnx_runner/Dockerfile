FROM python:3.10

RUN pip install --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache \
    boto3 \
    gradio \
    lancedb \
    langchain \
    onnxruntime \
    optimum[onnxruntime] \
    pandas \
    pyarrow \
    transformers

RUN mkdir /.cache
RUN chmod 777 /.cache

CMD ["python"]

# docker build -t onnx_runner .
