FROM python:3.10

RUN pip install --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache \
    optimum[onnxruntime] \
    transformers

RUN mkdir /.cache
RUN chmod 777 /.cache

CMD ["python"]

# docker build -t onnx_converter .
