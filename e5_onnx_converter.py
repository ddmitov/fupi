#!/usr/bin/env python3

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app onnx_converter python /app/e5_onnx_converter.py


def main():
    model_checkpoint = "intfloat/multilingual-e5-large"
    save_directory = "/app/data/e5"

    model = ORTModelForFeatureExtraction.from_pretrained(
        model_checkpoint,
        export=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)


if __name__ == '__main__':
    main()
