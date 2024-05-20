#!/usr/bin/env python3

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

model_checkpoint = 'ddmitov/bge_m3_dense_colbert_onnx'
save_directory = '/tmp/model'

hf_hub_download(
    repo_id=model_checkpoint,
    filename='model.onnx',
    local_dir=save_directory,
    repo_type='model'
)

hf_hub_download(
    repo_id=model_checkpoint,
    filename='model.onnx_data',
    local_dir=save_directory,
    repo_type='model'
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.save_pretrained(save_directory)
