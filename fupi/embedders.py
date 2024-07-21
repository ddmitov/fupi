import os
import copy
from typing import Dict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import onnxruntime as ort
from transformers import PreTrainedTokenizer
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

from fupi.data import Dataset
from fupi.utils import (lancedb_tables_creator, 
                        create_dataset_bucket, 
                        init_minio)


class Embedder(ABC):
    
    @abstractmethod
    def embed(self, dataset: Dataset, model: nn.Module):
        pass


class LanceDBEmbedder(ABC):
    """Creates text and sentence embeddings and saves them into LanceDB.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        model (InferenceSession): ONNX Inference session.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: InferenceSession):
        self._tokenizer = tokenizer
        self._model = model

        self._create_bucket()

    def embed(self, dataset: Dataset):
        """Creates embeddings from dataset and saves them into LanceDB.

        Args:
            dataset (Dataset): Fupi Dataset object.
        """
        batches = tqdm(dataset, total=len(dataset))
        batches.set_description("Creating embeddings")
        sentences_list = []

        # Create embeddings for each batch of sentences.
        for i, batch in enumerate(batches):
            sentences_batch = copy.deepcopy(batch)
            tokens = self._tokenizer([sample["sentence"] for sample in sentences_batch], 
                               padding="longest", 
                               return_tensors="np")
            dense_embeddings, colbert_centroids = self._get_embeddings(tokens)

            for i in range(len(sentences_batch)):
                sentences_batch[i]["dense_embedding"] = dense_embeddings[i]
                sentences_batch[i]["colbert_embedding"] = colbert_centroids[i]

            sentences_list.extend(sentences_batch)

        sentences_df = pd.DataFrame(sentences_list)
        texts_df = self._create_texts_dataframe(sentences_df)

        self._add_to_lancedb(sentence_dataframe=sentences_df, 
                             texts_dataframe=texts_df)

    def _create_bucket(self):
        minio_client = init_minio()
        create_dataset_bucket(
            minio_client, 
            bucket_name=os.environ['DEV_LANCEDB_BUCKET']
        )

    def _get_embeddings(self, sequences: Dict[str, np.ndarray]):
        inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in sequences.items()}

        if isinstance(self._model, InferenceSession):
            outputs = self._model.run(None, inputs_onnx)
        else:
            raise Exception("We are currently only using ONNX InferenceSession objects for embedding.")

        dense_embeddings, colbert_embeddings = outputs
        colbert_centroids = np.average(colbert_embeddings, axis=1)
        
        return dense_embeddings, colbert_centroids

    def _create_texts_dataframe(self, sentences_dataframe: pd.DataFrame):
        texts_df = sentences_dataframe.drop_duplicates(subset=["text_id"])
        texts_df = texts_df.drop(columns=["sentence_id", 
                                          "sentence", 
                                          "dense_embedding", 
                                          "colbert_embedding"])

        aggregated_texts_df = (
            sentences_dataframe.groupby(
                [
                    'text_id'
                ]
            ).agg(
                {
                    'dense_embedding': [self._centroid_maker_for_series]
                }
            )
        ).reset_index()
        aggregated_texts_df.columns = aggregated_texts_df.columns.get_level_values(0)
        texts_df = pd.merge(
            texts_df,
            aggregated_texts_df,
            on='text_id',
            how='left'
        )

        return texts_df

    def _add_to_lancedb(self, sentence_dataframe: pd.DataFrame, texts_dataframe: pd.DataFrame):
        text_level_table, sentence_level_table = (
            lancedb_tables_creator(os.environ['DEV_LANCEDB_BUCKET'])
        )

        sentence_level_table.add(sentence_dataframe)
        text_level_table.add(texts_dataframe)

        text_level_table.compact_files()
        sentence_level_table.compact_files()

    @staticmethod
    def _centroid_maker_for_series(group: pd.Series) -> list:
        embeddings_list = group.tolist()
        average_embedding_list = np.average(embeddings_list, axis=0).tolist()

        return average_embedding_list
