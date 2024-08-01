from typing import List, Any, Dict
from abc import ABC, abstractmethod

import pysbd
import duckdb


class Dataset(ABC):
        
    @abstractmethod
    def __getitem__(self, index: int):
        pass
    
    @abstractmethod
    def __len__(self):
        pass


class HuggingFaceDataset(Dataset):
    """HuggingFace dataset handler that reads it from a file and splits it into batches of sentences.

    Args:
        path (str): Path to HuggingFace dataset file.
        num_samples (int, optional): Number of sample to take. Defaults to -1, which means all samples.
        batch_size (int, optional): Sentences batch size. Defaults to 1.
        segmenter (Any, optional): Segmenter to use. Defaults to pysbd.Segmenter.
    """

    def __init__(self, path: str, num_samples: int = -1, batch_size: int = 1, segmenter: Any = None):
        self._path = path
        self._num_samples = num_samples

        if not segmenter:
            self._segmenter = pysbd.Segmenter(language='bg', clean=False)
        else:
            self._segmenter = segmenter

        duckdb.create_function('newlines_remover', self._newlines_remover)
        duckdb.sql('CREATE SEQUENCE text_id_maker START 1')

        self._entries = duckdb.sql(
            self. _get_dataset_query()
        ).to_arrow_table().to_pylist()

        self._entries = self._split_into_sentences(self._entries)
        self._entries = self._create_batches(self._entries, batch_size)

    def __getitem__(self, index: int):
        return self._entries[index]

    def __len__(self):
        return len(self._entries)
    
    def _newlines_remover(self, text: str) -> str:
        return text.replace('\n', ' ')
    
    def _get_dataset_query(self):
        query = f'''
            SELECT
                nextval('text_id_maker') AS text_id,
                date_publish_final AS date,
                newlines_remover(title) AS title,
                newlines_remover(maintext) AS text
            FROM read_json_auto("{self._path}")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND maintext IS NOT NULL
                AND title NOT LIKE '%...'
        '''
        if self._num_samples > 0:
            return query + f' LIMIT {self._num_samples}'

        return query
    
    def _create_batches(self, entries: List[Any], batch_size: int):
        if batch_size == 1:
            return entries
        
        batches = []
        for i in range(0, len(entries), batch_size):
            batches.append(entries[i:i + batch_size])

        return batches
    
    def _split_into_sentences(self, entries: List[Dict[str, Any]]):
        result = []
        sentence_id = 0

        for entry in entries:
            sentences = self._segmenter.segment(entry['text'])
            for sentence in sentences:
                sentence_id += 1
                result.append({"date": entry["date"],
                               "text_id": entry["text_id"], 
                               "title": entry["title"], 
                               "sentence_id": sentence_id,
                               "sentence": sentence})
                
        return result
