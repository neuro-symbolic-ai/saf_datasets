import os
import gzip
from csv import DictReader
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from saf_datasets.annotators.spacy import SpacyAnnotator
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet, BASE_URL

PATH = "STSB/stsbenchmark.jsonl.bz2"
URL = BASE_URL + "stsbenchmark.jsonl.bz2"


class STSBDataSet(SentenceDataSet):
    """
    Wrapper for the STSB dataset (Cer et al. 2017): https://aclanthology.org/S17-2001/

    Sentence pairs are split into adjacent entries, sharing the same 'sid'.

    Sentence annotations: id, sid, split, genre, dataset
    """
    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        super(STSBDataSet, self).__init__(path, url, **kwargs)

