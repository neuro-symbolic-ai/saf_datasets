import os
import gzip
from csv import DictReader
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token, Vocabulary
from .dataset import SentenceDataSet, BASE_URL

PATH = "AllNLI/AllNLI.jsonl.bz2"
URL = BASE_URL + "AllNLI.jsonl.bz2"


class AllNLIDataSet(SentenceDataSet):
    """
    Wrapper for the AllNLI dataset: https://www.sbert.net/examples/datasets/README.html

    Sentence annotations: id, dataset, split, label
    """
    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        super(AllNLIDataSet, self).__init__(path, url, **kwargs)

