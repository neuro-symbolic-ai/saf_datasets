import bz2
from typing import Callable
from tqdm import tqdm
from spacy.lang.en import English
from saf import Token, Sentence, Vocabulary
from .dataset import SentenceDataSet, BASE_URL
from .wiktionary import WiktionaryDefinitionCorpus

PATH = "CPAE/cpae_definitions.jsonl.bz2"
URL = BASE_URL + "cpae_definitions.jsonl.bz2"


class CPAEDataSet(SentenceDataSet):
    """
    Wrapper for the CPAE dataset (Bosc, Vincent. 2018): https://aclanthology.org/D18-1181/
    """
    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        super(CPAEDataSet, self).__init__(path, url, **kwargs)

    def vocabulary(self, source: str = "_token", lowercase: bool = True) -> Vocabulary:
        return WiktionaryDefinitionCorpus.vocabulary(self, source, lowercase)
