from typing import Callable
from .dataset import SentenceDataSet, BASE_URL

FILE_VERSION = "complex_sentences"
PATH = "ComplexSentences/%s.json.bz2" % FILE_VERSION
URL = BASE_URL + "%s.json.bz2" % FILE_VERSION


class ComplexSentencesDataSet(SentenceDataSet):
    """
    Wrapper for the Complex Sentences dataset (Niklaus et al.): https://aclanthology.org/P19-1333/

    Sentence annotations: source ["wiki", "newsela"], split
    Token annotations: label (see `ComplexSentencesDataSet.LABELS`)
    """

    LABELS = {
        "CNP": "coordinate noun phrases",
        "CVP":" coordinate verb phrases",
        "PreP": "prepositional phrase",
        "ParP": "participial phrase",
        "AP": "appositive phrase",
        "AM": "adverbial modifier",
        "CC": "coordinate clauses",
        "AC": "adverbial clause",
        "RC": "relative clause",
        "IDC": "(in)direct speech"
    }

    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        super(ComplexSentencesDataSet, self).__init__(path, url, **kwargs)

