from .dataset import SentenceDataSet, BASE_URL

PATH = "SNLI/snli_1.0.jsonl.bz2"
URL = BASE_URL + "snli_1.0.jsonl.bz2"


class SNLIDataSet(SentenceDataSet):
    """
    Wrapper for the Stanford Natural Language Inference (SNLI) Corpus (Bowman et al., 2015): http://nlp.stanford.edu/pubs/snli_paper.pdf
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(SNLIDataSet, self).__init__(path, url)
