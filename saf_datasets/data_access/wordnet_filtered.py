from typing import Callable
from saf import Vocabulary
from .dataset import SentenceDataSet, BASE_URL
from .wiktionary import WiktionaryDefinitionCorpus

PATH = "WordNet/wordnet_filtered_spacy_dsr_srl.jsonl.bz2"
URL = BASE_URL + "wordnet_filtered_spacy_dsr_srl.jsonl.bz2"

ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+dsr+srl": {
        "path": "WordNet/wordnet_filtered_spacy_dsr_srl.jsonl.bz2",
        "url": BASE_URL + "wordnet_filtered_spacy_dsr_srl.jsonl.bz2"
    }
}


class WordNetFilteredDataSet(SentenceDataSet):
    """
    WordNet filtered data set: A collection of filtered definition sentences from WordNet

    Each element of this dataset is a definition sentence (gloss) of a single synset, as it appears in WordNet.
    Definitions were filtered to only include those which definiendum (the term being defined) is representable by
    a single token from the LLaMa tokenizer.
    Each sentence is annotated with a "definiendum" and corresponding metadata from WordNet (WordNet id,
    brown_frequency, wordnet_frequency, category, abstraction_level, generalization_level).

    A pre-annotated instance can be obtained as follows:

    >>> dataset = WordNetFilteredDataSet.from_resource("pos+lemma+ctag+dep+dsr+srl")

    """
    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        super(WordNetFilteredDataSet, self).__init__(path, url, **kwargs)

    def vocabulary(self, source: str = "_token", lowercase: bool = True) -> Vocabulary:
        return WiktionaryDefinitionCorpus.vocabulary(self, source, lowercase)

    @classmethod
    def from_resource(cls, locator: str):
        """
        Downloads a pre-annotated resource available at the specified locator

        Example:
            >>> dataset = WordNetFilteredDataSet.from_resource("pos+lemma+ctag+dep+dsr+srl")
        """
        dataset = None
        if (locator in ANNOT_RESOURCES):
            dataset = cls(**ANNOT_RESOURCES[locator])
        else:
            print(f"No resource found at locator: {locator}")

        return dataset





