from .dataset import SentenceDataSet, BASE_URL


PATH = "WordNet/Spanish_WordNet_filtered_data.jsonl.bz2"
URL = BASE_URL + "Spanish_WordNet_filtered_data.jsonl.bz2"


class SpanishWordNetFilteredDataSet(SentenceDataSet):
    """
        Spanish WordNet filtered data set: A collection of filtered definition sentences from the Spanish WordNet

        Each element of this dataset is a definition sentence (gloss) of a single synset, as it appears in WordNet.
        Definitions were filtered to only include those which definiendum (the term being defined) is representable by
        a single token from the LLaMa tokenizer.
        Each sentence is annotated with a "definiendum" and corresponding metadata from WordNet (WordNet id,
        cess_esp_frequencies, wordnet_frequency, category, abstraction_level, generalization_level).

    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(SpanishWordNetFilteredDataSet, self).__init__(path, url)
