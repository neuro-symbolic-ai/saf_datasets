import gzip
import jsonlines
from tqdm import tqdm
from saf import Token
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet, BASE_URL


PATH = "WordNet/Spanish_WordNet_filtered_data.jsonl.gz"
URL = BASE_URL + "Spanish_WordNet_filtered_data.jsonl.gz"


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
        with gzip.open(self.data_path, 'rt') as dataset_file:
            with jsonlines.Reader(dataset_file) as reader:
                self.data = list()
                for line in tqdm(reader, desc="Loading WordNet filtered data"):
                    if (not line["definition"]):
                        continue

                    sentence = Sentence()
                    sentence.surface = line["definition"]
                    sentence.annotations["definiendum"] = line["name"]
                    sentence.annotations["cess_esp_frequencies"] = line["cess_esp_frequencies"]
                    sentence.annotations["wordnet_frequency"] = line["wordnet_frequency"]
                    sentence.annotations["category"] = line["category"]
                    sentence.annotations["abstraction_level"] = line["abstraction_level"]
                    sentence.annotations["generalization_level"] = line["generalization_level"]
                    sentence.annotations['id'] = line["id"]
                    sentence.annotations['definition_src'] = line['definition_src']

                    for tok in line["definition_attributes"]:
                        token = Token()
                        token.surface = tok['name']
                        token.annotations = tok #add the ID to the token annotations
                        sentence.tokens.append(token)

                    self.data.append(sentence)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sentence:
        """Fetches the ith definition in the dataset.

        Args:
            idx (int): index for the ith term in the dataset.

        :return: A single term definition (Sentence).
        """
        return self.data[idx]