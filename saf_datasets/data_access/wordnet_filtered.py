import gzip
import jsonlines
import pickle
from tqdm import tqdm
from saf import Token
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet
from .wiktionary import WiktionaryDefinitionCorpus

PATH = "WordNet/WordNet_filtered_data.jsonl.gz"
URL = "https://drive.google.com/uc?id=1v6UpPjbJ_QnsFZNCJQ0avB6yxjfhKime"

ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+dsr+srl": {
        "path": "WordNet/wordnet_filtered_spacy_dsr_srl.pickle.gz",
        "url": "https://drive.google.com/uc?id=1RuYJCvXz0iDQId-7RvMKP5z56SNQSdWM"
    }
}


class WordNetFilteredDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL):
        super(WordNetFilteredDataSet, self).__init__(path, url)

        with gzip.open(self.data_path) as dataset_file:
            with jsonlines.Reader(dataset_file) as reader:
                self.data = list()
                for line in tqdm(reader, desc="Loading WordNet filtered data"):
                    sentence = Sentence()
                    sentence.surface = line["definition"]
                    sentence.annotations["definiendum"] = line["name"]
                    sentence.annotations["brown_frequency"] = line["brown_frequency"]
                    sentence.annotations["wordnet_frequency"] = line["wordnet_frequency"]
                    sentence.annotations["category"] = line["category"]
                    sentence.annotations["abstraction_level"] = line["abstraction_level"]
                    sentence.annotations["generalization_level"] = line["generalization_level"]
                    sentence.annotations['id'] = line["id"]

                    for tok in line["definition_attributes"]:
                        token = Token()
                        token.surface = tok['name']
                        token.annotations = tok #add the ID to the token annotations
                        del token.annotations['name']
                        del token.annotations['definition']
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

    def vocabulary(self, source: str = "_token", lowercase: bool = True) -> Vocabulary:
        return WiktionaryDefinitionCorpus.vocabulary(self, source, lowercase)


    @staticmethod
    def from_resource(locator: str):
        dataset = None
        if (locator in ANNOT_RESOURCES):
            path = ANNOT_RESOURCES[locator]["path"]
            url = ANNOT_RESOURCES[locator]["url"]
            data_path = WiktionaryDefinitionCorpus.download_resource(path, url)
            with gzip.open(data_path, "rb") as resource_file:
                dataset = pickle.load(resource_file)
        else:
            print(f"No resource found at locator: {locator}")

        return dataset





