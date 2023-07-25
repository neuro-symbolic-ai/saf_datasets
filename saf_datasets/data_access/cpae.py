import os
import bz2
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from saf_datasets.annotators.spacy import SpacyAnnotator
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet
from .wiktionary import WiktionaryDefinitionCorpus

PATH = "CPAE/cpae_definitions.csv.bz2"
URL = "https://drive.google.com/uc?id=16B8hVf5NkubN4G_J_SrryA6A8YoxEZLP"


class CPAEDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL):
        super(CPAEDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

        with bz2.open(self.data_path, "rt", encoding="utf-8") as dataset_file:
            self.data = list()
            for line in tqdm(dataset_file, desc="Loading CPAE definition data"):
                fields = line.split(";")
                sentence = Sentence()
                sentence.annotations["definiendum"] = fields[2]
                definition = fields[3]
                sentence.surface = definition
                for tok in self.tokenizer(definition):
                    token = Token()
                    token.surface = tok.text
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
