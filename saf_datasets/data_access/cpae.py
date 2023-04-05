import os
import bz2
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from saf_datasets.annotators.spacy import SpacyAnnotator
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet

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

    def vocabulary(self, source: str = "_token") -> Vocabulary:
        if (source not in self._vocab):
            self._vocab[source] = Vocabulary(self, source=source)
        if (source == "_token"):
            definiendum_vocab = Vocabulary(self, source="definiendum")
            self._vocab[source].add_symbols(list(definiendum_vocab.symbols))
            definiendum_vocab._vocab = self._vocab[source]._vocab
            definiendum_vocab.freqs = self._vocab[source].freqs
            self._vocab["definiendum"] = definiendum_vocab

        return self._vocab[source]
