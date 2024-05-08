import os
import gzip
from csv import DictReader
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token, Vocabulary
from .dataset import SentenceDataSet

PATH = "AllNLI/AllNLI.tsv.gz"
URL = "https://sbert.net/datasets/AllNLI.tsv.gz"


class AllNLIDataSet(SentenceDataSet):
    """
    Wrapper for the AllNLI dataset: https://www.sbert.net/examples/datasets/README.html

    Sentence annotations: id, dataset, split, label
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(AllNLIDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

        with gzip.open(self.data_path, "rt", encoding="utf-8") as dataset_file:
            self.data = list()
            reader = DictReader(dataset_file, delimiter="\t")
            i = 0
            for row in tqdm(reader, desc="Loading AllNLI data"):
                for sent in [row["sentence1"], row["sentence2"]]:
                    sentence = Sentence()
                    sentence.annotations["split"] = row["split"]
                    sentence.annotations["label"] = row["label"]
                    sentence.annotations["dataset"] = row["dataset"]
                    sentence.annotations["id"] = i
                    sentence.surface = sent
                    for tok in self.tokenizer(sent):
                        token = Token()
                        token.surface = tok.text
                        sentence.tokens.append(token)

                    self.data.append(sentence)
                i += 1

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sentence:
        """Fetches the ith sentence in the dataset.

        Args:
            idx (int): index for the ith sentence in the dataset.

        :return: A single term decomposition (Sentence).
        """
        return self.data[idx]

