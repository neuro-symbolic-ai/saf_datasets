import os
import gzip
from csv import DictReader
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from saf_datasets.annotators.spacy import SpacyAnnotator
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet

PATH = "STSB/stsbenchmark.tsv.gz"
URL = "https://sbert.net/datasets/stsbenchmark.tsv.gz"


class STSBDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL):
        super(STSBDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

        with gzip.open(self.data_path, "rt", encoding="utf-8") as dataset_file:
            self.data = list()
            reader = DictReader(dataset_file, delimiter="\t")
            i = 0
            for row in tqdm(reader, desc="Loading STSB data"):
                for sent in [row["sentence1"], row["sentence2"]]:
                    sentence = Sentence()
                    sentence.annotations["split"] = row["split"]
                    sentence.annotations["genre"] = row["genre"]
                    sentence.annotations["dataset"] = row["dataset"]
                    sentence.annotations["year"] = row["year"]
                    sentence.annotations["sid"] = row["sid"]
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

    def __getitem__(self, idx) -> Sentence:
        """Fetches the ith sentence in the dataset.

        Args:
            idx (int): index for the ith sentence in the dataset.

        :return: A single term decomposition (Sentence).
        """
        return self.data[idx]

