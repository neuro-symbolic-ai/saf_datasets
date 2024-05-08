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
    """
    Wrapper for the STSB dataset (Cer et al. 2017): https://aclanthology.org/S17-2001/

    Sentence pairs are split into adjacent entries, sharing the same 'sid'.

    Sentence annotations: id, sid, split, genre, dataset
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(STSBDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

        with gzip.open(self.data_path, "rt", encoding="utf-8") as dataset_file:
            self.data = list()
            i = -1
            for row in tqdm(dataset_file, desc="Loading STSB data"):
                if (i < 0):
                    i += 1
                    continue

                fields = row.split("\t")
                sentence1 = fields[6]
                sentence2 = fields[7]
                split = fields[0]
                genre = fields[1]
                data_set = fields[2]
                year = fields[3]
                sid = fields[4]
                score = fields[5]
                for sent in [sentence1, sentence2]:
                    sentence = Sentence()
                    sentence.annotations["split"] = split
                    sentence.annotations["genre"] = genre
                    sentence.annotations["dataset"] = data_set
                    sentence.annotations["year"] = year
                    sentence.annotations["sid"] = sid
                    sentence.annotations["score"] = score
                    sentence.annotations["id"] = i
                    sentence.surface = sent.strip()
                    for tok in self.tokenizer(sentence.surface):
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

