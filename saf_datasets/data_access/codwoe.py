import os
import bz2
import json
import torch
from typing import Tuple, List, Dict, Iterable
from tqdm import tqdm
from tarfile import TarFile
from spacy.lang.en import English
from torch import Tensor
from saf import Sentence, Token
from saf_datasets.annotators.spacy import SpacyAnnotator
from .dataset import SentenceDataSet, BASE_URL

PATH = "CODWOE/CODWOE.tar"
URL = BASE_URL + "CODWOE.tar"


class CODWOEDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL, langs: Tuple[str] = ("en",)):
        """
        Wrapper for the CODWOE dataset from Semeval-2022 Task 1 (Mickus et al. 2022): https://github.com/TimotheeMickus/codwoe
        """
        super(CODWOEDataSet, self).__init__(path, url)
        TarFile.open(self.data_path).extractall(os.path.join(*self.data_path.split(os.sep)[:-2]))
        path = os.path.join(*self.data_path.split(os.sep)[:-1])

        self._ids: List[str] = list()
        self._glosses: List[str] = list()
        self._emb_char: Tensor = None
        self._emb_electra: Tensor = None
        self._emb_sgns: Tensor = None

        self.split_indices: Dict[str, Tuple[int, int]] = dict()

        emb_char = list()
        emb_electra = list()
        emb_sgns = list()
        for lang in langs:
            prev_split_idx = 0
            for split in ("train", "dev"):
                with bz2.open(os.path.join(path, f"{lang}.{split}.json.bz2"), "rb") as dataset_file:
                    data = json.load(dataset_file)
                    self.split_indices[split] = (prev_split_idx, len(data))
                    prev_split_idx = len(data)
                    for defn in tqdm(data, desc=f"Loading CODWOE data ({lang}.{split})"):
                        self._ids.append(defn["id"])
                        self._glosses.append(defn["gloss"])
                        emb_char.append(defn["char"])
                        emb_electra.append(defn["electra"])
                        emb_sgns.append(defn["sgns"])

        self._emb_char = torch.tensor(emb_char)
        self._emb_electra = torch.tensor(emb_electra)
        self._emb_sgns = torch.tensor(emb_sgns)
        self._size: int = len(self._ids)
        self._index: Dict[str, List[Sentence]] = dict()
        self._definitions: List[Sentence] = list()
        self.tokenizer = English().tokenizer

        del emb_char
        del emb_electra
        del emb_sgns

    def __iter__(self):
        return CODWOEDataSetIterator(self)

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int) -> Sentence:
        """Fetches the ith definition in the dataset or all definitions for a given term.

        :param item: (int) for the ith definition in the dataset.
        :return: A single definition (Sentence).
        """
        return self.def_convert(idx)

    def def_convert(self, idx: int) -> Sentence:
        """Converts CODWOE definition entry to SAF sentence format

        :param idx: (int) for the ith definition in the corpus.
        :return: A single definition (Sentence).
        """
        sentence = Sentence()
        sentence.annotations["id"] = self._ids[idx]
        sentence.annotations["emb_char"] = self._emb_char[idx]
        sentence.annotations["emb_electra"] = self._emb_electra[idx]
        sentence.annotations["emb_sgns"] = self._emb_sgns[idx]
        sentence.surface = self._glosses[idx].strip()
        for tok in self.tokenizer(sentence.surface):
            token = Token()
            token.surface = tok.text
            sentence.tokens.append(token)

        return sentence

    def embeddings(self, tag: str, device: str = "cpu") -> Tensor:
        embeddings = None
        if (tag == "char"):
            self._emb_char = self._emb_char.to(device)
            embeddings = self._emb_char
        elif (tag == "electra"):
            self._emb_electra = self._emb_electra.to(device)
            embeddings = self._emb_electra
        elif (tag == "sgns"):
            self._emb_sgns = self._emb_sgns.to(device)
            embeddings = self._emb_sgns
        else:
            embeddings = super(CODWOEDataSet, self).embeddings(tag, device)

        return embeddings

    def train_indices(self) -> Tuple[int, int]:
        return self.split_indices["train"]

    def dev_indices(self) -> Tuple[int, int]:
        return self.split_indices["dev"]


class CODWOEDataSetIterator:
    def __init__(self, codwoeds: CODWOEDataSet):
        self._codwoeds: CODWOEDataSet = codwoeds
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        for i in range(len(self._codwoeds)):
            yield self._codwoeds.def_convert(i)


if __name__ == "__main__":
    defs = CODWOEDataSet()

    print(defs.vocabulary().freqs.most_common())

    exit(0)

    i = 0
    for sent in defs:
        print("Sent annotations:", sent.annotations)
        print("Token annotations:", [token.surface for token in sent.tokens])
        i += 1
        if (i > 10):
            break

    examples = [defs[5], defs[8]]
    annotator = SpacyAnnotator()
    annotator.annotate(examples)
    for example in examples:
        print(f"{example.annotations['id']} (\"{example.surface}\"): ")
        print(f"{[(token.surface, token.annotations) for token in example.tokens]}")

    print("Corpus size:", len(defs))
