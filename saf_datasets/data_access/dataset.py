import os
import gdown
import torch
from pathlib import Path
from typing import List, Dict, Iterable
from torch import Tensor
from tqdm import tqdm
from saf import Sentence, Vocabulary

BASE_PATH = ".saf_data"


class SentenceDataSet(Iterable[Sentence]):
    def __init__(self, path: str, url: str):
        self.data_path: str = os.path.normpath(os.path.join(str(Path.home()), BASE_PATH, path))
        self._vocab: Dict[str, Vocabulary] = dict()
        self._embeddings: Dict[str, Tensor] = dict()
        self._emb_indices: Dict[str, Tensor] = dict()
        if (not os.path.exists(self.data_path)):
            os.makedirs(os.path.join(*os.path.split(self.data_path)[:-1]))
            gdown.download(url, self.data_path)

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def vocabulary(self, source: str = "_token") -> Vocabulary:
        if (source not in self._vocab):
            self._vocab[source] = Vocabulary(self, source=source)

        return self._vocab[source]

    def add_symbols(self, symbols: List[str], source: str = "_token"):
        vocab = self.vocabulary(source)
        vocab.add_symbols(symbols)

    def del_symbols(self, symbols: List[str], source: str = "_token"):
        vocab = self.vocabulary(source)
        vocab.del_symbols(symbols)

    def to_indices(self, source: str = "_token", default: int = -1, padding: int = 0, pad_symbol: str = None,
                   start_symbol: str = None, end_symbol: str = None) -> Tensor:
        indices = self.vocabulary(source).to_indices(self, default, padding, pad_symbol, start_symbol, end_symbol)
        return torch.tensor(indices, dtype=torch.int64)

    def embeddings(self, tag: str, device: str = "cpu") -> Tensor:
        if (tag not in self._embeddings):
            if (tag in self[0].annotations):
                embeddings = [sent.annotations[tag] for sent in tqdm(self, desc="Building embedding indices")]
                self._embeddings = torch.stack(embeddings).to(device)
                for i, sent in enumerate(self):
                    sent.annotations[tag] = self._embeddings[i]
                    sent.annotations[f"{tag}_idx"] = i
                self._emb_indices[tag] = torch.tensor([sent.annotations[f"{tag}_idx"] for sent in self], dtype=torch.int64)
            else:
                embeddings = list()
                for sent in tqdm(self, desc="Building embedding indices"):
                    embeddings.append(torch.stack([tok.annotations[tag] for tok in sent.tokens]))
                self._embeddings[tag] = torch.stack(embeddings).to(device)
                for i, sent in enumerate(self):
                    for j, tok in enumerate(sent.tokens):
                        tok.annotations[tag] = self._embeddings[tag][i][j]
                        tok.annotations[f"{tag}_idx"] = (i, j)

        return self._embeddings[tag]

    # def to_embedding_indices(self, tag: str, pad_emb_idx: int = None, start_emb_idx: int = None,
    #                          end_emb_idx: int = None, device: str = "cpu") -> Tensor:
    #     self.embeddings(tag, device)
    #     if (tag not in self._emb_indices):
    #         indices = list()
    #         for sent in enumerate(self):
    #             indices.append()
    #             for tok in enumerate(sent.tokens):
    #
    #
    #     return self._emb_indices[tag]







