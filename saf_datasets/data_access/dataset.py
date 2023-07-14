import os
import gdown
import torch
from pathlib import Path
from typing import List, Dict, Iterable
from collections import Counter
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
            os.makedirs(os.path.join(*os.path.split(self.data_path)[:-1]), exist_ok=True)
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
        return torch.tensor(indices, dtype=torch.int32)

    def embeddings(self, tag: str, device: str = "cpu") -> Tensor:
        if (tag not in self._embeddings):
            if (tag in self[0].annotations):
                embeddings = [sent.annotations[tag] for sent in tqdm(self, desc="Building embedding indices")]
                self._embeddings[tag] = torch.stack(embeddings).to(device)
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

    def to_positional_indices(self, source: str = "_token", default: int = -1, repetitions: int = 4) -> Tensor:
        indices = self.vocabulary(source).to_indices(self, default, 0, None, None, None)
        pos_idx = list()
        pos_val = list()
        for i in range(len(indices)):
            rel_pos = torch.tensor(range(1, len(indices[i]) + 1)) / len(indices[i])
            rep_counter = Counter()
            for j in range(len(indices[i])):
                rep_counter.update((indices[i][j],))
                if (rep_counter[indices[i][j]] <= repetitions):
                    pos_idx.append([i, indices[i][j], rep_counter[indices[i][j]] - 1])
                    pos_val.append(rel_pos[j])

        return torch.sparse_coo_tensor(list(zip(*pos_idx)), pos_val, (len(indices), len(self.vocabulary()), repetitions))

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







