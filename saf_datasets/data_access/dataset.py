import os
import logging
import gdown
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Iterable, Callable
from collections import Counter
from torch import Tensor
from torch.nn.functional import relu
from tqdm import tqdm
from pyarrow import json as pj
from spacy.lang.en import English
from saf import Token, Sentence, Vocabulary

logging.basicConfig(level=logging.INFO)

BASE_PATH = ".saf_data"
BASE_URL = "http://personalpages.manchester.ac.uk/staff/danilo.carvalho/saf_datasets/"


class SentenceDataSet(Iterable[Sentence]):
    """
    Base class for the sentence datasets. Exposes an iterable of SAF Sentence objects.

    Attributes:
        data_path (str): The path where the dataset is stored.
    """
    def __init__(self, path: str, url: str, tokenizer: Callable = None):
        """
        Initializes the SentenceDataSet object by setting up the data path and downloading the data if necessary.

        Args:
            path (str): The subpath to the dataset within the base path.
            url (str): The URL from which to download the dataset if it does not exist locally.
            tokenizer (Callable): Optional custom tokenizer. Defaults to `spacy.lang.en.English`.
        """
        self.data_path: str = os.path.normpath(os.path.join(str(Path.home()), BASE_PATH, path))
        self._vocab: Dict[str, Vocabulary] = dict()
        self._embeddings: Dict[str, Tensor] = dict()
        self._emb_indices: Dict[str, Tensor] = dict()
        self._annotations: Dict[str, List[str]] = dict()
        if (not os.path.exists(self.data_path) and url):
            os.makedirs(os.path.join(*os.path.split(self.data_path)[:-1]), exist_ok=True)
            gdown.download(url, self.data_path)

        self._logger = logging.getLogger(self.__class__.__name__)
        console = logging.StreamHandler()
        self._logger.addHandler(console)
        self._logger.setLevel(logging.INFO)

        if (self.data_path.endswith(".jsonl.bz2")):
            self._logger.info(" Loading datafile...")
            self._source = pj.read_json(self.data_path, read_options=pj.ReadOptions(use_threads=True))
        else:
            self._source = None

        self._data: List[Sentence] = None
        self.tokenizer = English().tokenizer if (tokenizer is None) else tokenizer
        self.editing: bool = False

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return self._source.num_rows

    def __getitem__(self, item):
        if (self.editing):
            if (self._data is None):
                self.editing = False
                self._logger.info(f" Preloading {len(self)} entries for editing / indexing")
                self._data = [Sentence.from_dict(obj) for obj in tqdm(self._source.to_pylist(), desc="Preloading")]
                self.editing = True

            result = self._data[item]

        elif (isinstance(item, slice)):
            start, stop, step = item.indices(self._source.num_rows)
            indices = list(range(start, stop, step))
            result = [Sentence.from_dict(obj) for obj in self._source.take(indices).to_pylist()]
        else:
            index = item if (item >= 0) else self._source.num_rows + item
            result = Sentence.from_dict(self._source.take([index]).to_pylist()[0])

        for sent in (result if isinstance(result, list) else [result]):
            if (not sent.tokens):
                for tok in self.tokenizer(sent.surface):
                    token = Token()
                    token.surface = tok.text
                    sent.tokens.append(token)

        return result

    def edit(self):
        self.editing = True

    def vocabulary(self, source: str = "_token", lowercase: bool = True) -> Vocabulary:
        """
        Retrieves or creates a Vocabulary object for a specified annotation source.

        Args:
            source (str): The source (annotation) key for which the vocabulary should be retrieved or created (e.g., POS, NER).
            lowercase (bool): Whether to convert all tokens to lowercase in the vocabulary.

        Returns:
            Vocabulary: The vocabulary object for the specified annotation source.
        """
        if (source not in self._vocab):
            self._vocab[source] = Vocabulary(self, source=source, lowercase=lowercase)

        return self._vocab[source]

    def add_symbols(self, symbols: List[str], source: str = "_token"):
        """
        Adds symbols to the vocabulary for a specified annotation source.

        Args:
            symbols (List[str]): A list of symbols to add to the vocabulary.
            source (str): The source (annotation) key for which the symbols should be added.
        """
        vocab = self.vocabulary(source)
        vocab.add_symbols(symbols)

    def del_symbols(self, symbols: List[str], source: str = "_token"):
        """
        Deletes symbols from the vocabulary for a specified annotation source.

        Args:
            symbols (List[str]): A list of symbols to delete from the vocabulary.
            source (str): The source (annotation) key for which the symbols should be deleted.
        """
        vocab = self.vocabulary(source)
        vocab.del_symbols(symbols)

    @property
    def annotations(self) -> Dict[str, List[str]]:
        if (not self._annotations):
            annot = self[0].tokens[0].annotations
            for key in annot:
                self._annotations[key] = list()
                if (isinstance(annot[key], list)):
                    for sent in self:
                        for token in sent.tokens:
                            self._annotations[key].extend(set(token.annotations[key]))
                    self._annotations[key] = sorted(set(self._annotations[key]))
                else:
                    self._annotations[key] = sorted(self.vocabulary(key, lowercase=False).symbols)

        return self._annotations

    def to_indices(self, source: str = "_token", default: int = -1, padding: int = 0, pad_symbol: str = None,
                   start_symbol: str = None, end_symbol: str = None) -> Tensor:
        """
        Converts sentences to indices based on the vocabulary for a specified source.

        Args:
            source (str): The source (annotation) key for which the indices should be generated.
            default (int): The default index to use for tokens not found in the vocabulary.
            padding (int): The padding size to apply to the sequences.
            pad_symbol (str): The symbol used for padding.
            start_symbol (str): The symbol used to denote the start of a sequence.
            end_symbol (str): The symbol used to denote the end of a sequence.

        Returns:
            Tensor: A tensor of indices representing the sentences.
        """
        indices = self.vocabulary(source).to_indices(self, default, padding, pad_symbol, start_symbol, end_symbol)
        return torch.tensor(indices, dtype=torch.int32)

    def embeddings(self, tag: str, device: str = "cpu") -> Tensor:
        """
        Retrieves or computes embeddings for a specified type (tag) and stores them.

        Args:
            tag (str): The type for which embeddings should be retrieved or computed (e.g., w2v, electra, bert).
            device (str): The device in which the embeddings should be stored (e.g., 'cpu', 'cuda').

        Returns:
            Tensor: A tensor containing the embeddings for the specified tag.
        """
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

    def to_positional_indices(self, source: str = "_token", default: int = -1,
                              repetitions: int = 4, omit_default: bool = True) -> Tensor:
        """
        Converts sentences to positional indices based on the vocabulary for a specified annotation source.

        Args:
            source (str): The source (annotation) key for which the positional indices should be generated.
            default (int): The default index to use for tokens not found in the vocabulary.
            repetitions (int): The maximum number of repetitions for each token.
            omit_default (bool): Whether to omit the default index in the positional indices.

        Returns:
            Tensor: A sparse tensor of positional indices.
        """
        indices = self._vocab[source].to_indices(self, default, 0, None, None, None)
        pos_idx = list()
        pos_val = list()
        for i in range(len(indices)):
            rel_pos = torch.tensor(range(1, len(indices[i]) + 1)) / len(indices[i])
            rep_counter = Counter()
            for j in range(len(indices[i])):
                if (indices[i][j] != default or not omit_default):
                    rep_counter.update((indices[i][j],))
                    if (rep_counter[indices[i][j]] <= repetitions):
                        pos_idx.append([i, indices[i][j], rep_counter[indices[i][j]] - 1])
                        pos_val.append(rel_pos[j])

        return torch.sparse_coo_tensor(list(zip(*pos_idx)), pos_val, (len(indices), len(self.vocabulary()), repetitions)).mT.coalesce()

    def from_positional_indices(self, pos_indices: Tensor, source: str = "_token") -> List[List[str]]:
        """
        Converts positional indices back to sentences based on the vocabulary for a specified source.

        Args:
            pos_indices (Tensor): A tensor of positional indices.
            source (str): The source (annotation) key for which the sentences should be reconstructed.

        Returns:
            List[List[str]]: A list of str lists containing the reconstructed sentences.
        """
        vocab = self._vocab[source]
        sentences = list()
        for i in range(pos_indices.shape[0]):
            sorted, indices = torch.sort(pos_indices[i].to_dense().flatten())
            pos_relu = relu(sorted)
            indices = indices[pos_relu.nonzero()].flatten()
            sentences.append([vocab.get_symbol(k % len(vocab)) for k in indices.tolist()])

        return sentences

    def to_positional_encoding(self, source: str = "_token", default: int = -1, repetitions: int = 4, n: int = 1000,
                               omit_default: bool = True, offset: float = 1.0) -> Tuple[Tensor, Tensor]:
        """
        Converts sentences to positional encodings based on the vocabulary for a specified source.

        Args:
            source (str): The source (annotation) key for which the positional encodings should be generated.
            default (int): The default index to use for tokens not found in the vocabulary.
            repetitions (int): The number of positional encoding dimensions per token.
            n (int): The base of the positional encoding formula.
            omit_default (bool): Whether to omit the default index in the positional encodings.
            offset (float): The offset to apply to the positional encodings.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the positional encoding tensor and the positional encoding table.
        """
        indices = self._vocab[source].to_indices(self, default, 0, None, None, None)
        pos_idx = list()
        pos_val = list()
        max_len = max([len(idxs) for idxs in indices])
        pe_tbl = torch.zeros(max_len, repetitions)

        for i in range(max_len):
            for j in range(repetitions // 2):
                pe_tbl[i, 2 * j] = torch.sin(i / torch.tensor(pow(n, (2 * j) / repetitions)))
                pe_tbl[i, 2 * j + 1] = torch.cos(i / torch.tensor(pow(n, (2 * j) / repetitions)))

        for i in range(len(indices)):
            rep_counter = Counter()
            for j in range(len(indices[i])):
                if (indices[i][j] != default or not omit_default):
                    rep_counter.update((indices[i][j],))
                    if (rep_counter[indices[i][j]] <= repetitions):
                        pos_idx.append([i, indices[i][j], rep_counter[indices[i][j]] - 1])
                        pos_val.append(pe_tbl[j, rep_counter[indices[i][j]] - 1] + offset)

        penc = torch.sparse_coo_tensor(list(zip(*pos_idx)), pos_val, (len(indices), len(self._vocab[source]), repetitions))

        return penc.mT.coalesce(), pe_tbl

    def from_positional_encoding(self, pos_encoded: Tensor, pe_table: Tensor, offset: float = 1.0,
                                 source: str = "_token") -> List[List[str]]:
        """
        Converts positional encodings back to sentences based on the vocabulary for a specified source.

        Args:
            pos_encoded (Tensor): A tensor of positional encodings.
            pe_table (Tensor): The positional encoding table.
            offset (float): The offset applied to the positional encodings.
            source (str): The source (annotation) key for which the sentences should be reconstructed.

        Returns:
            List[List[str]]: A list of lists containing the reconstructed sentences.
        """
        vocab = self._vocab[source]
        sentences = list()
        for i in range(pos_encoded.shape[0]):
            penc = pos_encoded[i]
            indices = penc._indices().mT.tolist()
            sent = [None] * len(indices)
            for j, k in indices:
                if (penc[j, k] >= offset - 1):
                    dist = torch.abs(pe_table - penc[j, k])
                    pe_k, pe_i = tuple((dist == dist.max()).nonzero()[0].tolist())
                    sent[pe_k] = vocab.get_symbol(k)

            sentences.append([sym for sym in sent if (sym)])

        return sentences

    @staticmethod
    def download_resource(path: str, url: str):
        """
        Downloads a resource from a specified URL to a specified path

        This method is used to download original or preprocessed data that backs the SentenceDataset instances.

        Args:
            path (str): The subpath to the resource within the base path.
            url (str): The URL from which to download the resource.

        Returns:
            str: The normalized path where the resource has been downloaded.
        """
        data_path: str = os.path.normpath(os.path.join(str(Path.home()), BASE_PATH, path))
        if (not os.path.exists(data_path)):
            os.makedirs(os.path.join(*os.path.split(data_path)[:-1]), exist_ok=True)
            gdown.download(url, data_path)

        return data_path


class BasicSentenceDataSet(SentenceDataSet):
    def __init__(self, sentences: List[Sentence]):
        super().__init__("", "")
        self.data = sentences

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sentence:
        return self.data[idx]



