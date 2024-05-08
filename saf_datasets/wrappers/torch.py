import torch
import torch.nn.functional as F
from typing import List, Union, Iterable
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from saf import Sentence


class TokenizedDataSet(Dataset):
    """
    A wrapper class for acessing a collection of Sentence objects or a list of strings as a torch.utils.data.Dataset


    Args:
        source (Union[Iterable[Sentence], List[str]]): a iterable of SAF Sentence objects (e.g., SentenceDataSet) or list of strings.
        tokenizer (PreTrainedTokenizer): tokenizer object to be used for tokenizing the sentences.
        max_len (int): maximum sentence length.
        one_hot (bool): whether the token ids will be one-hot encoded in the sentence tensors.
    """
    def __init__(self,  source: Union[Iterable[Sentence], List[str]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 one_hot: bool = False):
        self.source = source
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.one_hot = one_hot

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        sentences = self.source[idx]
        if (isinstance(idx, slice)):
            if (isinstance(self.source[0], Sentence)):
                sentences = [sent.surface for sent in sentences]
        else:
            if (isinstance(sentences, Sentence)):
                sentences = [sentences.surface]
            else:
                sentences = [sentences]

        if (not self.tokenizer.pad_token):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        tokenized = self.tokenizer(sentences, padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')

        if (self.one_hot):
            tokenized = F.one_hot(tokenized["input_ids"], num_classes=len(self.tokenizer.get_vocab())).to(torch.int8)
        else:
            tokenized = tokenized["input_ids"]

        return tokenized
