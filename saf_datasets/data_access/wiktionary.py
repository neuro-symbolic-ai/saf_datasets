import jsonlines
from typing import Tuple, List, Dict, Union
from zipfile import ZipFile
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token, Vocabulary
from saf_datasets.annotators.spacy import SpacyAnnotator
from .dataset import SentenceDataSet

PATH = "wiktionary/enwiktdb_sorted_en.jsonl.zip"
URL = "https://drive.google.com/uc?id=110xF4OGjUiwc7ONHq9L5C4znByamq4vv"


class WiktionaryDefinitionCorpus(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL, langs: Tuple[str] = ("English",)):
        super(WiktionaryDefinitionCorpus, self).__init__(path, url)
        wikt_zip = ZipFile(self.data_path)
        self._source = wikt_zip.open(wikt_zip.namelist()[0])
        self._size: int = 0
        self.langs: Tuple[str] = langs
        self._index: Dict[str, List[Sentence]] = dict()
        self._definitions: List[Sentence] = list()
        self.tokenizer = English().tokenizer

        for line in self._source:
            self._size += 1

        self._source.seek(0)

    def __iter__(self):
        if (self._index):
            return iter(self._definitions)
        else:
            self._source.seek(0)
            return WiktionaryDefinitionCorpusIterator(self)

    def __len__(self):
        return self._size

    def __getitem__(self, item) -> Union[Sentence, List[Sentence]]:
        """Fetches the ith definition in the corpus or all definitions for a given term.

        Args:
            item: (str) for all definitions of the givem term; (int) for the ith definition in the corpus.

        :return: A single definition (Sentence) or list of definitions.
        """
        self.load_index()
        definition = None
        if (isinstance(item, str)):
            if (item in self._index):
                definition = self._index[item]
            else:
                definition = []
        elif (isinstance(item, int)):
            definition = self._definitions[item]

        return definition

    def load_index(self):
        """Loads the corpus data and indexes the term definitions."""
        if not self._index:
            for definition in tqdm(self, desc="Loading data"):
                term = definition.annotations["definiendum"]
                if (term not in self._index):
                    self._index[term] = list()
                self._index[term].append(definition)
                self._definitions.append(definition)
            self._size = len(self._definitions)

    def vocabulary(self, source: str = "_token", lowercase: bool = True) -> Vocabulary:
        if (source not in self._vocab):
            self._vocab[source] = Vocabulary(self, source=source, lowercase=lowercase)
        if (source == "_token" or source == "lemma"):
            definiendum_vocab = Vocabulary(self, source="definiendum", lowercase=lowercase)
            self._vocab[source].add_symbols(list(definiendum_vocab.symbols))
            definiendum_vocab._vocab = self._vocab[source]._vocab
            definiendum_vocab.freqs = self._vocab[source].freqs
            self._vocab["definiendum"] = definiendum_vocab

        return self._vocab[source]


class WiktionaryDefinitionCorpusIterator:
    def __init__(self, wiktc: WiktionaryDefinitionCorpus):
        self._wiktc = wiktc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        with jsonlines.Reader(self._wiktc._source) as reader:
            for term in reader:
                for lang in self._wiktc.langs:
                    for pos in term["langs"][lang]["meanings"]:
                        for meaning in term["langs"][lang]["meanings"][pos]:
                            sentence = Sentence()
                            sentence.annotations["definiendum"] = term["title"].strip()
                            sentence.annotations["definition_pos"] = pos
                            definition = meaning["meaning"].replace("</text>", "").strip()
                            sentence.surface = definition
                            for tok in self._wiktc.tokenizer(definition):
                                token = Token()
                                token.surface = tok.text
                                sentence.tokens.append(token)

                            yield sentence


if __name__ == "__main__":
    defs = WiktionaryDefinitionCorpus()

    i = 0
    for sent in defs:
        print("Sent annotations:", sent.annotations)
        print("Token annotations:", [token.surface for token in sent.tokens])
        i += 1
        if (i > 10):
            break

    examples = defs["avocado"]
    annotator = SpacyAnnotator()
    annotator.annotate(examples)
    for example in examples:
        print(f"{example.annotations['definiendum']} ({example.annotations['definition_pos']}): ")
        print(f"{[(token.surface, token.annotations) for token in example.tokens]}")

    print("Corpus size:", len(defs))
