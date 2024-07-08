import jsonlines
import pickle
import gzip
from typing import Tuple, List, Dict, Union
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token, Vocabulary
from saf_datasets.annotators.spacy import SpacyAnnotator
from .dataset import SentenceDataSet, BASE_URL

PATH = "wiktionary/raw-wiktextract-data_2024-01-20.jsonl.gz"  # From https://kaikki.org/dictionary/rawdata.html
URL = "https://drive.google.com/uc?id=1ZMZe25lhjl14p0kEJVxedwmMZV2KUBrz"
ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+dsr": {
        "path": "wiktionary/wikdef_spacy_dsr.pickle.gz",
        "url": BASE_URL + "wikdef_spacy_dsr.pickle.gz"
    },
    "pos+lemma+ctag+dep+dsr#sample": {
        "path": "wiktionary/wikdef_spacy_dsr_sample.pickle.gz",
        "url": BASE_URL + "wikdef_spacy_dsr_sample.pickle.gz"
    }
}


class WiktionaryDefinitionCorpus(SentenceDataSet):
    """
    Wiktionary definition corpus: A collection of definition sentences from Wiktionary

    Each element of this dataset is a definition sentence (gloss) of a single sense, as it appears in Wiktionary.
    Each sentence is annotated with a "definiendum" (the term being defined) and its corresponding part-of-speech (definition_pos).

    A pre-annotated instance can be obtained as follows:

    >>> dataset = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")

    """
    def __init__(self, path: str = PATH, url: str = URL, langs: Tuple[str] = ("English",)):
        super(WiktionaryDefinitionCorpus, self).__init__(path, url)

        self._size: int = -1
        self.langs: Tuple[str] = langs
        self._index: Dict[str, List[Sentence]] = dict()
        self._definitions: List[Sentence] = list()
        self.tokenizer = English().tokenizer

        if (not url):
            return

        self._source = gzip.open(self.data_path)
        self._source.seek(0)

    def __iter__(self):
        if (self._index):
            return iter(self._definitions)
        else:
            self._source.seek(0)
            return WiktionaryDefinitionCorpusIterator(self)

    def __len__(self):
        if (self._size == -1):
            with jsonlines.Reader(self._source) as reader:
                for term in tqdm(reader, desc="Checking data"):
                    if ("word" in term and term["lang"] in self.langs):
                        self._size += 1

            self._source.seek(0)

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
        elif (isinstance(item, int) or isinstance(item, slice)):
            definition = self._definitions[item]

        return definition

    def load_index(self):
        """Loads the corpus data and indexes the term definitions."""
        if not self._index:
            self._size = 0
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
            self._vocab[source].freqs.update(definiendum_vocab.freqs)
            definiendum_vocab._vocab = self._vocab[source]._vocab
            definiendum_vocab.freqs = self._vocab[source].freqs
            self._vocab["definiendum"] = definiendum_vocab

        return self._vocab[source]

    @staticmethod
    def from_resource(locator: str):
        """
        Downloads a pre-annotated resource available at the specified locator

        Example:
            >>> dataset = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")
        """
        wiktdef = None
        if (locator in ANNOT_RESOURCES):
            path = ANNOT_RESOURCES[locator]["path"]
            url = ANNOT_RESOURCES[locator]["url"]
            data_path = WiktionaryDefinitionCorpus.download_resource(path, url)
            with gzip.open(data_path, "rb") as resource_file:
                data = pickle.load(resource_file)

            wiktdef = WiktionaryDefinitionCorpus(url="")
            for definition in tqdm(data, desc=f"Loading data from resource: {locator}"):
                term = definition.annotations["definiendum"]
                if (term not in wiktdef._index):
                    wiktdef._index[term] = list()
                wiktdef._index[term].append(definition)
                wiktdef._definitions.append(definition)
            wiktdef._size = len(wiktdef._definitions)

        return wiktdef


class WiktionaryDefinitionCorpusIterator:
    def __init__(self, wiktc: WiktionaryDefinitionCorpus):
        self._wiktc = wiktc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        with jsonlines.Reader(self._wiktc._source) as reader:
            for term in reader:
                if ("word" in term and term["lang"] in self._wiktc.langs):
                    for sense in term["senses"]:
                        if ("glosses" in sense):
                            for gloss in sense["glosses"]:
                                sentence = Sentence()
                                sentence.annotations["definiendum"] = term["word"]
                                sentence.annotations["definition_pos"] = term["pos"]
                                if ("tags" in sense):
                                    sentence.annotations["tags"] = sense["tags"]
                                # if ("categories" in sense):
                                #     sentence.annotations["categories"] = [ct for ct in sense["categories"] if ":" in ct]
                                definition = gloss
                                sentence.surface = definition
                                for tok in self._wiktc.tokenizer(definition):
                                    token = Token()
                                    token.surface = tok.text
                                    sentence.tokens.append(token)

                                if (len(sentence.tokens) == 0):
                                    continue

                                yield sentence

                # for lang in self._wiktc.langs:
                #     for pos in term["langs"][lang]["meanings"]:
                #         for meaning in term["langs"][lang]["meanings"][pos]:
                #             sentence = Sentence()
                #             sentence.annotations["definiendum"] = term["title"].strip()
                #             sentence.annotations["definition_pos"] = pos
                #             definition = meaning["meaning"].replace("</text>", "").strip()
                #             sentence.surface = definition
                #             for tok in self._wiktc.tokenizer(definition):
                #                 token = Token()
                #                 token.surface = tok.text
                #                 sentence.tokens.append(token)
                #
                #             if (len(sentence.tokens) == 0):
                #                 continue
                #
                #             yield sentence


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
