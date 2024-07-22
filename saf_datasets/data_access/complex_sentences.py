import json
from zipfile import ZipFile
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

FILE_VERSION = "complex_sentences"
PATH = "ComplexSentences/%s.zip" % FILE_VERSION
URL = BASE_URL + "%s.zip" % FILE_VERSION


class ComplexSentencesDataSet(SentenceDataSet):
    """
    Wrapper for the Complex Sentences dataset (Niklaus et al.): https://aclanthology.org/P19-1333/

    Sentence annotations: source ["wiki", "newsela"], split
    Token annotations: label (see `ComplexSentencesDataSet.LABELS`)
    """

    LABELS = {
        "CNP": "coordinate noun phrases",
        "CVP":" coordinate verb phrases",
        "PreP": "prepositional phrase",
        "ParP": "participial phrase",
        "AP": "appositive phrase",
        "AM": "adverbial modifier",
        "CC": "coordinate clauses",
        "AC": "adverbial clause",
        "RC": "relative clause",
        "IDC": "(in)direct speech"
    }

    def __init__(self, path: str = PATH, url: str = URL):
        super(ComplexSentencesDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer
        if (not url):
            return

        with ZipFile(self.data_path) as dataset_file:
            self.data = list()
            for data_filename in dataset_file.namelist():
                path_split = data_filename.split("/")
                if (len(path_split) < 3 or path_split[-1] == ""):
                    continue
                pkg, source, split = path_split
                with dataset_file.open(data_filename) as data_file:
                    data = json.load(data_file)

                    for entry in tqdm(data, desc=f"Loading Complex Sentences: {source} -- {split}"):
                        sentence = Sentence()
                        sentence.annotations["source"] = source
                        sentence.annotations["split"] = split.split(".")[0]
                        sentence.surface = entry[0]

                        for tok, label in zip(sentence.surface.split(), entry[1].split()):
                            token = Token()
                            token.surface = tok
                            token.annotations["label"] = label
                            sentence.tokens.append(token)

                        if (sentence.surface):
                            self.data.append(sentence)

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



