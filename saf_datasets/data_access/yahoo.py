import csv
import io
from zipfile import ZipFile
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

PATH = "Yahoo/yahoo_answers.zip"
URL = BASE_URL + "yahoo_answers.zip"


class YahooAnswersDataSet(SentenceDataSet):
    """
    Wrapper for the Yahoo Answers dataset (Zhang, Zhao, LeCun, 2015): https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(YahooAnswersDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

        with ZipFile(self.data_path) as dataset_file:
            self.data = list()
            for data_filename in dataset_file.namelist():
                if (not data_filename.endswith(".csv")):
                    continue

                split = data_filename.replace(".csv", "").split("/")[-1]

                with io.TextIOWrapper(dataset_file.open(data_filename), encoding="utf-8") as data_file:
                    reader = csv.reader(data_file)
                    i = 0
                    for row in tqdm(reader, desc=f"Loading SNLI [{split}]"):
                        category, title, question, answer = row
                        i += 1
                        for type, sent in {"title": title, "question": question, "answer": answer}.items():
                            sentence = Sentence()
                            sentence.annotations["id"] = i
                            sentence.annotations["split"] = split
                            sentence.annotations["category"] = category
                            sentence.annotations["type"] = type
                            sentence.surface = sent
                            for tok in self.tokenizer(sentence.surface):
                                token = Token()
                                token.surface = tok.text
                                sentence.tokens.append(token)

                            self.data.append(sentence)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sentence:
        """Fetches the ith definition in the dataset.

        Args:
            idx (int): index for the ith term in the dataset.

        :return: A single term definition (Sentence).
        """
        return self.data[idx]
