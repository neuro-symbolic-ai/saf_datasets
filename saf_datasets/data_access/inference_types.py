import pickle
import gzip
from csv import DictReader
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

FILE_VERSION = "tr_data_type_amr_op_v0.2"
PATH = "InferenceTypes/%s.csv.gz" % FILE_VERSION
URL = BASE_URL + "%s.csv.gz" % FILE_VERSION

ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+amr": {
        "path": "InferenceTypes/inftypes_v0.2.pickle.gz",
        "url": BASE_URL + "inftypes_v0.2.pickle.gz"
    }
}


class InferenceTypesDataSet(SentenceDataSet):
    """
    Wrapper for the Inference Types dataset, an annotated subset of
    the EntailmentBank: https://allenai.org/data/entailmentbank

    Premises and conclusion sentences for a single entry in the original dataset are split
    adjacently, and can be grouped by their 'id' annotation.

    Sentence annotations: id, role, type, new_type, type_amr_op
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(InferenceTypesDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer
        if (not url):
            return

        with gzip.open(self.data_path, "rt", encoding="utf-8") as dataset_file:
            self.data = list()
            for row in tqdm(DictReader(dataset_file), desc="Loading inference types data"):
                premise1 = row["premise1"]
                premise2 = row["premise2"]
                conclusion = row["conclusion"]
                for sent, role in [(premise1, "P1"), (premise2, "P2"), (conclusion, "C")]:
                    sentence = Sentence()
                    sentence.annotations["id"] = row["id"]
                    sentence.annotations["role"] = role
                    sentence.annotations["type"] = row["type"]
                    sentence.annotations["new_type"] = row["new_type"]
                    sentence.annotations["type_amr_op"] = row["type_amr_op"]
                    sentence.surface = sent.strip()
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
        """Fetches the ith sentence in the dataset.

        Args:
            idx (int): index for the ith sentence in the dataset.

        :return: A single term decomposition (Sentence).
        """
        return self.data[idx]

    @staticmethod
    def from_resource(locator: str):
        """
        Downloads a pre-annotated resource available at the specified locator

        Example:
            >>> dataset = InferenceTypesDataSet.from_resource("pos+lemma+ctag+dep+amr")
        """
        dataset = None
        if (locator in ANNOT_RESOURCES):
            path = ANNOT_RESOURCES[locator]["path"]
            url = ANNOT_RESOURCES[locator]["url"]
            data_path = SentenceDataSet.download_resource(path, url)
            with gzip.open(data_path, "rb") as resource_file:
                dataset = pickle.load(resource_file)
        else:
            print(f"No resource found at locator: {locator}")

        return dataset

