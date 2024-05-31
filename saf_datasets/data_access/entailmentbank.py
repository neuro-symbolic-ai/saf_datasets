import os
import pickle
import jsonlines
import gzip
from zipfile import ZipFile
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

FILE_VERSION = "entailment_trees_emnlp2021_data_v3"
PATH = "EntailmentBank/%s.zip" % FILE_VERSION
URL = BASE_URL + "%s.zip" % FILE_VERSION

ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+srl#noproof": {
        "path": "EntailmentBank/eb_spacy_srl_noproof.pickle.gz",
        "url": BASE_URL + "eb_spacy_srl_noproof.pickle.gz"
    }
}


class EntailmentBankDataSet(SentenceDataSet):
    """
    Wrapper for the EntailmentBank dataset: https://allenai.org/data/entailmentbank

    Context, hypothesis, question, answer and proof sentences for a single entry in the original dataset are split
    adjacently, and can be grouped by their 'id' annotation.

    Sentence annotations: id, task, split, type
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(EntailmentBankDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer
        if (not url):
            return

        with ZipFile(self.data_path) as dataset_file:
            self.data = list()
            for task in ["task_1", "task_2", "task_3"]:
                for split in ["train", "dev", "test"]:
                    split_file = dataset_file.open(os.path.join(FILE_VERSION, "dataset", task, split + ".jsonl"))
                    reader = jsonlines.Reader(split_file)
                    features = dict()
                    for entry in tqdm(reader, desc=f"Loading EntailmentBank: {task} -- {split}"):
                        for key in ["question", "answer", "hypothesis", "proof", "full_text_proof"]:
                            if (key in entry):
                                features[key] = entry[key]
                        for key in entry["meta"]["triples"]:
                            features["context:" + key] = entry["meta"]["triples"][key]
                        for feat in features:
                            sentence = Sentence()
                            sentence.annotations["task"] = task
                            sentence.annotations["split"] = split
                            sentence.annotations["id"] = entry["id"]
                            sentence.annotations["type"] = feat
                            sentence.surface = features[feat] if features[feat] else ""
                            for tok in self.tokenizer(sentence.surface):
                                token = Token()
                                token.surface = tok.text
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

    @staticmethod
    def from_resource(locator: str):
        """
        Downloads a pre-annotated resource available at the specified locator

        Example:
            >>> dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#noproof")
        """
        dataset = None
        if (locator in ANNOT_RESOURCES):
            path = ANNOT_RESOURCES[locator]["path"]
            url = ANNOT_RESOURCES[locator]["url"]
            data_path = SentenceDataSet.download_resource(path, url)
            dataset = EntailmentBankDataSet(url="")
            with gzip.open(data_path, "rb") as resource_file:
                dataset.data = pickle.load(resource_file)
        else:
            print(f"No resource found at locator: {locator}")

        return dataset

