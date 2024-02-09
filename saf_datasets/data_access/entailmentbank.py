import os
import jsonlines
from zipfile import ZipFile
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet

FILE_VERSION = "entailment_trees_emnlp2021_data_v3"
PATH = "EntailmentBank/%s.zip" % FILE_VERSION
URL = "https://drive.google.com/uc?id=1laTMVEiR2n1VAEXu1aCq4ndpRqVnW1-R"


class EntailmentBankDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL):
        super(EntailmentBankDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

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

