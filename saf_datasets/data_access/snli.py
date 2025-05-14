import jsonlines
from zipfile import ZipFile
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

PATH = "SNLI/snli_1.0.zip"
URL = BASE_URL + "snli_1.0.zip"


class SNLIDataSet(SentenceDataSet):
    """
    Wrapper for the Yelp Language Style Transfer dataset (Shen et al., 2017): https://proceedings.neurips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(SNLIDataSet, self).__init__(path, url)
        self.tokenizer = English().tokenizer

        with ZipFile(self.data_path) as dataset_file:
            self.data = list()
            for data_filename in dataset_file.namelist():
                if (not data_filename.endswith(".jsonl")):
                    continue

                split = data_filename.replace(".jsonl", "").split("_")[2]

                with dataset_file.open(data_filename) as data_file:
                    reader = jsonlines.Reader(data_file)
                    for row in tqdm(reader, desc=f"Loading SNLI [{split}]"):
                        for sent_idx in ["sentence1", "sentence2"]:
                            sentence = Sentence()
                            sentence.annotations["split"] = split
                            sentence.annotations["annotator_labels"] = row["annotator_labels"]
                            sentence.annotations["captionID"] = row["captionID"]
                            sentence.annotations["gold_label"] = row["gold_label"]
                            sentence.annotations["pairID"] = row["pairID"]
                            # sentence.annotations[f"{sent_idx}_binary_parse"] = row[f"{sent_idx}_binary_parse"]
                            # sentence.annotations[f"{sent_idx}_parse"] = row[f"{sent_idx}_parse"]
                            sentence.surface = row[sent_idx]
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
