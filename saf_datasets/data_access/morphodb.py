import json
import bz2
from saf import Sentence
from saf.importers.morphodb import MorphoDBImporter
from .dataset import SentenceDataSet

PATH = "morphodb/enmorphodb.json.bz2"
URL = "https://drive.google.com/uc?id=1s1Ic6KR-pZF7fdmyQxyg9OjuTn4AIfcX"


class MorphoDecompDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL, chars: bool = True):
        super(MorphoDecompDataSet, self).__init__(path, url)
        with bz2.open(self.data_path, "rb") as dataset_file:
            data = json.load(dataset_file)
        importer = MorphoDBImporter()
        self.data = importer.import_document(data, chars=chars).sentences

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Sentence:
        """Fetches the ith definition in the dataset or all definitions for a given term.

        Args:
            idx (int): index for the ith term in the dataset.

        :return: A single term decomposition (Sentence).
        """
        return self.data[idx]
