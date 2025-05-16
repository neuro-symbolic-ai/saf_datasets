import pickle
import gzip
from csv import DictReader
from tqdm import tqdm
from spacy.lang.en import English
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

FILE_VERSION = "inftypes_v0.3"
PATH = "InferenceTypes/%s.jsonl.bz2" % FILE_VERSION
URL = BASE_URL + "%s.jsonl.bz2" % FILE_VERSION

ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+amr": {
        "path": "InferenceTypes/inftypes_v0.3.jsonl.bz2",
        "url": BASE_URL + "inftypes_v0.3.jsonl.bz2"
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

    @classmethod
    def from_resource(cls, locator: str):
        """
        Downloads a pre-annotated resource available at the specified locator

        Example:
            >>> dataset = InferenceTypesDataSet.from_resource("pos+lemma+ctag+dep+amr")
        """
        dataset = None
        if (locator in ANNOT_RESOURCES):
            dataset = cls(**ANNOT_RESOURCES[locator])
        else:
            print(f"No resource found at locator: {locator}")

        return dataset

