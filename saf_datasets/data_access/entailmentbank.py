from .dataset import SentenceDataSet, BASE_URL

FILE_VERSION = "entailment_trees_emnlp2021_data_v3"
PATH = "EntailmentBank/%s.jsonl.bz2" % FILE_VERSION
URL = BASE_URL + "%s.jsonl.bz2" % FILE_VERSION

ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+srl#expl_only-noreps": {
        "path": "EntailmentBank/eb_spacy_srl_explonly_noreps.jsonl.bz2",
        "url": BASE_URL + "eb_spacy_srl_explonly_noreps.jsonl.bz2"
    },
    "explanations+srl": {
        "path": "EntailmentBank/explanations.jsonl.bz2",
        "url": BASE_URL + "explanations.jsonl.bz2"
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

    @classmethod
    def from_resource(cls, locator: str):
        """
        Downloads a pre-annotated resource available at the specified locator

        Example:
            >>> dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")
        """
        dataset = None
        if (locator in ANNOT_RESOURCES):
            dataset = cls(**ANNOT_RESOURCES[locator])
        else:
            print(f"No resource found at locator: {locator}")

        return dataset

