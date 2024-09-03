import gzip
import jsonlines
from tqdm import tqdm
from saf import Sentence, Vocabulary
from saf import Token
from .dataset import SentenceDataSet, BASE_URL

PATH = "/Users/user/Desktop/Year 2/saf_datasets/saf_datasets/data_access/improved_definitions_combined_data_15_39_48_49_57_annotated.jsonl.gz"
URL = "/Users/user/Desktop/Year 2/saf_datasets/saf_datasets/data_access/improved_definitions_combined_data_15_39_48_49_57_annotated.jsonl.gz"


# ANNOT_RESOURCES = {
#     "pos+lemma+ctag+dep+dsr+srl": {
#         "path": "WordNet/wordnet_filtered_spacy_dsr_srl.pickle.gz",
#         "url": BASE_URL + "wordnet_filtered_spacy_dsr_srl.pickle.gz"
#     }
# }


class WordNetFormalisedDataSet():
    '''
    Wordnet filtered data set: A collection of formalised definition sentences from WordNet
    Each element of this dataset is a definition sentence (gloss) of a single synset, as it appears in WordNet.
    '''
    def __init__(self, path: str = PATH, url: str = URL):
        super(WordNetFormalisedDataSet, self).__init__()
        with gzip.open(path, 'rt') as dataset_file:
            with jsonlines.Reader(dataset_file) as reader:
                self.data = list()
                for line in tqdm(reader, desc="Loading WordNet formalised data"):
                    sentence = Sentence()
                    sentence.annotations['id'] = line['id']
                    sentence.annotations['definiendum'] = line['name']
                    sentence.annotations['features'] = line['features']
                    sentence.annotations['hypernym'] = line['hypernym']
                    sentence.surface = line['definition']
                    for tok in line["definition_attributes"]:
                        token = Token()
                        token.surface = tok['name']
                        token.annotations = tok
                        del token.annotations['name']
                        del token.annotations['definition']
                        sentence.tokens.append(token)

                    self.data.append(sentence)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sentence:
        return self.data[idx]



# w = WordNetFormalisedDataSet()