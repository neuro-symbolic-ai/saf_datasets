import gzip
import jsonlines
from tqdm import tqdm
from saf import Sentence, Vocabulary
from .dataset import SentenceDataSet, BASE_URL

PATH = "/Users/user/Desktop/Code/Definitions and concepts/concepts_formalisation/data/sample_concept/48/improved_definitions_combined_data_15_39_48_49_57_.jsonl.gz"
URL = "/Users/user/Desktop/Code/Definitions and concepts/concepts_formalisation/data/sample_concept/48/improved_definitions_combined_data_15_39_48_49_57_.jsonl.gz"


ANNOT_RESOURCES = {
    "pos+lemma+ctag+dep+dsr+srl": {
        "path": "WordNet/wordnet_filtered_spacy_dsr_srl.pickle.gz",
        "url": BASE_URL + "wordnet_filtered_spacy_dsr_srl.pickle.gz"
    }
}


class WordNetFormalisedDataSet(SentenceDataSet):
    '''
    Wordnet filtered data set: A collection of formalised definition sentences from WordNet
    Each element of this dataset is a definition sentence (gloss) of a single synset, as it appears in WordNet.
    '''
    def __init__(self, path: str = PATH, url: str = URL):
        super(WordNetFormalisedDataSet, self).__init__(path, url)
        with gzip.open(self.data_path, 'rt') as dataset_file:
            print('path', self.data_path)
            with jsonlines.Reader(dataset_file) as reader:
                self.data = list()
                for line in tqdm(reader, desc="Loading WordNet formalised data"):
                    sentence = Sentence()
                    sentence.annotations['id'] = line['id']
                    sentence.annotations['definiendum'] = line['definiendum']
                    sentence.annotations['features'] = line['definition']
                    sentence.annotations['hypernym'] = line['hypernym']
                    sentence.surface = self.format_definition(line['features'])
                    self.data.append(sentence)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sentence:
        return self.data[idx]

    def format_definition(self, characteristics, conjunction='AND', prompt='A term that {char_string} is called \"') -> str:
        formatted_chars = [f"{char['predicate']} {char['object']}" for char in characteristics]
        if conjunction == 'AND':
            char_string = " AND ".join(formatted_chars[:-1])
        else:
            char_string = ", ".join(formatted_chars[:-1])

        # Add the last characteristic with "and"
        if len(formatted_chars) > 1:
            char_string += f" AND {formatted_chars[-1]}"
        else:
            char_string = formatted_chars[0]

        # Create the full prompt
        prompt = f"{prompt}{char_string}\""

        return prompt


# w = WordNetFormalisedDataSet()