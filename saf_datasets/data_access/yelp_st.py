from zipfile import ZipFile
from tqdm import tqdm
from saf import Sentence, Token
from .dataset import SentenceDataSet, BASE_URL

PATH = "Yelp_ST/yelp_st.zip"
URL = BASE_URL + "yelp_st.zip"


class YelpSTDataSet(SentenceDataSet):
    """
    Wrapper for the Yelp Language Style Transfer dataset (Shen et al., 2017): https://proceedings.neurips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(YelpSTDataSet, self).__init__(path, url)

        with ZipFile(self.data_path) as dataset_file:
            self.data = list()
            for data_filename in dataset_file.namelist():
                _, split, style = data_filename.split(".")

                with dataset_file.open(data_filename) as data_file:
                    for line in tqdm(data_file, desc=f"Loading Yelp [{split}] [{style}]"):
                        sentence = Sentence()
                        sentence.annotations["split"] = split
                        sentence.annotations["style"] = style
                        sentence.surface = line.decode().strip()
                        for tok in sentence.surface.split():
                            token = Token()
                            token.surface = tok
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
