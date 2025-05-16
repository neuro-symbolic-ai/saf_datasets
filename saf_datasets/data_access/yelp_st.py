from .dataset import SentenceDataSet, BASE_URL

PATH = "Yelp_ST/yelp_st.jsonl.bz2"
URL = BASE_URL + "yelp_st.jsonl.bz2"


class YelpSTDataSet(SentenceDataSet):
    """
    Wrapper for the Yelp Language Style Transfer dataset (Shen et al., 2017): https://proceedings.neurips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html
    """
    def __init__(self, path: str = PATH, url: str = URL):
        super(YelpSTDataSet, self).__init__(path, url)
