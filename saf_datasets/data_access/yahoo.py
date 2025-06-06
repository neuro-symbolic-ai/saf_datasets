from .dataset import SentenceDataSet, BASE_URL

PATH = "Yahoo/yahoo_answers.jsonl.bz2"
URL = BASE_URL + "yahoo_answers.jsonl.bz2"


class YahooAnswersDataSet(SentenceDataSet):
    """
    Wrapper for the Yahoo Answers dataset (Zhang, Zhao, LeCun, 2015): https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html
    """
    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        super(YahooAnswersDataSet, self).__init__(path, url, **kwargs)
