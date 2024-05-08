import spacy
from typing import Iterable
from tqdm import tqdm
from saf import Sentence
from saf.annotators import Annotator


class SpacyAnnotator(Annotator):
    """
        Annotator class for part-of-speech (POS), lemmatization, constituency tree tagging,
        and dependency grammar tagging, based on the SpaCy library annotation models

        Args:
            annot_model (str): SpaCy annotation model to be used.
        """
    def __init__(self, annot_model: str = "en_core_web_sm"):
        super(SpacyAnnotator, self).__init__()
        self.annot_model = spacy.load(annot_model)

    def annotate(self, sentences: Iterable[Sentence]):
        for sent in tqdm(sentences, desc="Annotating (Spacy)"):
            annots = self.annot_model(sent.surface)
            for i in range(len(sent.tokens)):
                sent.tokens[i].annotations["pos"] = annots[i].pos_
                sent.tokens[i].annotations["lemma"] = annots[i].lemma_
                sent.tokens[i].annotations["dep"] = annots[i].dep_
                sent.tokens[i].annotations["ctag"] = annots[i].tag_
