import spacy
from typing import Iterable
from tqdm import tqdm
from saf import Sentence
from saf.annotators import Annotator


class SpacyAnnotator(Annotator):
    def __init__(self, annot_model: str = "en_core_web_sm"):
        super(SpacyAnnotator, self).__init__()
        self.annot_model = spacy.load(annot_model)

    def annotate(self, definitions: Iterable[Sentence]):
        for defn in tqdm(definitions, desc="Annotating (Spacy)"):
            annots = self.annot_model(defn.surface)
            for i in range(len(defn.tokens)):
                defn.tokens[i].annotations["pos"] = annots[i].pos_
                defn.tokens[i].annotations["lemma"] = annots[i].lemma_
                defn.tokens[i].annotations["dep"] = annots[i].dep_
                defn.tokens[i].annotations["ctag"] = annots[i].tag_
