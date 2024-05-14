from typing import Iterable
from tqdm import tqdm
from allennlp_models import pretrained
from saf import Sentence
from saf.annotators import Annotator


class AllenSRLAnnotator(Annotator):
    """
    Annotator class for Semantic Role Labelling, based on the AllenNLP SRL model (Argument Structure)
    """
    def __init__(self):
        super(AllenSRLAnnotator, self).__init__()
        self.annot_model = pretrained.load_predictor("structured-prediction-srl-bert")

    def annotate(self, sentences: Iterable[Sentence]):
        for sent in tqdm(sentences, desc="Annotating (AllenNLP SRL)"):
            annots = self.annot_model.predict(sent.surface)
            sent.tokens = [tok for tok in sent.tokens if (tok.surface.strip())]
            for i in range(len(sent.tokens)):
                sent.tokens[i].annotations["srl"] = list()
                if (len(annots["verbs"]) > 0):
                    for v_idx in range(len(annots["verbs"])):
                        # print("V_ANNOTS:", annots["verbs"][v_idx])
                        # print([tok.surface for tok in sent.tokens])
                        # print(annots["words"])
                        sent.tokens[i].annotations["srl"].append(annots["verbs"][v_idx]["tags"][i])
                else:
                    sent.tokens[i].annotations["srl"].append("O")
