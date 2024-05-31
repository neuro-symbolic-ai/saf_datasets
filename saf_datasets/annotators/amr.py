import os
import tarfile
import amrlib
import gdown
from typing import Iterable
from saf import Sentence
from saf.annotators import Annotator

AMRLIB_MODELS_BASE_URL = "https://github.com/bjascob/amrlib-models/releases/download/"
AMRLIB_MODELS = {
    "parse_xfm_bart_large": "parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz",
    "parse_xfm_bart_base": "parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz",
    "parse_spring": "model_parse_spring-v0_1_0/model_parse_spring-v0_1_0.tar.gz",
    "parse_t5": "model_parse_t5-v0_2_0/model_parse_t5-v0_2_0.tar.gz"
}


class AMRAnnotator(Annotator):
    """
        Annotator class for Abstract Meaning Representation (AMR), based on the amrlib library annotation models

        Args:
            annot_model (str): amrlib annotation model to be used. See `https://github.com/bjascob/amrlib-models`
        """
    def __init__(self, annot_model: str = "parse_xfm_bart_large"):
        super(AMRAnnotator, self).__init__()
        amrlib_path = amrlib.__file__.replace("__init__.py", "")
        model_path = os.path.join(amrlib_path, "data", "model_stog")

        if (not os.path.exists(model_path)):
            os.makedirs(os.path.join(amrlib_path, "data"), exist_ok=True)
            model_filename = AMRLIB_MODELS[annot_model].split("/")[-1]
            model_filepath = os.path.join(amrlib_path, "data", model_filename)
            gdown.download(AMRLIB_MODELS_BASE_URL + AMRLIB_MODELS[annot_model], model_filepath, resume=True)
            model_file = tarfile.open(model_filepath, "r:gz")
            model_file.extractall(os.path.join(amrlib_path, "data"))
            os.replace(os.path.join(amrlib_path, "data", model_filename.split(".")[0]),
                       os.path.join(amrlib_path, "data", "model_stog"))

        self.annot_model = amrlib.load_stog_model()

    def annotate(self, sentences: Iterable[Sentence]):
        graphs = self.annot_model.parse_sents([sent.surface for sent in sentences], disable_progress=False)
        for sent, graph in zip(sentences, graphs):
            sent.annotations["amr"] = graph
