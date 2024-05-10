import os
import gdown
from typing import Dict, Iterable
from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile
from torch import argmax, as_tensor, cuda
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from saf import Sentence
from saf.annotators import Annotator
from .models import PRETRAINED_MODELS

BASE_PATH = ".saf_models"

class TransformerAnnotator(Annotator):
    """
    Annotator class for general transformer-based annotation (token classification) models

    Args:
        annot_model (str): the path or identifier for a pretrained model checkpoint.
        tag (str): the annotation tag to which the annotations will be stored (e.g., NER, SRL, DSR).
        labels (Dict[int, str]): valid labels and their corresponding values (ids).
        max_len (int): maximum sentence length.
        device (str): device (e.g., 'cpu', 'cuda') where the model will be stored and ran.
    """
    def __init__(self, annot_model: str, tag: str, labels: Dict[int, str] = None, max_len: int = 128, device: str = None):
        super(TransformerAnnotator, self).__init__()
        if (not device):
            if (cuda.is_available()):
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if (annot_model in PRETRAINED_MODELS):
            path = PRETRAINED_MODELS[annot_model]["path"]
            url = PRETRAINED_MODELS[annot_model]["url"]
            self.model_path: str = os.path.normpath(os.path.join(str(Path.home()), BASE_PATH, path))
            if (not os.path.exists(self.model_path)):
                os.makedirs(os.path.join(*os.path.split(self.model_path)[:-1]), exist_ok=True)
                gdown.download(url, self.model_path)
                ZipFile(self.model_path).extractall(self.model_path.replace(".zip", ""))
            annot_model = self.model_path.replace(".zip", "")

        self.annot_model = AutoModelForTokenClassification.from_pretrained(annot_model).to(self.device)
        self.annot_tokenizer = AutoTokenizer.from_pretrained(annot_model)
        self.annot_config = AutoConfig.from_pretrained(annot_model)
        self.ids_to_labels = labels if labels else self.annot_config.id2label
        self.max_len = max_len
        self.tag = tag

    def annotate(self, sentences: Iterable[Sentence]):
        for sent in tqdm(sentences, desc="Annotating (Transformer)"):
            #tag each definition
            sentence = sent.surface
            inputs = self.annot_tokenizer(sentence.strip().split(),
                                          is_split_into_words=True,
                                          return_offsets_mapping=True,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.max_len)
            item = {key: as_tensor(val) for key, val in inputs.items()}
            ids = item["input_ids"].unsqueeze(0).to(self.device)
            mask = item["attention_mask"].unsqueeze(0).to(self.device)
            # forward pass
            output = self.annot_model(input_ids=ids, attention_mask=mask)
            logits = output.logits
            active_logits = logits.view(-1, self.annot_model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level
            # mapping tags to words
            tokens = self.annot_tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [self.ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
            wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)
            prediction = []
            for token_pred, mapping in zip(wp_preds, item["offset_mapping"].squeeze().tolist()):
                #only predictions on first word pieces are considered
                if mapping[0] == 0 and mapping[1] != 0:
                    prediction.append(token_pred[1])
                else:
                    continue
            word_tags = [[word, label] for word, label in zip(sentence.split(), prediction)]
            #mapping back to the original tokens via string matching
            for i in range(len(sent.tokens)):
                if sent.tokens[i].surface.isalnum():
                    for j in range(len(word_tags)):
                        if sent.tokens[i].surface in word_tags[j][0]:
                            word_tags[j][0] = word_tags[j][0].replace(sent.tokens[i].surface,"")
                            sent.tokens[i].annotations[self.tag] = word_tags[j][1]
                            break
                else:
                    sent.tokens[i].annotations[self.tag] = "O"
