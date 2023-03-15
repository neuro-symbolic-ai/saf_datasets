import json
from typing import Dict, Iterable
from tqdm import tqdm
from torch import argmax, as_tensor, cuda
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from transformers import EarlyStoppingCallback, IntervalStrategy
from saf import Sentence
from saf.annotators import Annotator


class TransformerAnnotator(Annotator):
    def __init__(self, annot_model: str, labels: Dict[str, str], max_len: int = 128):
        super(TransformerAnnotator, self).__init__()
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.annot_model = AutoModelForTokenClassification.from_pretrained(annot_model).to(self.device)
        self.annot_tokenizer = AutoTokenizer.from_pretrained(annot_model)
        self.ids_to_labels = labels
        self.MAX_LEN = max_len

    def annotate(self, definitions: Iterable[Sentence]):
        for defn in tqdm(definitions, desc="Annotating (Transformer)"):
            #tag each definition
            sentence = defn.surface
            inputs = self.annot_tokenizer(sentence.strip().split(),
                                          is_split_into_words=True,
                                          return_offsets_mapping=True,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.MAX_LEN)
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
            token_predictions = [self.ids_to_labels[str(i)] for i in flattened_predictions.cpu().numpy()]
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
            for i in range(len(defn.tokens)):
                if defn.tokens[i].surface.isalnum():
                    for j in range(len(word_tags)):
                        if defn.tokens[i].surface in word_tags[j][0]:
                            word_tags[j][0] = word_tags[j][0].replace(defn.tokens[i].surface,"")
                            defn.tokens[i].annotations["dsr"] = word_tags[j][1]
                            break
                else:
                    defn.tokens[i].annotations["dsr"] = "O"