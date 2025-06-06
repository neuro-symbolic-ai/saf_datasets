import torch
from torch import Tensor
from saf import Sentence
from .dataset import SentenceDataSet, BASE_URL

PATH = "CODWOE/CODWOE.jsonl.bz2"
URL = BASE_URL + "CODWOE.jsonl.bz2"


class CODWOEDataSet(SentenceDataSet):
    def __init__(self, path: str = PATH, url: str = URL, **kwargs):
        """
        Wrapper for the CODWOE dataset from Semeval-2022 Task 1 (Mickus et al. 2022): https://github.com/TimotheeMickus/codwoe
        """
        super(CODWOEDataSet, self).__init__(path, url, **kwargs)

    def __getitem__(self, idx: int) -> Sentence:
        """Fetches the ith definition in the dataset or all definitions for a given term.

        :param item: (int) for the ith definition in the dataset.
        :return: A single definition (Sentence).
        """
        sent = super().__getitem__(idx)
        sent.annotations["emb_char"] = torch.tensor(sent.annotations["emb_char"])
        sent.annotations["emb_electra"] = torch.tensor(sent.annotations["emb_electra"])
        sent.annotations["emb_sgns"] = torch.tensor(sent.annotations["emb_sgns"])

        return sent

    def embeddings(self, tag: str, device: str = "cpu") -> Tensor:
        embeddings = None
        annots = [self[i].annotations for i in range(len(self))]
        if (tag == "char"):
            embeddings = torch.stack([annot["emb_char"] for annot in annots])
        elif (tag == "electra"):
            embeddings = torch.stack([annot["emb_electra"] for annot in annots])
        elif (tag == "sgns"):
            embeddings = torch.stack([annot["emb_sgns"] for annot in annots])
        else:
            embeddings = super(CODWOEDataSet, self).embeddings(tag, device)

        return embeddings.to(device)


if __name__ == "__main__":
    defs = CODWOEDataSet()

    print(defs.vocabulary().freqs.most_common())

    exit(0)

    i = 0
    for sent in defs:
        print("Sent annotations:", sent.annotations)
        print("Token annotations:", [token.surface for token in sent.tokens])
        i += 1
        if (i > 10):
            break

    examples = [defs[5], defs[8]]
    annotator = SpacyAnnotator()
    annotator.annotate(examples)
    for example in examples:
        print(f"{example.annotations['id']} (\"{example.surface}\"): ")
        print(f"{[(token.surface, token.annotations) for token in example.tokens]}")

    print("Corpus size:", len(defs))
