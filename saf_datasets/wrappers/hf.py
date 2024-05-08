import datasets
from saf_datasets.data_access.dataset import SentenceDataSet


def gen(dataset: SentenceDataSet):
    for sent in dataset:
        entry = {"tokens": list()}
        for token in sent.tokens:
            entry["tokens"].append(token.surface)
            for annot in token.annotations:
                if (annot not in entry):
                    entry[annot] = list()
                entry[annot].append(token.annotations[annot])
        for annot in sent.annotations:
            entry[f"s:{annot}"] = sent.annotations[annot]

        yield entry


def to_hf(dataset: SentenceDataSet) -> datasets.Dataset:
    """Returns a HuggingFace datasets.Dataset wrapper for the given SentenceDataSet"""
    return datasets.Dataset.from_generator(gen, gen_kwargs={"dataset": dataset})

