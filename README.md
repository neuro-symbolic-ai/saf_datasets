# SAF-Datasets
### Dataset loading and annotation facilities for the Simple Annotation Framework

The *saf-datasets* library provides easy access to Natural Language Processing (NLP) datasets, and tools to facilitate annotation at document, sentence and token levels. 

It is being developed to address a need for flexibility in manipulating NLP annotations that is not entirely covered by popular dataset libraries, such as HuggingFace Datasets and torch Datasets, Namely:

- Including and modifying annotations on existing datasets.
- Standardized API.
- Support for complex and multi-level annotations.

*saf-datasets* is built upon the [Simple Annotation Framework (SAF)](https://github.com/dscarvalho/saf) library, which provides its data model and API.

It also provides annotator classes to automatically label existing and new datasets.


## Installation

To install, you can use pip:

```bash
pip install saf-datasets
```

## Usage
### Loading datasets

```python
from saf_datasets import STSBDataSet

dataset = STSBDataSet()
print(len(dataset))  # Size of the dataset
# 17256
print(dataset[0].surface)  # First sentence in the dataset
# A plane is taking off
print([token.surface for token in dataset[0].tokens])  # Tokens (SpaCy) of the first sentence.
# ['A', 'plane', 'is', 'taking', 'off', '.']
print(dataset[0].annotations)  # Annotations for the first sentence
# {'split': 'train', 'genre': 'main-captions', 'dataset': 'MSRvid', 'year': '2012test', 'sid': '0001', 'score': '5.000', 'id': 0}

# There are no token annotations in this dataset
print([(tok.surface, tok.annotations) for tok in dataset[0].tokens])
# [('A', {}), ('plane', {}), ('is', {}), ('taking', {}), ('off', {}), ('.', {})]
```

**Available datasets:** AllNLI, CODWOE, CPAE, EntailmentBank, STSB, Wiktionary, WordNet (Filtered).

### Annotating datasets

```python
from saf_datasets import STSBDataSet
from saf_datasets.annotators import SpacyAnnotator

dataset = STSBDataSet()
annotator = SpacyAnnotator()  # Needs spacy and en_core_web_sm to be installed.
annotator.annotate(dataset)

# Now tokens are annotated
for tok in dataset[0].tokens:
    print(tok.surface, tok.annotations)

# A {'pos': 'DET', 'lemma': 'a', 'dep': 'det', 'ctag': 'DT'}
# plane {'pos': 'NOUN', 'lemma': 'plane', 'dep': 'nsubj', 'ctag': 'NN'}
# is {'pos': 'AUX', 'lemma': 'be', 'dep': 'aux', 'ctag': 'VBZ'}
# taking {'pos': 'VERB', 'lemma': 'take', 'dep': 'ROOT', 'ctag': 'VBG'}
# off {'pos': 'ADP', 'lemma': 'off', 'dep': 'prt', 'ctag': 'RP'}
# . {'pos': 'PUNCT', 'lemma': '.', 'dep': 'punct', 'ctag': '.'}
```

### Using with other libraries

*saf-datasets* provides wrappers for using the datasets with libraries expecting HF or torch datasets:

```python
from saf_datasets import CPAEDataSet
from saf_datasets.wrappers.torch import TokenizedDataSet
from transformers import AutoTokenizer

dataset = CPAEDataSet()
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left", add_prefix_space=True)
tok_ds = TokenizedDataSet(dataset, tokenizer, max_len=128, one_hot=False)
print(tok_ds[:10])
# tensor([[50256, 50256, 50256,  ...,  2263,   572,    13],
#         [50256, 50256, 50256,  ...,  2263,   572,    13],
#         [50256, 50256, 50256,  ...,   781,  1133,    13],
#         ...,
#         [50256, 50256, 50256,  ...,  2712, 19780,    13],
#         [50256, 50256, 50256,  ...,  2685,    78,    13],
#         [50256, 50256, 50256,  ...,  2685,    78,    13]])

print(tok_ds[:10].shape)
# torch.Size([10, 128])
```