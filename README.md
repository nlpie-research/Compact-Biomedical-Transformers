# Compact Biomedical Transformers
<p align="justify"> This repository contains the code used for distillation and fine-tuning of compact biomedical transformers that have been introduced in the paper "On The Effectiveness of Compact Biomedical Transformers". </p>

# Abstract of the Research
<p align="justify"> Language models pre-trained on biomedical corpora, such as BioBERT, have re- cently shown promising results on downstream biomedical tasks. Many existing pre-trained models, on the other hand, are resource-intensive and computationally heavy owing to factors such as embedding size, hidden dimension, and number of layers. The natural language processing (NLP) community has developed nu- merous strategies to compress these models utilising techniques such as pruning, quantisation, and knowledge distillation, resulting in models that are considerably faster, smaller, and subsequently easier to use in practice. By the same token, in this paper we introduce six lightweight models, namely, BioDistilBERT, BioTiny- BERT, BioMobileBERT, DistilBioBERT, TinyBioBERT, and CompactBioBERT which are obtained either by knowledge distillation from a biomedical teacher or continual learning on the Pubmed dataset via the Masked Language Modelling (MLM) objective. We evaluate all of our models on three biomedical tasks and compare them with BioBERT-v1.1 to create efficient lightweight models that per- form on par with their larger counterparts. </p>

# Available Models
| Model Name  | isDistiiled | #Layers  | #Params | Huggingface Path | Huggingface Link |
| :------------ |:------------ | :------------: | :------------: | :------------ | :------------:|
| DistilBioBERT         | True  | 6  | 65M |  nlpie/distil-biobert         | [here](https://huggingface.co/nlpie/distil-biobert)         |
| CompactBioBERT        | True  | 6  | 65M |  nlpie/compact-biobert        | [here](https://huggingface.co/nlpie/compact-biobert)        |
| TinyBioBERT           | True  | 4  | 15M |  nlpie/tiny-biobert           | [here](https://huggingface.co/nlpie/tiny-biobert)           |
| BioDistilBERT-cased   | False | 6  | 65M |  nlpie/bio-distilbert-cased   | [here](https://huggingface.co/nlpie/bio-distilbert-cased)   |
| BioDistilBERT-uncased | False | 6  | 65M |  nlpie/bio-distilbert-uncased | [here](https://huggingface.co/nlpie/bio-distilbert-uncased) |
| BioTinyBERT           | False | 4  | 15M |  nlpie/bio-tinybert           | [here](https://huggingface.co/nlpie/bio-tinybert)           |
| BioMobileBERT         | False | 24 | 25M |  nlpie/bio-mobilebert         | [here](https://huggingface.co/nlpie/bio-mobilebert)         |
