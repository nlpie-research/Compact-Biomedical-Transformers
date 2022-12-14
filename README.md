# Compact Biomedical Transformers
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-effectiveness-of-compact-biomedical/named-entity-recognition-on-bc5cdr-chemical)](https://paperswithcode.com/sota/named-entity-recognition-on-bc5cdr-chemical?p=on-the-effectiveness-of-compact-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-effectiveness-of-compact-biomedical/named-entity-recognition-on-bc5cdr-disease)](https://paperswithcode.com/sota/named-entity-recognition-on-bc5cdr-disease?p=on-the-effectiveness-of-compact-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-effectiveness-of-compact-biomedical/named-entity-recognition-ner-on-ncbi-disease)](https://paperswithcode.com/sota/named-entity-recognition-ner-on-ncbi-disease?p=on-the-effectiveness-of-compact-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-effectiveness-of-compact-biomedical/named-entity-recognition-on-bc2gm)](https://paperswithcode.com/sota/named-entity-recognition-on-bc2gm?p=on-the-effectiveness-of-compact-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-effectiveness-of-compact-biomedical/named-entity-recognition-ner-on-jnlpba)](https://paperswithcode.com/sota/named-entity-recognition-ner-on-jnlpba?p=on-the-effectiveness-of-compact-biomedical)

<p align="justify"> This repository contains the code used for distillation and fine-tuning of compact biomedical transformers that have been introduced in the paper "On The Effectiveness of Compact Biomedical Transformers". </p>

# Abstract of the Research
<p align="justify"> Language models pre-trained on biomedical corpora, such as BioBERT, have recently shown promising results on downstream biomedical tasks. Many existing pre-trained models, on the other hand, are resource-intensive and computationally heavy owing to factors such as embedding size, hidden dimension, and number of layers. The natural language processing (NLP) community has developed numerous strategies to compress these models utilising techniques such as pruning, quantisation, and knowledge distillation, resulting in models that are considerably faster, smaller, and subsequently easier to use in practice. By the same token, in this paper we introduce six lightweight models, namely, <strong>BioDistilBERT</strong>, <strong>BioTinyBERT</strong>, <strong>BioMobileBERT</strong>, <strong>DistilBioBERT</strong>, <strong>TinyBioBERT</strong>, and <strong>CompactBioBERT</strong> which are obtained either by knowledge distillation from a biomedical teacher or continual learning on the Pubmed dataset via the Masked Language Modelling (MLM) objective. We evaluate all of our models on three biomedical tasks and compare them with BioBERT-v1.1 to create efficient lightweight models that per- form on par with their larger counterparts. </p>

# Available Models
| Model Name  | isDistiiled | #Layers  | #Params | Huggingface Path | Link |
| :------------ |:------------ | :------------: | :------------: | :------------: | :-----:|
| DistilBioBERT         | True  | 6  | 65M |  nlpie/distil-biobert         | [here](https://huggingface.co/nlpie/distil-biobert)         |
| CompactBioBERT        | True  | 6  | 65M |  nlpie/compact-biobert        | [here](https://huggingface.co/nlpie/compact-biobert)        |
| TinyBioBERT           | True  | 4  | 15M |  nlpie/tiny-biobert           | [here](https://huggingface.co/nlpie/tiny-biobert)           |
| BioDistilBERT-cased   | False | 6  | 65M |  nlpie/bio-distilbert-cased   | [here](https://huggingface.co/nlpie/bio-distilbert-cased)   |
| BioDistilBERT-uncased | False | 6  | 65M |  nlpie/bio-distilbert-uncased | [here](https://huggingface.co/nlpie/bio-distilbert-uncased) |
| BioTinyBERT           | False | 4  | 15M |  nlpie/bio-tinybert           | [here](https://huggingface.co/nlpie/bio-tinybert)           |
| BioMobileBERT         | False | 24 | 25M |  nlpie/bio-mobilebert         | [here](https://huggingface.co/nlpie/bio-mobilebert)         |

# How to prepare your coding environment

First, install the below packages using the following command:

```bash
pip install transformers datasets seqeval evaluate
```

Second, clone this repository:

```bash
git clone https://github.com/nlpie-research/Compact-Biomedical-Transformers.git
```

Third, add the path of the cloned repository to your project using the below command so you can access the files in it:

```python
import sys
sys.path.append("PATH_TO_REPO/Compact-Biomedical-Transformers")
```

Forth, download and extract the pre-processed datasets from the [BioBERT Github Repo](https://github.com/dmis-lab/biobert) via these commands:

```bash
wget http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/datasets.tar.gz
tar -xvzf datasets.tar.gz
```

# Run Models on NER

First, import the `load_and_preprocess_dataset` and `train_and_evaluate` functions from the ner.py:

```python
from ner import load_and_preprocess_dataset, train_and_evaluate
```

Then, specify the pre-trained model, dataset, path to dataset and logging file path like this:
```python
datasetName = "BC5CDR-chem"

modelPath = "nlpie/distil-biobert"
tokenizerPath = "nlpie/distil-biobert"

datasetPath = f"PATH_TO_DOWNLOADED_DATASET/datasets/NER/{datasetName}/"
logsPath = f"{datasetName}-logs.txt"
```
Next, load the pre-trained tokeniser from huggingface and call the `load_and_preprocess_dataset` function:
```python
import transformers as ts

tokenizer = ts.AutoTokenizer.from_pretrained(tokenizerPath)

tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset, compute_metrics, label_names = load_and_preprocess_dataset(
    datasetPath=datasetPath,
    tokenizer=tokenizer
)
```
Finally, call the `train_and_evaluate` function and wait for the results:
```python
model, valResults, testResults = train_and_evaluate(lr=5e-5,
                                                    batchsize=16,
                                                    epochs=5,
                                                    tokenizer=tokenizer,
                                                    tokenizedTrainDataset=tokenizedTrainDataset,
                                                    tokenizedValDataset=tokenizedValDataset,
                                                    tokenizedTestDataset=tokenizedTestDataset,
                                                    compute_metrics=compute_metrics,
                                                    label_names=label_names,
                                                    logsPath=logsPath,
                                                    trainingArgs=None)
```

Note that, you can either use our pre-defined TrainingArguments with your desired learning rate, batch size and number of epochs or use the `trainingArgs` argument (which by default is set to None) and pass your custom TrainingArguments to it.

# Run Models on QA

First import the `load_and_preprocess_train_dataset`, `load_test_dataset`, `train`, and `evaluate` from the qa.py along with the following libraries:

```python
import transformers as ts

import torch
import torch.nn as nn
from torch.functional import F

from qa import load_and_preprocess_train_dataset, load_test_dataset, train, evaluate
```

Then, Specify the model, tokeniser, dataset etc, as shown below:

```python
modelPath = "nlpie/bio-distilbert-cased"
tokenizerPath = "nlpie/bio-distilbert-cased"

trainPath = "PATH_TO_DOWNLOADED_DATASET/datasets/QA/BioASQ/BioASQ-train-factoid-7b.json"
testPath = "PATH_TO_DOWNLOADED_DATASET/datasets/QA/BioASQ/BioASQ-test-factoid-7b.json"
goldenPath = "PATH_TO_DOWNLOADED_DATASET/datasets/QA/BioASQ/7B_golden.json"

logsPath = "qa_logs/"
```

Next, load the tokeniser and the train and test datasets:

```python
tokenizer = ts.AutoTokenizer.from_pretrained(tokenizerPath)

trainDataset, tokenizedTrainDataset = load_and_preprocess_train_dataset(trainPath, 
                                                                        tokenizer,
                                                                        max_length=384, 
                                                                        stride=128)
                                                                        
testDataset = load_test_dataset(testPath)
```

Afterwards, train the model using the code below:

```python
model = train(tokenizedTrainDataset,
              modelPath,
              tokenizer,
              learning_rate=3e-5,
              num_epochs=5,
              batch_size=16,
              training_args=None)
```
Please note that you can either use our pre-defined `TrainingArguments` or pass your own `TrainingArguments` to the `training_args` argument.

Finally, use the below code for making predictions on the test dataset and saving it into a json file in the correct format expected by the evalutaion script used in the BioASQ competition.

```python
answersDict = evaluate(model,
                       tokenizer,
                       testDataset,
                       goldenPath,
                       logsPath,
                       top_k_predictions=5,
                       max_seq_len=384,
                       doc_stride=128)
```

### Evaluation using BioASQ evaluation script

First, clone the BioASQ repository with the code below:

```bash
git clone https://github.com/BioASQ/Evaluation-Measures.git
```

Afterwards, use the following code for evaluation:

```bash
java -Xmx10G -cp $CLASSPATH:/FULL_PATH_TO_CLONED_REPO/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 /FULL_PATH_TO_DOWNLOADED_DATASET/datasets/QA/BioASQ/7B_golden.json /FULL_PATH_TO_LOGS_FOLDER/qa_logs/prediction_7B_golden.json
```

Finally, you will get a result like below, in which the second to forth numbers are `Strict Accuracy`, `Lenient Accuracy`, and `Mean Reciprocal Rank` scores respectively.

```bash
1.0 0.2345679012345679 0.36419753086419754 0.28524397413286307 1.0 1.0 1.0 1.0 1.0 1.0
```

# Citation
```bibtex
@misc{https://doi.org/10.48550/arxiv.2209.03182,
  doi = {10.48550/ARXIV.2209.03182},
  url = {https://arxiv.org/abs/2209.03182},
  author = {Rohanian, Omid and Nouriborji, Mohammadmahdi and Kouchaki, Samaneh and Clifton, David A.},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, 68T50},
  title = {On the Effectiveness of Compact Biomedical Transformers},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
