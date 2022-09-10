import transformers as ts
from datasets import Dataset
from datasets import load_dataset, load_from_disk

import numpy as np
import numpy.core.defchararray as nchar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers.adapters import AdapterConfig

import math

ds = load_dataset('pubmed', cache_dir="dataset/")

print(ds)

def getTextFromSample(sample):
  article = sample["Article"]
  title = article["ArticleTitle"].replace("[","").replace("]","").strip()
  abstract = article["Abstract"]["AbstractText"].strip()

  text = (title + " " + abstract).strip()

  return text

tokenizer = ts.AutoTokenizer.from_pretrained("bert-base-cased")

print(tokenizer)

def mappingFunction(dataset):
  texts = []
  
  for sample in dataset["MedlineCitation"]:
    texts.append(getTextFromSample(sample))
  
  return tokenizer(texts, truncation=True, max_length=256, return_special_tokens_mask=True)

ds["train"] = ds["train"].map(mappingFunction, batched=True)

datasetPath = "tokenizedDatasets/pubmed-256/"

ds.save_to_disk(datasetPath)

print(load_from_disk(datasetPath))
