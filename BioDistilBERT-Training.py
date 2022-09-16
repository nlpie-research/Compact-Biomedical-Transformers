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

import math

ds = load_from_disk("tokenizedDatasets/pubmed-256/") #This dataset is the tokenized version of the PubMed dataset, tokenized with the "distilbert-base-cased" tokenizer from Huggingface. The preprocessing.py can be used to reproduce this dataset.

modelPath = "distilbert-base-cased"

tokenizer = ts.AutoTokenizer.from_pretrained(modelPath)
model = ts.AutoModelForMaskedLM.from_pretrained(modelPath)

print(tokenizer)

count = 0

for name , param in model.named_parameters():
  if param.requires_grad == True:
    print(name)
    count += param.numel()

print(count/1e6)

data_collator = ts.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")

savePath = "distil-biobert/models/bio-distilbert-cased/"

try:
  with open(savePath + "logs.txt", "w+") as f:
                f.write("")
except:
  pass

class CustomCallback(ts.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            with open(savePath + "logs.txt", "a+") as f:
              f.write(str(logs) + "\n")

trainingArguments = ts.TrainingArguments(
    savePath + "checkpoints",
    logging_steps=500,
    overwrite_output_dir=True,
    save_steps=2500,
    num_train_epochs=(1/173989)*(200000),
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=10000,
    per_gpu_train_batch_size=24, #We used 8 gpus so the total batch_size is 192
    weight_decay=0.01,
    save_total_limit=5,
    remove_unused_columns=True,
)

trainer = ts.Trainer(
    model=model,
    args=trainingArguments,
    train_dataset=ds["train"],
    data_collator=data_collator,
    callbacks=[ts.ProgressCallback(), CustomCallback()],
)

trainer.train()

trainer.save_model(savePath + "final/model/")
