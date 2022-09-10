""" The code used for distillation of the DistilBioBERT.
    It it partially taken from the implementation of the DistillBERT model at https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation
"""

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

ds = load_from_disk("tokenizedDatasets/pubmed-256/")
tokenizer = ts.AutoTokenizer.from_pretrained("distilbert-base-cased")

def initializeStudent(save_path):
  bertModel = ts.AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

  distilBertConfig = bertModel.config.to_dict()
  distilBertConfig["num_hidden_layers"] //= 2

  distillationModel = ts.BertModel(config=ts.BertConfig.from_dict(distilBertConfig))

  distillationModel.embeddings = bertModel.embeddings

  for index , layer in enumerate(distillationModel.encoder.layer):
    distillationModel.encoder.layer[index] = bertModel.encoder.layer[2*index]

  distillationModel.save_pretrained(save_path)

  return save_path

def load_and_save_pretrained(model, checkpoint_path, save_path):
  print(model.load_state_dict(torch.load(checkpoint_path)))
  model.student.save_pretrained(save_path)
  return model

#Use this line if you don't have the initialized model
initializeStudent("distil-biobert/models/distil-biobert-initialized/")

student = ts.AutoModelForMaskedLM.from_pretrained("distil-biobert/models/distil-biobert-initialized/")
teacher = ts.AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

for param in teacher.parameters():
  param.requires_grad = False

print(tokenizer)

from transformers.modeling_outputs import MaskedLMOutput

class DistillationWrapper(nn.Module):
  def __init__(self, student, teacher, temperature=2.0, alpha_ce=5.0, alpha_mlm=2.0, alpha_cos=1.0):
    super().__init__()

    self.student = student
    self.teacher = teacher

    self.temperature = temperature

    self.vocab_size = self.teacher.config.vocab_size
    self.dim = self.teacher.config.hidden_size

    self.restrict_ce_to_mask = True

    self.alpha_ce = alpha_ce
    self.alpha_mlm = alpha_mlm
    self.alpha_cos = alpha_cos

    self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
    self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

  def forward(self, 
              input_ids, 
              attention_mask,
              labels=None,
              **kargs):

    student_outputs = self.student(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels,
                                   output_hidden_states=True,
                                   **kargs)   
    
    s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]

    loss = None

    if labels != None:
      
      with torch.no_grad():
        teacher_outputs = self.teacher(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       output_hidden_states=True,
                                       **kargs)

      t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]    


      if self.restrict_ce_to_mask:
        mask = (labels > -1).unsqueeze(-1).expand_as(s_logits).bool()
      else:
        mask = attention_mask.unsqueeze(-1).expand_as(s_logits).bool()

      s_logits_slct = torch.masked_select(s_logits, mask)  
      s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  
      t_logits_slct = torch.masked_select(t_logits, mask)  
      t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1)) 
      assert t_logits_slct.size() == s_logits_slct.size()
      
      loss_mlm = student_outputs.loss

      loss_ce = (
          self.ce_loss_fct(
              nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
              nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
          )
          * (self.temperature) ** 2
      )

      loss = (self.alpha_mlm * loss_mlm) + (self.alpha_ce * loss_ce)

      if self.alpha_cos > 0.0:
          s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
          t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
          mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states).bool()  # (bs, seq_length, dim)
          assert s_hidden_states.size() == t_hidden_states.size()
          dim = s_hidden_states.size(-1)

          s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
          s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
          t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
          t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

          target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
          loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
          loss += (self.alpha_cos * loss_cos)


    return MaskedLMOutput(
        loss=loss,
        logits=student_outputs.logits,
        hidden_states=student_outputs.hidden_states,
        attentions=student_outputs.attentions,
    )

model = DistillationWrapper(student=student, teacher=teacher)

count = 0

for name , param in model.named_parameters():
  if param.requires_grad == True:
    print(name)
    count += param.numel()

print(count / 1e6)

data_collator = ts.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")

savePath = "distil-biobert/models/distil-biobert/"

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
    logging_steps=250,
    overwrite_output_dir=True,
    save_steps=2500,
    num_train_epochs=(1/173989)*(100000 - 1),
    learning_rate=5e-4,
    lr_scheduler_type="linear",
    warmup_steps=5000,
    per_gpu_train_batch_size=48//2,
    weight_decay=0.0,
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

trainer.save_model(savePath + "final/rawModel/")

load_and_save_pretrained(model, savePath + "final/rawModel/pytorch_model.bin", savePath + "final/model/")
