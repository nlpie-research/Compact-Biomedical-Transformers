""" The code used for distillation of the TinyBioBERT.
    It it partially taken from the implementation of the TinyBERT model at https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
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

from transformers.modeling_outputs import MaskedLMOutput

import math

ds = load_from_disk("tokenizedDatasets/pubmed-256/")

modelPath = "distilbert-base-cased"
tokenizer = ts.AutoTokenizer.from_pretrained(modelPath)

teacher = ts.AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

studentConfig = ts.AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").config.to_dict()
studentConfig["vocab_size"] = teacher.config.vocab_size
student = ts.BertForMaskedLM(config=ts.BertConfig.from_dict(studentConfig))

for param in teacher.parameters():
  param.requires_grad = False

print(tokenizer)

class DistillationWrapper(nn.Module):
  def __init__(self, student, teacher):
    super().__init__()

    self.student = student
    self.teacher = teacher

    self.mse_loss = nn.MSELoss()
    self.output_loss = nn.CrossEntropyLoss()

    self.teacherDim = self.teacher.config.hidden_size
    self.studentDim = self.student.config.hidden_size

    self.fit_dense = nn.Linear(self.studentDim, self.teacherDim)

    self.temperature = 1.0

  def forward(self, 
              input_ids,
              attention_mask=None,
              labels=None,
              **kargs):
    
    student_outputs = self.student(input_ids=input_ids,
                                   attention_mask=attention_mask, 
                                   labels=labels, 
                                   output_hidden_states=True,
                                   output_attentions=True,
                                   **kargs)

    with torch.no_grad():
      teacher_outputs = self.teacher(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True,
                                     output_attentions=True,
                                     **kargs)
  
    s_attentions = student_outputs["attentions"]
    t_attentions = [att.detach() for att in teacher_outputs["attentions"]]

    s_hiddens = student_outputs["hidden_states"]
    t_hiddens = [hidden.detach() for hidden in teacher_outputs["hidden_states"]]

    s_logits = student_outputs["logits"]
    t_logits = teacher_outputs["logits"].detach()

    att_loss = 0
    rep_loss = 0
    
    teacher_layer_num = len(t_attentions)
    student_layer_num = len(s_attentions)
    
    assert teacher_layer_num % student_layer_num == 0

    layers_per_block = int(teacher_layer_num / student_layer_num)

    new_teacher_atts = [t_attentions[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)]
    
    for student_att, teacher_att in zip(s_attentions, new_teacher_atts):
        att_loss += self.mse_loss(student_att, teacher_att)

    new_teacher_reps = [t_hiddens[i * layers_per_block] for i in range(student_layer_num + 1)]
    new_student_reps = s_hiddens

    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        rep_loss += self.mse_loss(self.fit_dense(student_rep), teacher_rep)

    mask = (labels > -1).unsqueeze(-1).expand_as(s_logits).bool()

    s_logits_slct = torch.masked_select(s_logits, mask)  
    s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  
    t_logits_slct = torch.masked_select(t_logits, mask)  
    t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1)) 
    assert t_logits_slct.size() == s_logits_slct.size()
    
    output_loss = self.output_loss(
        (s_logits_slct / self.temperature),
        nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
    )

    loss = (att_loss + rep_loss) + output_loss

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

savePath = "distil-biobert/models/tiny-biobert/"

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
    num_train_epochs=(1/173989)*(100000),
    learning_rate=5e-4,
    lr_scheduler_type="linear",
    warmup_steps=10000,
    per_gpu_train_batch_size=24, #We used 8 gpus so the total batch_size is 192
    weight_decay=1e-4,
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

def load_and_save_pretrained(model, checkpoint_path, save_path):
  print(model.load_state_dict(torch.load(checkpoint_path)))
  model.student.save_pretrained(save_path)
  return model

trainer.save_model(savePath + "final/rawModel/")
load_and_save_pretrained(model, savePath + "final/rawModel/pytorch_model.bin", savePath + "final/model/")
