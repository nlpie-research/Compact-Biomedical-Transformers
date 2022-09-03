import transformers as ts
from transformers import DataCollatorForTokenClassification

from datasets import Dataset
from datasets import load_metric

import numpy as np
import pandas as pd
import csv

datasetName = "NCBI-disease"

modelPath = "nlpie/distil-biobert"
tokenizerPath = "nlpie/distil-biobert"

datasetPath = f"biobert-datasets/datasets/NER/{datasetName}/"
logsPath = f"ner_logs/{modelPath}-{datasetName}-logs.txt"

def load_and_preprocess_dataset(datasetPath, tokenizer):
    def load_ner_dataset(folder):
        allLabels = set(pd.read_csv(folder + "train.tsv", sep="\t",
                                    header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')[1])

        label_to_index = {label: index for index,
                          label in enumerate(allLabels)}
        index_to_label = {index: label for index,
                          label in enumerate(allLabels)}

        def load_subset(subset):
            lines = []

            with open(folder + subset, mode="r") as f:
                lines = f.readlines()

            sentences = []
            labels = []

            currentSampleTokens = []
            currentSampleLabels = []

            for line in lines:
                if line.strip() == "":
                    sentences.append(currentSampleTokens)
                    labels.append(currentSampleLabels)
                    currentSampleTokens = []
                    currentSampleLabels = []
                else:
                    cleanedLine = line.replace("\n", "")
                    token, label = cleanedLine.split(
                        "\t")[0].strip(), cleanedLine.split("\t")[1].strip()
                    currentSampleTokens.append(token)
                    currentSampleLabels.append(label_to_index[label])

            dataDict = {
                "tokens": sentences,
                "ner_tags": labels,
            }

            return Dataset.from_dict(dataDict)

        trainingDataset = load_subset("train.tsv")
        validationDataset = Dataset.from_dict(
            load_subset("train_dev.tsv")[len(trainingDataset):])
        testDataset = load_subset("test.tsv")

        return {
            "train": trainingDataset,
            "validation": validationDataset,
            "test": testDataset,
            "all_ner_tags": list(allLabels),
        }

    dataset = load_ner_dataset(datasetPath)

    label_names = dataset["all_ner_tags"]

    # Get the values for input_ids, token_type_ids, attention_mask
    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(
            all_samples_per_split["tokens"], is_split_into_words=True, max_length=512)
        total_adjusted_labels = []

        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid != prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])

            total_adjusted_labels.append(adjusted_label_ids)

        tokenized_samples["labels"] = total_adjusted_labels

        return tokenized_samples

    tokenizedTrainDataset = dataset["train"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedValDataset = dataset["validation"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedTestDataset = dataset["test"].map(
        tokenize_adjust_labels, batched=True)

    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(
            predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }

        return flattened_results

    return tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset, compute_metrics, label_names

def trainAndEvaluate(lr,
                     batchsize,
                     epochs,
                     tokenizer,
                     tokenizedTrainDataset,
                     tokenizedValDataset,
                     tokenizedTestDataset,
                     compute_metrics,
                     label_names,
                     trainingArgs=None):
    model = ts.AutoModelForTokenClassification.from_pretrained(
        modelPath, num_labels=len(label_names))
    data_collator = DataCollatorForTokenClassification(tokenizer)

    model.train()

    if trainingArgs == None:
        trainingArguments = ts.TrainingArguments(
            "output/",
            seed=42,
            logging_steps=250,
            save_steps=2500,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=batchsize,
            per_device_eval_batch_size=batchsize,
            weight_decay=0.01,
        )
    else:
        trainingArguments = trainingArgs

    trainer = ts.Trainer(
        model=model,
        args=trainingArguments,
        train_dataset=tokenizedTrainDataset,
        eval_dataset=tokenizedValDataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.eval()

    evaluationResult = trainer.evaluate()

    trainer.eval_dataset = tokenizedTestDataset

    testResult = trainer.evaluate()

    if logsPath != None:
        with open(logsPath, mode="a+") as f:
            f.write(
                f"---HyperParams---\nBatchsize= {batchsize} Lr= {lr}\n---Val Results---\n{str(evaluationResult)}\n---Test Results---\n{str(testResult)}\n\n")

    return model, evaluationResult, testResult

