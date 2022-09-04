import transformers as ts
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

from datasets import Dataset

import json
import os

def load_and_preprocess_train_dataset(path, tokenizer, max_length=384, stride=128):
    def load_data(path):
        dataDict = {
            "id": [],
            "title": [],
            "context": [],
            "question": [],
            "answers": [],
        }

        def load_subset(path, dataDict):
            with open(path) as d:
                dictData = json.load(d)

                sample = dictData["data"]

                for sample in dictData["data"]:
                    for paragraph in sample["paragraphs"]:
                        for qas in paragraph["qas"]:
                            id = qas["id"]
                            title = sample["title"]
                            context = paragraph["context"]
                            question = qas["question"]

                            for answer in qas["answers"]:
                                dataDict["id"].append(id)
                                dataDict["title"].append(title)
                                dataDict["context"].append(context)
                                dataDict["question"].append(question)
                                dataDict["answers"].append(
                                    {"text": [answer["text"]], "answer_start": [answer["answer_start"]]})

        load_subset(path, dataDict)

        return Dataset.from_dict(dataDict)

    trainDataset = load_data(path)

    def preprocessing_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenizedTrainDataset = trainDataset.map(
        preprocessing_function, batched=True, remove_columns=trainDataset.column_names)

    return trainDataset, tokenizedTrainDataset


def load_and_preprocess_test_dataset(path, tokenizer, max_length=384, stride=128):
    def load_data(path):
        dataDict = {
            "id": [],
            "title": [],
            "context": [],
            "question": [],
        }

        def load_subset(path, dataDict):
            with open(path) as d:
                dictData = json.load(d)

                sample = dictData["data"]

                for sample in dictData["data"]:
                    for paragraph in sample["paragraphs"]:
                        for qas in paragraph["qas"]:
                            id = qas["id"]
                            title = sample["title"]
                            context = paragraph["context"]
                            question = qas["question"]

                            dataDict["id"].append(id)
                            dataDict["title"].append(title)
                            dataDict["context"].append(context)
                            dataDict["question"].append(question)

        load_subset(path, dataDict)

        return Dataset.from_dict(dataDict)

    testDataset = load_data(path)

    def preprocess_test_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    tokenizedTestDataset = testDataset.map(
        preprocess_test_examples, batched=True, remove_columns=testDataset.column_names)

    return testDataset, tokenizedTestDataset


def train(tokenizedTrainDataset,
          modelPath,
          tokenizer,
          learning_rate=3e-5,
          num_epochs=5,
          batch_size=16,
          trainingArgs=None):

    model = AutoModelForQuestionAnswering.from_pretrained(modelPath)

    """#Training"""

    data_collator = ts.DefaultDataCollator()

    training_args = TrainingArguments(
        "output/",
        seed=42,
        logging_steps=250,
        save_steps=2500,
        num_train_epochs=5,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenizedTrainDataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return model


def evaluate(model, testDataset, goldenPath, logsPath, top_k_predictions=5):
    if not os.path.exists(logsPath):
        os.mkdir(logsPath)

    qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)

    answersDict = {}

    for testSample in testDataset:
        model.cpu()

        answers = []

        for answer in qa_model(question=testSample["question"], context=testSample["context"], top_k=top_k_predictions):
            answers.append(answer["answer"])

        id = testSample["id"].split("_", 1)[0]

        if id in answersDict:
            answersDict[id].append(answers)
        else:
            answersDict[id] = [answers]

    def write_answers_to_csv(answersDict, goldenPath):
        with open(path) as f:
            data = json.load(f)

        questions = data["questions"]

        for sample in questions:
            if sample["type"] == "factoid":
                if "exact_answer" in sample:
                    predicted_answers = answersDict[sample["id"]]

                    sample["ideal_answer"] = ["dummy"]
                    sample["exact_answer"] = predicted_answers

        data["questions"] = questions

        outputTitle = goldenPath.split("/")[-1]
        outputPath = f"{logsPath}/prediction_{outputTitle}"

        with open(outputPath, mode="w") as f:
            json.dump(data, f, indent=6)

    write_answers_to_csv(answersDict, goldenPath)

    return answersDict
