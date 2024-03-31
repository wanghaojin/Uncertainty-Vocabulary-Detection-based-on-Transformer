import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import torch
import os
from pathlib import Path
import numpy as np
from accelerate import Accelerator, DataLoaderConfiguration


def model_exists(model_path):
    path = Path(model_path)
    return path.is_dir() and any(path.iterdir())


class UncertaintyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].flatten()
        attention_mask = inputs["attention_mask"].flatten()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label),
        }


class OnlineLearningDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].flatten()
        attention_mask = inputs["attention_mask"].flatten()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label),
        }


model_name = "bert-large-uncased"
model_path = "./saved_model_large"
if not model_exists(model_path):
    file_path = "final_updated_shuffled_sentences.xlsx"
    df = pd.read_excel(file_path, header=None, names=["text", "label"])
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = UncertaintyDataset(texts_train, labels_train, tokenizer)
    test_dataset = UncertaintyDataset(texts_test, labels_test, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results_large",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs_large",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

model_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

while True:
    user_sentence = input("请输入一个句子（输入'!'结束）：")
    if user_sentence == "!":
        break

    predict_dataset = OnlineLearningDataset([user_sentence], [0], tokenizer)
    predict_trainer = Trainer(model=model)
    outputs = predict_trainer.predict(predict_dataset)
    predictions = np.argmax(outputs.predictions, axis=-1)
    print(f"Predict Label:{predictions.item()}")

    true_label = int(input("True Label(0 or 1):"))

    if predictions.item() != true_label:
        fine_tune_dataset = OnlineLearningDataset(
            [user_sentence], [true_label], tokenizer
        )
        fine_tune_args = TrainingArguments(
            output_dir="./fine_tune_results",
            num_train_epochs=20,
            per_device_train_batch_size=1,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./fine_tune_logs",
            learning_rate=1e-5,
        )
        fine_tune_trainer = Trainer(
            model=model,
            args=fine_tune_args,
            train_dataset=fine_tune_dataset,
        )

        fine_tune_trainer.train()
        print("False.Training!")
    else:
        print("True!")
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
