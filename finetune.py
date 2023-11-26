# Ref -> https://jupyterbook.hnishi.com/language-models/fine_tune_jp_bert_part02.html

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertJapaneseTokenizer,
    EarlyStoppingCallback
)
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import argparse

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class JpSentiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], dtype=torch.int64)
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item

    def __len__(self):
        return len(self.labels)


def data_drive(df):
    df = df.drop(
        columns=["user_id", "datetime", "writer", "reader1", "reader2", "reader3"]
    )
    tmp = df["avg_readers"].to_list()
    df_avg_new = []
    for li in tmp:
        df_avg_new.append(li["sentiment"])
    df_sentiment = pd.DataFrame(df_avg_new)
    df_sentiment.columns = ["LABEL"]
    df_text = df.drop(columns="avg_readers")
    df_text.columns = ["TEXT"]
    df_text.index = df_sentiment.index
    df = pd.concat([df_text, df_sentiment], axis=1)

    df = df.replace(
        {-2: "NEGATIVE", -1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE", 2: "POSITIVE"}
    )
    return df


# Holdout method(7:3)
def data_create_wrime():
    dataset = load_dataset("shunk031/wrime", "ver2")

    df_train = pd.DataFrame(dataset["train"])
    df_valid = pd.DataFrame(dataset["validation"])

    df_train = data_drive(df_train)
    df_valid = data_drive(df_valid)

    return df_train, df_valid

def data_create(path):
    df = pd.read_csv(path,delimiter="\t")
    df_train, df_valid = train_test_split(df,shuffle=True,random_state=0)

    return df_train, df_valid

def train_finetune(df_train, df_valid, args):
    EARLY_STOPPING = False
    callbacks = None
    metric_for_best_model = None
    if(int(args.epoch)==0):
        EARLY_STOPPING = True
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        metric_for_best_model = "f1"

    df_train = df_train.dropna(how="any")
    df_valid = df_valid.dropna(how="any")

    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    df_train["LABEL"] = df_train["LABEL"].map(label2id)
    df_valid["LABEL"] = df_valid["LABEL"].map(label2id)

    label_tarin = df_train["LABEL"].tolist()
    label_valid = df_valid["LABEL"].tolist()
    text_train = df_train["TEXT"].tolist()
    text_valid = df_valid["TEXT"].tolist()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = args.model_dir
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id
    )
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    model = model.to(device)
    train_encodings = tokenizer(
        text_train,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)
    valid_encodings = tokenizer(
        text_valid,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    train_dataset = JpSentiDataset(train_encodings, label_tarin)
    valid_dataset = JpSentiDataset(valid_encodings, label_valid)
    
    epoch = args.epoch
    output_dir = args.output_dir

    if(int(epoch) == 0):
        epoch = 100

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=int(epoch),  # total number of training epochs
        per_device_train_batch_size=int(args.batch_size),  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        save_total_limit=1,  # limit the total amount of checkpoints. Deletes the older checkpoints.
        dataloader_pin_memory=False,  # Whether you want to pin memory in data loaders or not. Will default to True
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        logging_strategy="epoch",
        logging_dir="./logs",

        # Early stopping settings
        load_best_model_at_end=EARLY_STOPPING,
        metric_for_best_model=metric_for_best_model
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # The function that will be used to compute metrics at evaluation
        callbacks=callbacks
    )

    trainer.train()
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    trainer.save_state()

def main(args):
    if args.input_file=="wrime":
        df_train, df_valid = data_create_wrime()
    else:
        df_train, df_valid = data_create(args.input_file)
    train_finetune(df_train, df_valid, args)

if __name__ == '__main__':
    # Read paramaters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        help='Path to the target model file.',
                        required=True
                        )
    parser.add_argument('--output_dir',
                        help='Path to the output file of the model.',
                        required=False,
                        default="./"
                        )
    parser.add_argument('--input_file',
                        help='Path to the input data file.',
                        required=False,
                        default="wrime"
                        )
    parser.add_argument('--epoch',
                        help='Set epoch number.',
                        required=False,
                        default=20
                        )
    parser.add_argument('--logging_dir',
                        help='Path to the log directory.',
                        required=False,
                        default="./logs"
                        )
    parser.add_argument('--batch_size',
                        help='Set to batch size',
                        required=False,
                        default=16
                        )
    
    args = parser.parse_args()
    main(args)
