from transformers import (
    AutoModelForSequenceClassification,
    pipeline,
    BertJapaneseTokenizer,
)
import torch
import os
import argparse
import pandas as pd


def classifier(model_dir, text):
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
    )

    res_dict = [{}]
    res_dict = sentiment_analyzer(text, batch_size=1)
    return res_dict


# Remove line if text lengh is more 512
def append_dataset(filename, text, output):
    with open(filename, mode="w") as f:
        f.write("TEXT\tLABEL\tSCORE\n")
        for t, o in zip(text, output):
            content = '"' + t + '"' + "\t" + o["label"] + "\t" + str(o["score"]) + "\n"
            f.write(content)
            #print(content)


def preprocess(text):
    new_text = []
    for t in text:
        if not (len(t) > 512):
            new_text.append(t)
    return new_text


def inference(args):
    df = pd.read_csv(args.input_file, sep="\t")
    df = df.dropna()
    text_dataset = df["TEXT"].tolist()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model_dir_remove_slash = os.path.basename(os.path.dirname(args.model_dir))

    save_filename = args.output_dir + model_dir_remove_slash + ".tsv"

    text_dataset = preprocess(text_dataset)
    output = classifier(args.model_dir, text_dataset)
    append_dataset(save_filename, text_dataset, output)

    return save_filename


def main(args):
    print("[START] Start inference.")
    inference(args)
    print("[PROGRESS] Complete inference.")


if __name__ == "__main__":
    # Read paramaters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Path to the model file.", required=True)
    parser.add_argument("--output_dir", help="Path to the output file.", required=True)
    parser.add_argument(
        "--input_file", help="Path to the input data file.", required=True
    )
    args = parser.parse_args()
    main(args)
