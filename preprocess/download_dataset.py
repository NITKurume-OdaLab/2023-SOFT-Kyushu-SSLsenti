from datasets import load_dataset
import argparse
import pandas as pd


def wrime(output_dir, dataset_split):
    if dataset_split == None:
        print("[ERROR] This dataset require dataset_split")
        return
    # shunk031/wrime(Hugging Face) -> https://huggingface.co/datasets/shunk031/wrime
    # ids-cv/wrime(GitHub) -> https://github.com/ids-cv/wrime
    dataset = load_dataset("shunk031/wrime", "ver2")

    df = pd.DataFrame(dataset[dataset_split])
    df_all = pd.concat([df], axis=0)

    df_all = df_all.drop(
        columns=["user_id", "datetime", "writer", "reader1", "reader2", "reader3"]
    )
    tmp = df_all["avg_readers"].to_list()
    df_avg_new = []
    for li in tmp:
        df_avg_new.append(li["sentiment"])
    df_sentiment = pd.DataFrame(df_avg_new)
    df_sentiment.columns = ["LABEL"]

    df_text = df_all.drop(columns="avg_readers")
    df_text.columns = ["TEXT"]
    df_text.index = df_sentiment.index
    df = pd.concat([df_text, df_sentiment], axis=1)

    df.to_csv(output_dir + "wrime.tsv", sep="\t", header=True, index=False)


def main(args):
    if args.dataset == "wrime":
        wrime(args.output_dir, args.split)


if __name__ == "__main__":
    # Read paramaters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name", required=True)
    parser.add_argument("--split", help="Split name", required=False, default=None)
    parser.add_argument(
        "--output_dir",
        help="Path to directory of output.",
        required=False,
        default="./",
    )
    args = parser.parse_args()
    main(args)
