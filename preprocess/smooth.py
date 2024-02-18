import pandas as pd
import argparse


def main(args):
    input_file = args.input_path
    output_file = args.output_path

    df = pd.read_csv(input_file, delimiter="\t")
    print("Before nomarized")
    print(df.value_counts(subset=["LABEL"], normalize=True))

    df = df.sort_values("SCORE", ascending=False)
    df = df.query("SCORE > 0.5")
    count = df.value_counts(subset=["LABEL"], normalize=False)

    df_neutral = df.query('LABEL != "POSITIVE" and LABEL != "NEGATIVE"')
    df_neutral = df_neutral.sort_values("SCORE", ascending=False)
    df_neutral = df_neutral.head(count["POSITIVE"])

    df_negative = df.query('LABEL != "POSITIVE" and LABEL != "NEUTRAL"')
    df_negative = df_negative.sort_values("SCORE", ascending=False)
    df_negative = df_negative.head(count["POSITIVE"])

    df_positive = df.query('LABEL != "NEGATIVE" and LABEL != "NEUTRAL"')
    df_positive = df_positive.sort_values("SCORE", ascending=False)
    df_positive = df_positive.head(count["POSITIVE"])

    df_training = pd.concat([df_positive, df_negative, df_neutral])
    df_training.to_csv(output_file, sep="\t", index=False)

    print("After nomarized")
    print(df_training.value_counts(subset=["LABEL"], normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input filepath name.", required=True)
    parser.add_argument(
        "--output_path",
        help="Path to directory of output.",
        required=False,
        default="./step2-prepro.tsv",
    )
    args = parser.parse_args()
    main(args)
