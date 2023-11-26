import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import argparse


def print_score(y_pred, y_true):
    print("SKRLEARN SCORE")
    report = classification_report(y_true, y_pred)
    print(report)

    with open(output + "report.txt", "w") as f:
        f.write(report)


def print_matrix(y_pred, y_true):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    cmp = ConfusionMatrixDisplay(cm, display_labels=["NEGATIVE", "NEUTRAL", "POSITIVE"])
    print(cmp)


def count(df, file_name):
    count = df.value_counts(subset=["LABEL"], normalize=True)
#    plt.bar(["NEUTRAL", "NEGATIVE", "POSITIVE"], count)
#    plt.title("Distribution of LABEL in " + os.path.splittext(file_name))
#    plt.savefig(output + "count_" + os.path.splitext(file_name) + ".png")
    print(file_name + ":")
    print(count)


def read_csv(file):
    df_file = []
    for f in file:
        df_file.append(pd.read_csv(f, sep="\t"))
    return df_file


def main(args):
    global output
    output = args.save_path

    file_path = []
    file_path.append(args.pred_file)
    file_path.append(args.test_file)

    filename = []
    for f in file_path:
        filename.append(os.path.basename(f))
    df_file = read_csv(file_path)

    # Delete NaN
    clean_df = []
    df_file[0] = df_file[0].drop("SCORE", axis=1)
    for f in df_file:
        clean_df.append(f.dropna(how="any"))

    for df, f in zip(clean_df, filename):
        count(df, f)

    clean_df[0] = clean_df[0].rename(columns={"LABEL": "PRED_LABEL"})
    clean_df[1] = clean_df[1].rename(columns={"LABEL": "TEST_LABEL"})
    
    # Marge columuns
    result_df = pd.merge(clean_df[0], clean_df[1], on="TEXT", how="inner")
    result_df = result_df.drop_duplicates()

    y_pred = result_df["PRED_LABEL"]
    y_true = result_df["TEST_LABEL"]
    print_matrix(y_pred, y_true)
    print_score(y_pred, y_true)


if __name__ == "__main__":
    # Read paramaters
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", help="Path to the output file.", required=True)
    parser.add_argument(
        "--test_file", help="Path to the input data file.", required=True
    )
    parser.add_argument(
        "--save_path", help="Path to the evaluation plot file.", required=False, default="./"
    )
    args = parser.parse_args()
    main(args)
