import re
import argparse
import pandas as pd

def main(args):
    print("[START] Input file: " + args.input_path)

    df = pd.read_csv(args.input_path,delimiter="\t")
    
    df = df.replace("　",'')
    r = r"《.+》"
    df = df.replace(re.compile(r),'')
    r = r"［＃.+］"
    df = df.replace(re.compile(r),'')
    r = r"-------------------------------------------------------\n.+-------------------------------------------------------\n"
    df = df.replace(re.compile(r),'')
    df = df.replace("\n", "")
    df = df.dropna()

    df.to_csv(args.output_path+"data.tsv",sep="\t")
    print("[DONE] Output file: " + args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input filepath name.", required=True)
    parser.add_argument(
        "--output_path",
        help="Path to directory of output.",
        required=False,
        default="./",
    )
    args = parser.parse_args()
    main(args)
