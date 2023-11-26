import os
import argparse


def join(input_dir, output_path):
    basename = os.listdir(input_dir)
    savefile = output_path + "prepro_output.tsv"
    with open(savefile, "w") as f:
        f.write("TEXT\n")
        for filename in basename:
            with open(input_dir + filename, "r") as rf:
                f.write(rf.read())
    return savefile


def main(args):
    print("[START] Input directory: " + args.input_dir)
    savefile = join(args.input_dir, args.output_path)
    print("[DONE] Output file: " + savefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Input directory name.", required=True)
    parser.add_argument(
        "--output_path",
        help="Path to directory of output.",
        required=False,
        default="./",
    )
    args = parser.parse_args()
    main(args)
