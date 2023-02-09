import argparse

import sentencepiece as spm
from tqdm import tqdm


def apply_spm(model, input, output):
    s = spm.SentencePieceProcessor(model)

    with open(input, "r", encoding="UTF8") as f:
        input_text = f.read().splitlines()

    output_text = [" ".join(s.encode(sent, out_type=str)) for sent in tqdm(input_text)]

    with open(output, "w", encoding="UTF8") as f:
        for sent in output_text:
            f.write(sent + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()

    apply_spm(args.model, args.input, args.output)
