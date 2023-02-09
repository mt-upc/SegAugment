import argparse
import os
import sys

import pandas as pd

sys.path.append(os.environ["FAIRSEQ_ROOT"])
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv

parser = argparse.ArgumentParser()
parser.add_argument("--input-tsvs", "-input", required=True, type=str)
parser.add_argument("--output-tsv", "-output", required=True, type=str)
parser.add_argument("--drop-duplicates", "-drop", action="store_true")
args = parser.parse_args()

tsv_paths = args.input_tsvs.split(",")

df = pd.DataFrame()
for tsv_file in tsv_paths:
    df = pd.concat([df, load_df_from_tsv(tsv_file)], ignore_index=True)

df.speaker = df.apply(lambda x: "spk." + x["id"].split("_")[1], axis=1)

if args.drop_duplicates:
    df.drop_duplicates(subset=["tgt_text", "speaker"], ignore_index=True, inplace=True)

save_df_to_tsv(df, args.output_tsv)
