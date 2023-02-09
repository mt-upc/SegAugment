import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

src_lang = sys.argv[1]
tgt_lang = sys.argv[2]
wav_path = Path(sys.argv[3])
txt_path = Path(sys.argv[4])
n_talks = int(sys.argv[5])
min_duration = int(sys.argv[6])
max_duration = int(sys.argv[7])

origin_name = "train"

name = f"train_{n_talks}"

new_wav_path = Path(str(wav_path).replace(origin_name, name))
new_txt_path = Path(str(txt_path).replace(origin_name, name))

new_wav_path.mkdir(exist_ok=True, parents=True)
new_txt_path.mkdir(exist_ok=True, parents=True)

with open(txt_path / f"{origin_name}.{src_lang}") as f:
    all_src_txt = f.read().splitlines()
with open(txt_path / f"{origin_name}.{tgt_lang}") as f:
    all_tgt_txt = f.read().splitlines()
with open(txt_path / f"{origin_name}.yaml") as f:
    all_segments = yaml.load(f, Loader=yaml.CLoader)

talks_info = {}
all_wavs = []
for src_txt, tgt_txt, segment in zip(all_src_txt, all_tgt_txt, all_segments):
    wav = segment["wav"]
    if wav not in talks_info.keys():
        talks_info[wav] = {
            "segment": [segment],
            "src_txt": [src_txt],
            "tgt_txt": [tgt_txt],
        }
        all_wavs.append(wav)
    else:
        talks_info[wav]["segment"].append(segment)
        talks_info[wav]["src_txt"].append(src_txt)
        talks_info[wav]["tgt_txt"].append(tgt_txt)

new_wavs = []
while len(new_wavs) < n_talks:
    wav = all_wavs[np.random.choice(len(all_wavs))]

    if wav in new_wavs:
        continue

    if any(
        [
            not (min_duration < float(segment["duration"]) < max_duration)
            for segment in talks_info[wav]["segment"]
        ]
    ):
        continue

    new_wavs.append(wav)
    shutil.copy(wav_path / wav, new_wav_path / wav)

new_src_txt, new_tgt_txt, new_segments = [], [], []
for wav in sorted(new_wavs):
    for src_txt, tgt_txt, segment in zip(
        talks_info[wav]["src_txt"],
        talks_info[wav]["tgt_txt"],
        talks_info[wav]["segment"],
    ):
        new_src_txt.append(src_txt)
        new_tgt_txt.append(tgt_txt)
        new_segments.append(segment)

with open(new_txt_path / f"{name}.{src_lang}", "w") as f:
    for src_txt in new_src_txt:
        f.write(src_txt + "\n")
with open(new_txt_path / f"{name}.{tgt_lang}", "w") as f:
    for tgt_txt in new_tgt_txt:
        f.write(tgt_txt + "\n")
with open(new_txt_path / f"{name}.yaml", "w") as f:
    yaml.dump(new_segments, f, default_flow_style=True)
