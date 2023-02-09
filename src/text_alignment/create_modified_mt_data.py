import argparse
from pathlib import Path

import nltk
import numpy as np
import yaml

nltk.download("punkt")

LANG_MAP = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "it": "italian",
    "pt": "portuguese",
    "fr": "french",
    "nl": "dutch",
}


def load_data(txt_root, split, src, tgt):
    """load data from yaml and txt files"""

    with open(txt_root / f"{split}.yaml", "r") as f:
        segmentation = yaml.load(f, Loader=yaml.CLoader)
    with open(txt_root / f"{split}.{src}", "r") as f:
        src_txt = f.read().splitlines()
    with open(txt_root / f"{split}.{tgt}", "r") as f:
        tgt_txt = f.read().splitlines()

    data = {}
    durations = []
    for sgm, s, t in zip(segmentation, src_txt, tgt_txt):
        durations.append(float(sgm["duration"]))
        talk_id = sgm["wav"].split(".")[0]
        if talk_id not in data.keys():
            data[talk_id] = [sgm]
        else:
            data[talk_id].append(sgm)
        data[talk_id][-1]["src_txt"] = s
        data[talk_id][-1]["tgt_txt"] = t

    return data, durations


def _add(sgm, queue):
    """add segment to queue"""
    if queue is None:
        return {
            "duration": float(sgm["duration"]),
            "src_txt": sgm["src_txt"],
            "tgt_txt": sgm["tgt_txt"],
        }
    queue["duration"] += float(sgm["duration"])
    queue["src_txt"] = " ".join([queue["src_txt"], sgm["src_txt"]])
    queue["tgt_txt"] = " ".join([queue["tgt_txt"], sgm["tgt_txt"]])
    return queue


def create_longer(data, min_, max_):
    """create longer segments by combining original segments"""
    new_src_txt, new_tgt_txt, durations = [], [], []
    for talk_data in data.values():
        queue = None
        for sgm in talk_data:
            d = float(sgm["duration"])
            if queue is not None:
                if (
                    (queue["duration"] < min_)
                    or (d < min_)
                    or (d + queue["duration"] < max_)
                ):
                    queue = _add(sgm, queue)
                else:
                    new_src_txt.append(queue["src_txt"])
                    new_tgt_txt.append(queue["tgt_txt"])
                    durations.append(queue["duration"])
                    queue = _add(sgm, None)
            else:
                queue = _add(sgm, None)
        if queue is not None:
            if queue["duration"] < min_:
                new_src_txt[-1] = " ".join([new_src_txt[-1], queue["src_txt"]])
                new_tgt_txt[-1] = " ".join([new_tgt_txt[-1], queue["tgt_txt"]])
                durations[-1] += queue["duration"]
            else:
                new_src_txt.append(queue["src_txt"])
                new_tgt_txt.append(queue["tgt_txt"])
                durations.append(queue["duration"])
    return new_src_txt, new_tgt_txt, durations


def maybe_split(sgm, src, tgt):
    """Split a segment into sentences."""
    sents_src = nltk.sent_tokenize(sgm["src_txt"], language=LANG_MAP[src])
    sents_tgt = nltk.sent_tokenize(sgm["tgt_txt"], language=LANG_MAP[tgt])
    if len(sents_src) == len(sents_tgt):
        return sents_src, sents_tgt
    else:
        return [sgm["src_txt"]], [sgm["tgt_txt"]]


def create_shorter(data, src, tgt):
    """Create shorter data."""
    new_src_txt, new_tgt_txt = [], []
    for talk_data in data.values():
        for sgm in talk_data:
            sents_src, sents_tgt = maybe_split(sgm, src, tgt)
            new_src_txt.extend(sents_src)
            new_tgt_txt.extend(sents_tgt)
        else:
            new_src_txt.append(sgm["src_txt"])
            new_tgt_txt.append(sgm["tgt_txt"])
    return new_src_txt, new_tgt_txt


def create_same(data):
    """Create same length data."""
    new_src_txt, new_tgt_txt = [], []
    for talk_data in data.values():
        for sgm in talk_data:
            new_src_txt.append(sgm["src_txt"])
            new_tgt_txt.append(sgm["tgt_txt"])
    return new_src_txt, new_tgt_txt


def create_modified_mt_split(args):
    min_ = 3
    max_ = args.max

    if int(args.min) == args.min:
        args.min = int(args.min)
    if int(args.max) == args.max:
        args.max = int(args.max)

    output_dir = Path(args.output_dir) / args.lang_pair / f"{args.min}-{args.max}"
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_root = Path(args.data_root) / args.lang_pair / "data" / args.split_name / "txt"
    src, tgt = args.lang_pair.split("-")
    assert (
        src in LANG_MAP.keys() and tgt in LANG_MAP.keys()
    ), "Need to add the language to the LANG_MAP"

    data, durations = load_data(txt_root, args.split_name, src, tgt)

    durations = np.array(durations)
    print("Original Stats")
    print(f"number of segments = {len(durations)}")
    print(f"min durations = {sorted(durations)[:10]}")
    print(f"max durations = {sorted(durations)[::-1][:10]}")
    print(f"avg duration = {round(np.mean(durations), 4)}")

    if args.max > 9 and not args.force_the_same:
        new_src_txt, new_tgt_txt, durations = create_longer(data, min_, max_)

        durations = np.array(durations)
        print("_" * 50)
        print("New Stats")
        print(f"number of segments = {len(durations)}")
        print(f"min durations = {sorted(durations)[:10]}")
        print(f"max durations = {sorted(durations)[::-1][:10]}")
        print(f"avg duration = {round(np.mean(durations), 4)}")
    elif args.max < 3 and not args.force_the_same:
        # romanian not supported by nltk
        # TODO: add here any languages that are not supported by nltk
        if tgt == "ro":
            new_src_txt, new_tgt_txt = create_same(data)
        else:
            new_src_txt, new_tgt_txt = create_shorter(data, src, tgt)
        print("_" * 50)
        print(f"number of segments = {len(new_src_txt)}")
    else:
        new_src_txt, new_tgt_txt = create_same(data)

    with open(output_dir / f"{args.split_name}_{args.min}-{args.max}.{src}", "w") as f:
        for txt in new_src_txt:
            f.write(txt + "\n")
    with open(output_dir / f"{args.split_name}_{args.min}-{args.max}.{tgt}", "w") as f:
        for txt in new_tgt_txt:
            f.write(txt + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-root", type=str, required=True)
    parser.add_argument("--output-dir", "-out", type=str, required=True)
    parser.add_argument("--lang-pair", "-l", type=str, required=True)
    parser.add_argument("--split-name", "-s", type=str, required=True)
    parser.add_argument("--min", "-min", type=float, required=True)
    parser.add_argument("--max", "-max", type=float, required=True)
    parser.add_argument("--force-the-same", action="store_true")
    args = parser.parse_args()

    create_modified_mt_split(args)
