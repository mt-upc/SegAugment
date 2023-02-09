import argparse
import json
from pathlib import Path
from tqdm import tqdm

import yaml
import numpy as np


def is_empty(txt: str) -> bool:
    return not txt or set(txt) == set(" ")


def overlaps(start1: float, end1: float, start2: float, end2: float) -> bool:
    """checks if two segments with start and end, overlap with each other

    Args:
        start1 (float): start of first segment
        end1 (float): end of first segment
        start2 (float): start of second segment
        end2 (float): end of second segment

    Returns:
        bool
    """
    if (start2 <= start1 < end2) or (start2 <= end1 < end2):
        return True
    elif (start1 <= start2) and (end1 > end2):
        return True
    else:
        False


def sort_segments(word_segments: list[dict]) -> list[dict]:
    """sorts the word_segments by their start time"""
    indices = np.argsort([word_sgm["start"] for word_sgm in word_segments])
    word_segments = [word_segments[i] for i in indices]
    return word_segments


def remove_failed(segments: list[dict], failed_segments: list[dict]) -> list[dict]:
    """remove segments that overlap with the failed_segments from the forced alignment

    Args:
        segments (list[dict]): segments of the original yaml
        failed_segments (list[dict]): segments that failed during forced alignment

    Returns:
        list[dict]: new segments
    """

    if not failed_segments:
        return segments

    new_segments = []
    for sgm in segments:
        is_ok = True
        for failed_sgm in failed_segments:
            if overlaps(
                sgm["offset"],
                sgm["offset"] + sgm["duration"],
                failed_sgm["start"],
                failed_sgm["end"],
            ):
                is_ok = False
                break

        if is_ok:
            new_segments.append(sgm)

    return new_segments


def get_text_for_segment(segment: dict, word_segments: list[dict]) -> tuple[str, str]:
    """finds the text that corresponds to this new segment based on the word_segments
    from the forced alignment

    Args:
        segment (dict): a segment from the new SHAS segmentation
        word_segments (list[dict]): result of the forced alignment

    Returns:
        tuple[str, str]: the clean (ASR-like) and original text that correspond to the segment
    """
    start = segment["offset"]
    end = segment["offset"] + segment["duration"]
    clean, original = [], []
    for word_sgm in word_segments:
        if word_sgm["start"] > start and word_sgm["start"] < end:
            clean.append(word_sgm["word"])
            original.append(word_sgm["text"])
        if word_sgm["end"] >= end:
            break
    clean = " ".join(clean)
    original = " ".join(original)
    return clean, original


def remove_open(txt: str, s_left: str, s_right: str) -> str:
    """removes "hanging" symbols from the text (if possible)

    Args:
        txt (str): text for a new segment
        s_left (str): left-side symbol
        s_right (str): right-side symbol

    Returns:
        str: text without the hanging symbols
    """
    n_left, n_right = txt.count(s_left), txt.count(s_right)
    if n_left != n_right and (n_left + n_right == 1):
        txt = txt.replace(s_left, "").replace(s_right, "")
    return txt


def post_process_text(txt: str) -> str:
    """cleans the text for a new segment
    by removing hanging symbols like apostrophes and parentheses
    capitalizing beginning of sentence
    adding or changing end-of-sentence punctuation

    Args:
        txt (str): the text for a new segment

    Returns:
        str: post-processed text for a new segment
    """

    n_apos = txt.count('"')
    if n_apos == 1:
        txt = txt.replace('"', "")

    txt = remove_open(txt, "(", ")")
    txt = remove_open(txt, "[", "]")
    txt = remove_open(txt, "“", "”")

    if txt.startswith("--"):
        txt = txt.split("--", maxsplit=1)[-1]
        txt = txt.strip()

    if is_empty(txt):
        return ""

    if txt[0].isalpha():
        txt = txt[0].upper() + txt[1:]
    elif txt[0] in ["¿", "¡"]:
        txt = txt[0] + txt[1].upper() + txt[2:]

    if not txt.endswith("."):
        if txt[-1].isalpha() or txt[-1].isdigit() or txt[-1] == "%":
            txt = txt + "."
        elif txt[-1] == ";":
            if len(txt) > 1 and txt[-2] == " ":
                txt = txt[:-2] + "."
            else:
                txt = txt[:-1] + "."
        elif txt[-1] in [",", ":", "-", " ", "—"]:
            if len(txt) > 1 and txt[-2] == " ":
                txt = txt[:-2] + "."
            else:
                txt = txt[:-1] + "."
        elif txt[-1] in ["?", '"', ")", "!", "…"]:
            pass
        else:
            pass

    if is_empty(txt):
        return ""

    return txt


def align_audio_source(args):
    path_to_new_yaml = Path(args.path_to_new_yaml)
    with open(path_to_new_yaml) as f:
        new_segmentation = yaml.load(f, Loader=yaml.CLoader)

    root, stem = path_to_new_yaml.parent, path_to_new_yaml.stem

    segments_per_talk = {}
    for segment in new_segmentation:
        talk_id = segment["wav"].split(".")[0]
        if talk_id not in segments_per_talk.keys():
            segments_per_talk[talk_id] = []
        segments_per_talk[talk_id].append(segment)

    dir = Path(args.path_to_forced_alignment_dir)

    extra = 0 if args.no_extra else 0.06

    asr_transcript = []
    reconstructed_transcript = []
    reconstructed_transcript_post = []
    segmentation_post = []
    for talk_id, talk_segments in tqdm(segments_per_talk.items()):
        alignment_file = dir / f"{talk_id}.json"
        if alignment_file.exists():
            with open(alignment_file) as f:
                forced_alignment_out = json.load(f)
        else:
            print(f"{alignment_file} not found. Skipping.")
            continue

        word_segments = forced_alignment_out["word_segments"]
        word_segments = sort_segments(word_segments)

        failed_segments = forced_alignment_out["failed_segments"]
        talk_segments = remove_failed(talk_segments, failed_segments)

        for segment in talk_segments:
            asr_txt, reconstructed_txt = get_text_for_segment(segment, word_segments)
            reconstructed_txt_post = post_process_text(
                reconstructed_txt, args.source_language
            )

            if reconstructed_txt_post:
                asr_transcript.append(asr_txt)
                reconstructed_transcript.append(reconstructed_txt)
                reconstructed_transcript_post.append(reconstructed_txt_post)
                segmentation_post.append(segment)

    for i in range(len(segmentation_post)):
        segmentation_post[i]["offset"] = round(
            segmentation_post[i]["offset"] - extra, 4
        )
        segmentation_post[i]["duration"] = round(
            segmentation_post[i]["duration"] + extra, 4
        )

    with open(root / f"{stem}.clean.{args.source_language}", "w") as f:
        for txt in asr_transcript:
            f.write(txt + "\n")

    with open(root / f"{stem}.{args.source_language}", "w") as f:
        for txt in reconstructed_transcript:
            f.write(txt + "\n")

    with open(root / f"{stem}.post.{args.source_language}", "w") as f:
        for txt in reconstructed_transcript_post:
            f.write(txt + "\n")

    with open(root / f"{stem}.post.yaml", "w") as f:
        yaml.dump(segmentation_post, f, default_flow_style=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-new-yaml", "-new_yaml", type=str, required=True)
    parser.add_argument(
        "--path-to-forced-alignment-dir", "-align", type=str, required=True
    )
    parser.add_argument("--source-language", "-lang", type=str, required=True)
    parser.add_argument("--no-extra", action="store_true")
    args = parser.parse_args()

    align_audio_source(args)
