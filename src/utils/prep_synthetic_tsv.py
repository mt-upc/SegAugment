# adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/prep_mustc_data.py

import argparse
import os
import shutil
import sys
from functools import partial
from itertools import groupby
from math import floor
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

sys.path.append(os.environ["FAIRSEQ_ROOT"])
from examples.speech_to_text.data_utils import (
    create_zip,
    filter_manifest_df,
    get_zip_manifest,
    save_df_to_tsv,
)
from fairseq.data.audio.audio_utils import (
    _get_torchaudio_fbank,
    convert_waveform,
    get_waveform,
)
from torch.utils.data import Dataset
from tqdm import tqdm

CHUNKSIZE = 500
MULT = 0.5


class AudioDataset(Dataset):
    def __init__(
        self, wav_root: str, txt_root: str, src_lang: str, tgt_lang: str
    ) -> None:
        # Load audio segments
        with open(txt_root / "new.post.yaml") as f:
            segments = yaml.load(f, Loader=yaml.CLoader)
        # Load source and target utterances
        for _lang in [src_lang, tgt_lang]:
            txt_file_path = txt_root / f"new.post.{_lang}"
            if txt_file_path.is_file():
                with open(txt_file_path) as f:
                    utterances = [r.strip() for r in f]
                assert len(segments) == len(utterances)
                for i, u in enumerate(utterances):
                    segments[i][_lang] = u
            else:
                for i in range(len(segments)):
                    segments[i][_lang] = "NA"
        # Gather info
        self.data = []
        for wav_filename, _seg_group in tqdm(groupby(segments, lambda x: x["wav"])):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src_lang],
                        segment[tgt_lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def extract_features_and_save(
    dataset: AudioDataset,
    audio_root: Path,
    index: int,
) -> None:
    waveform, sample_rate, _, _, _, utt_id = dataset[index]
    _waveform, _ = convert_waveform(waveform, sample_rate, to_mono=True)
    _waveform = _waveform * (2**15)
    _waveform = _waveform.numpy()
    features = _get_torchaudio_fbank(_waveform, sample_rate)
    np.save(audio_root / f"{utt_id}.npy", features)


def get_utt_manifest(
    dataset: AudioDataset,
    audio_paths: Dict[str, str],
    audio_lengths: Dict[str, int],
    index: int,
) -> Dict[str, Union[str, int]]:
    _, _, src_utt, tgt_utt, speaker_id, utt_id = dataset[index]
    return {
        "id": utt_id,
        "audio": audio_paths[utt_id],
        "n_frames": audio_lengths[utt_id],
        "src_text": src_utt,
        "tgt_text": tgt_utt,
        "speaker": speaker_id,
    }


def process(args):
    dataset_root = Path(args.original_dataset_root)
    dataset_name = dataset_root.name
    lang_pair = f"{args.src_lang}-{args.tgt_lang}"
    out_dir = Path(os.environ["OUTPUT_ROOT"])
    split = "train"

    wav_root = dataset_root / lang_pair / "data" / split / "wav"
    txt_root = out_dir / "synthetic_data" / dataset_name / lang_pair / args.ell / split

    audio_root = txt_root / "fbank80"

    zip_path = txt_root / f"{audio_root.name}.zip"
    dataset = None
    num_workers = floor(MULT * len(os.sched_getaffinity(0)))
    parallel = num_workers > 2 and not args.not_parallel

    if parallel:
        print(f"Parallelizing with {num_workers} workers")

    if not zip_path.is_file():
        audio_root.mkdir(exist_ok=True)

        print("Creating Dataset...")
        dataset = AudioDataset(wav_root, txt_root, args.src_lang, args.tgt_lang)

        print("Extracting and saving features...")
        if parallel:
            _extract_features_and_save = partial(
                extract_features_and_save, dataset, audio_root
            )
            with Pool(num_workers) as p:
                _ = list(
                    tqdm(
                        p.imap(
                            _extract_features_and_save,
                            range(len(dataset)),
                            chunksize=CHUNKSIZE,
                        ),
                        total=len(dataset),
                    )
                )
        else:
            for i in tqdm(range(len(dataset))):
                extract_features_and_save(dataset, audio_root, i)

        print("ZIPing audios/features...")
        create_zip(audio_root, zip_path)

    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(
        zip_path, is_audio=False, chunksize=CHUNKSIZE, num_workers=num_workers
    )

    print("Generating manifest...")
    if dataset is None:
        dataset = AudioDataset(wav_root, txt_root, args.src_lang, args.tgt_lang)
    if parallel:
        _get_utt_manifest = partial(
            get_utt_manifest, dataset, audio_paths, audio_lengths
        )
        with Pool(num_workers) as p:
            manifest = list(
                tqdm(
                    p.imap(
                        _get_utt_manifest,
                        list(range(len(dataset))),
                        chunksize=CHUNKSIZE,
                    ),
                    total=len(dataset),
                )
            )
    else:
        manifest = list(
            get_utt_manifest(dataset, audio_paths, audio_lengths, i)
            for i in tqdm(range(len(dataset)))
        )
    df = pd.DataFrame.from_records(manifest)

    df_asr = df[["id", "audio", "n_frames", "src_text", "speaker"]].copy()
    df_asr.rename(columns={"src_text": "tgt_text"}, inplace=True)
    df_asr["tgt_lang"] = f"{args.src_lang}_{args.ell}"
    df_asr = filter_manifest_df(df_asr, is_train_split=False)
    save_df_to_tsv(df_asr, txt_root / f"{split}_{txt_root.parent.name}_asr.tsv")

    if args.src_lang != args.tgt_lang:
        df_st = df[["id", "audio", "n_frames", "tgt_text", "speaker"]]
        df_st["tgt_lang"] = f"{args.tgt_lang}_{args.ell}"
        df_st = filter_manifest_df(df_st, is_train_split=False)
        save_df_to_tsv(df_st, txt_root / f"{split}_{txt_root.parent.name}_st.tsv")

    # Clean up
    if audio_root.is_dir():
        shutil.rmtree(audio_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset-root", "-data", required=True, type=str)
    parser.add_argument("--src-lang", "-src", required=True, type=str)
    parser.add_argument("--tgt-lang", "-tgt", required=True, type=str)
    parser.add_argument("--ell", "-ell", required=True, type=str)
    parser.add_argument("--not-parallel", action="store_true")
    args = parser.parse_args()

    process(args)
