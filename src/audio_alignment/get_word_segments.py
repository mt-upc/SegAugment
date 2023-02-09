import argparse
import json
from pathlib import Path

import forced_alignment
import numpy as np
import text_cleaning
import torch
import torchaudio
import yaml
from constants import SR, WAV2VEC_MODEL_NAME, N
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


class WavDataset(Dataset):
    def __init__(
        self,
        path_to_wav: Path,
        segments: list[dict],
        vocab: dict,
        lang: str,
        max_seconds_example: float,
    ):
        """dataset object for the original segments of a wav file

        Args:
            path_to_wav (Path): absolute path to the wav file
            segments (list[dict]): original segmentation for the wav file
            vocab (dict): vocabulary of wav2vec2.0 with mappings from chars to indices
            max_seconds_example (float): max length of an example
        """
        super().__init__()

        self.path_to_wav = path_to_wav
        self.segments = segments
        self.vocab = vocab
        self.lang = lang
        self.max_seconds_example = max_seconds_example

        self._filter_items()
        self._sort_items()

    def _filter_items(self):
        """removes long segments"""
        to_remove = []
        for idx in range(len(self.segments)):
            if self.segments[idx]["duration"] > self.max_seconds_example:
                to_remove.append(idx)

        self.long_segments = [
            {
                "start": self.segments[idx]["offset"],
                "end": self.segments[idx]["offset"] + self.segments[idx]["duration"],
                "flag": "long",
                "text": self.segments[idx]["text"],
            }
            for idx in to_remove
        ]

        self.segments = [
            self.segments[idx]
            for idx in range(len(self.segments))
            if idx not in to_remove
        ]

    def _sort_items(self):
        """sorts examples according to length"""
        durations = [sgm["duration"] for sgm in self.segments]
        self.segments = [self.segments[idx] for idx in np.argsort(durations)][::-1]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, indices: list[int]) -> tuple:
        """returns the batch for a list of indices"""

        offsets = [self.segments[index]["offset"] for index in indices]
        durations = [self.segments[index]["duration"] for index in indices]
        original_txts = [self.segments[index]["text"] for index in indices]

        wav_arrays = [
            torchaudio.backend.sox_io_backend.load(
                self.path_to_wav,
                int(offset * SR),
                int(duration * SR),
            )[0]
            for offset, duration in zip(offsets, durations)
        ]

        tokenized_cleaned_txts = [
            text_cleaning.tokenize_text(
                text_cleaning.clean_text(txt, self.lang), self.vocab, self.lang
            )
            for txt in original_txts
        ]

        return (wav_arrays, original_txts, tokenized_cleaned_txts, offsets, durations)

    def my_collate_fn(self, batch: tuple) -> tuple:
        """some necessary corrections to the format of the batch"""
        batch = batch[0]
        wav_arrays = [wav_array.numpy()[0] for wav_array in batch[0]]
        return wav_arrays, batch[1], batch[2], batch[3], batch[4]


class DurationBatchSampler(Sampler):
    def __init__(self, durations: list[float], max_seconds_batch: float):
        """creates batches of the indices of a dataset according to the
        max_seconds_batch

        Args:
            durations (list[float]): durations of the segments in the dataset
            max_seconds_batch (float): maximum seconds within a batch
        """
        super().__init__(durations)
        self.durations = durations
        self.max_seconds_batch = max_seconds_batch

    def __iter__(self):
        batches = []
        batch_total, batch = 0, []
        for idx, duration in enumerate(self.durations):
            if batch_total + duration < self.max_seconds_batch:
                batch.append(idx)
                batch_total += duration
            else:
                batches.append(batch)

                batch_total = duration
                batch = [idx]

        if batch:
            batches.append(batch)

        return iter(batches)


def load_data(path_to_yaml: Path, path_to_txt: Path) -> dict:
    """loads and combines the original segmentation and text for a dataset

    Args:
        path_to_yaml (Path): path to original segmentation
        path_to_txt (Path): path to original text

    Returns:
        dict: a list of segments for each talk
    """

    with open(path_to_yaml) as f:
        segments = yaml.load(f, Loader=yaml.CLoader)

    with open(path_to_txt) as f:
        text = f.read().splitlines()

    segments_per_talk = {}
    for txt, segment in zip(text, segments):
        talk_id = segment["wav"].split(".")[0]
        sgm_info = {
            "duration": segment["duration"],
            "offset": segment["offset"],
            "text": txt,
        }
        if not sgm_info["text"]:
            continue
        if talk_id not in segments_per_talk.keys():
            segments_per_talk[talk_id] = [sgm_info]
        else:
            segments_per_talk[talk_id].append(sgm_info)

    return segments_per_talk


def get_word_segments_for_wav(
    path_to_wav: Path,
    segments: list[dict],
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    vocab: dict,
    lang: str,
    device: torch.device,
    max_seconds_example: float,
    max_seconds_batch: float,
) -> tuple[list[forced_alignment.Segment], list[dict]]:
    """does forced-alignment with wav2vec2.0 for a wav file

    Args:
        path_to_wav (Path): path to wav file
        segments (list[dict]): list of original segments
        model (Wav2Vec2ForCTC): wav2vec2.0 model
        processor (Wav2Vec2Processor): wav2vec2.0 processor
        vocab (dict): wav2vec2.0 vocabulary
        device (torch.device): cuda device
        max_seconds_example (float): maximum length of an example
        max_seconds_batch (float): maximum length of the examples in a batch

    Returns:
        tuple[list[alignment.Segment], list[str]]: the output of the forced-alignment
        and a list of failed segments (either long or failed)
    """

    dataset = WavDataset(path_to_wav, segments, vocab, lang, max_seconds_example)
    batch_sampler = DurationBatchSampler(
        [sgm["duration"] for sgm in dataset.segments], max_seconds_batch
    )
    dataloader = DataLoader(
        dataset, sampler=batch_sampler, collate_fn=dataset.my_collate_fn, num_workers=0
    )

    all_word_segments, failed_segments = [], []
    with torch.no_grad():
        for (
            audios,
            original_texts,
            tokenized_cleaned_texts,
            offsets,
            durations,
        ) in iter(dataloader):
            tokenized_audio = processor(
                audios,
                return_tensors="pt",
                padding="longest",
                sampling_rate=SR,
            )
            input_values = tokenized_audio.input_values.to(device)
            attention_mask = tokenized_audio.attention_mask.to(device)
            logits = model(input_values, attention_mask=attention_mask).logits
            emissions = torch.log_softmax(logits, dim=-1).detach().cpu()

            for (
                emission,
                attn_mask,
                original_txt,
                tokenized_cleaned_txt,
                offset,
                duration,
            ) in zip(
                emissions,
                attention_mask,
                original_texts,
                tokenized_cleaned_texts,
                offsets,
                durations,
            ):
                # mapping for clean (ASR-like) to original text tokens
                clean2original = forced_alignment.get_monolingual_alignments(
                    original_txt.split(), tokenized_cleaned_txt.split("|"), lang
                )

                if tokenized_cleaned_txt == "":
                    # only one option
                    all_word_segments.append(
                        forced_alignment.Segment(
                            "", offset, offset + duration, -1.0, clean2original[0]
                        )
                    )
                    continue

                true_len = int(attn_mask.sum().item() / N)
                emission = emission[:true_len]

                tokens = [vocab[c] for c in tokenized_cleaned_txt]

                # find path
                trellis = forced_alignment.get_trellis(emission, tokens)
                path = forced_alignment.backtrack(trellis, emission, tokens)

                if not path:
                    failed_segments.append(
                        {
                            "start": offset,
                            "end": offset + duration,
                            "flag": "failed",
                            "text": original_txt,
                        }
                    )
                    continue

                char_segments = forced_alignment.merge_repeats(
                    path, tokenized_cleaned_txt
                )
                word_segments = forced_alignment.merge_words(char_segments, offset)

                # add corresponding original text
                for i, original_txt in clean2original.items():
                    word_segments[i].original_label = original_txt

                # merge segments with the same original text
                word_segments = forced_alignment.merge_original(word_segments)

                all_word_segments.append(word_segments)

    # type correction
    all_word_segments = [
        word_segments if type(word_segments) == list else [word_segments]
        for word_segments in all_word_segments
    ]
    # flatten
    all_word_segments = [
        word_sgm for word_segments in all_word_segments for word_sgm in word_segments
    ]
    # fix order
    indices = np.argsort([word_sgm.start for word_sgm in all_word_segments])
    all_word_segments = [all_word_segments[idx] for idx in indices]
    # combine failed
    failed_segments = dataset.long_segments + failed_segments

    return all_word_segments, failed_segments


def get_word_segments(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    wav2vec_model_name = WAV2VEC_MODEL_NAME[args.language_code]
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name).eval().to(device)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(wav2vec_model_name)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    wav_dir = Path(args.path_to_wav)
    out_dir = Path(args.path_to_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments_per_talk = load_data(args.path_to_yaml, args.path_to_txt)

    print("Iterating through talks ...")
    for talk_id, talk_segments in tqdm(segments_per_talk.items()):
        out_file = out_dir / f"{talk_id}.json"

        if not out_file.exists() or args.override_files:
            wav_file = wav_dir / f"{talk_id}.wav"

            try:
                word_segments, failed_segments = get_word_segments_for_wav(
                    wav_file,
                    talk_segments,
                    model,
                    processor,
                    tokenizer.encoder,
                    args.language_code,
                    device,
                    args.max_seconds_example,
                    args.max_seconds_batch,
                )
            except RuntimeError:
                print(f"Failed forced-alignment, skipping file: {wav_file}")
                continue

            word_segments = [
                {
                    "start": round(w_sgm.start, 2),
                    "end": round(w_sgm.end, 2),
                    "word": w_sgm.label,
                    "text": w_sgm.original_label,
                }
                for w_sgm in word_segments
            ]

            with open(out_file, "w") as f:
                json.dump(
                    {
                        "word_segments": word_segments,
                        "failed_segments": failed_segments,
                    },
                    f,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-code", "-lang", type=str, required=True)
    parser.add_argument("--path-to-wav", "-wav", type=str, required=True)
    parser.add_argument("--path-to-txt", "-txt", type=str, required=True)
    parser.add_argument("--path-to-yaml", "-yaml", type=str, required=True)
    parser.add_argument("--path-to-output-dir", "-out", type=str, required=True)
    parser.add_argument("--max-seconds-example", "-max1", type=float, default=60)
    parser.add_argument("--max-seconds-batch", "-max2", type=float, default=60)
    parser.add_argument("--override-files", "-ovr", action="store_true")
    args = parser.parse_args()

    get_word_segments(args)
