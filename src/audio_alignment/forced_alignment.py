from dataclasses import dataclass

import torch
from fuzzywuzzy import fuzz

import text_cleaning


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float
    original_label: str = ""

    def __repr__(self):
        return f"{self.label} -> {self.original_label}\t({self.score:4.2f}): [{self.start}, {self.end})"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def get_trellis(emission, tokens, blank_id=0):
    """taken from https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html"""
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    """taken from https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html"""
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return []
    return path[::-1]


def merge_repeats(path, txt):
    """taken from https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html"""
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                txt[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments: list[Segment], ofs: float, separator="|") -> list[Segment]:
    """taken from https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html"""
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(
                        word,
                        segments[i1].start / 50 + ofs,
                        segments[i2 - 1].end / 50 + ofs,
                        score,
                    )
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def merge_original(segments: list[Segment]) -> list[Segment]:
    """merges neighbouring segments according their original text

    Args:
        segments (list[Segment]): list of Segment objects (output of "merge_words")

    Returns:
        list[Segment]: new (merged) list of Segments
    """
    new_segments = []
    i = 0
    while i < len(segments):
        j = i + 1
        start = segments[i].start
        end = segments[i].end
        scores = [segments[i].score]
        labels = [segments[i].label]
        original_labels = segments[i].original_label.split()
        while j < len(segments) and segments[j].original_label in original_labels:
            end = segments[j].end
            scores.append(segments[j].score)
            labels.append(segments[j].label)
            j += 1
        new_segments.append(
            Segment(
                "|".join(labels),
                start,
                end,
                sum(scores) / len(scores),
                " ".join(original_labels),
            )
        )
        i = j
    return new_segments


def get_monolingual_alignments(
    original_tokens: list[str], clean_tokens: list[str], lang: str
) -> dict[int, str]:
    """finds mappings from the clean (ASR-like output) tokens
    to the original (unmodified) text tokens

    Args:
        original_tokens (list[str]): original tokens for a segment
        clean_tokens (list[str]): clean tokens for a segment
        lang (str): language id of the segment

    Returns:
        dict[int, str]: index of the clean tokens to the corresponding
            string of the original text
    """

    def get_original(idx=None, token=None):
        if token is None:
            token = original_tokens[idx]
        token = text_cleaning.handle_html_non_utf(token, lang)
        token = text_cleaning.my_num2words(token, lang)
        return token.lower()

    def get_clean(idx):
        return clean_tokens[idx].lower()

    def add_alignment(alignment, i, token_j):
        if i in alignment.keys():
            alignment[i].append(token_j)
        else:
            alignment[i] = [token_j]
        return alignment

    if clean_tokens == [""]:
        alignment = {0: " ".join(original_tokens)}
        return alignment

    alignment = {}
    j1, j2, i = 0, 0, 0
    while i < len(clean_tokens):
        queue = []
        original_tokens_iter = list(range(len(original_tokens)))[j1:]
        for j1 in original_tokens_iter:
            if i < len(clean_tokens) and get_clean(i) in get_original(j1):
                queue.append(j1)
                for j2 in queue:
                    alignment = add_alignment(alignment, i, original_tokens[j2])
                queue = []
                i += 1
                while i < len(clean_tokens) and get_clean(i) in get_original(j1):
                    alignment = add_alignment(alignment, i, original_tokens[j1])
                    i += 1
            else:
                queue.append(j1)
    for j2 in queue:
        alignment = add_alignment(alignment, i - 1, original_tokens[j2])

    # corrections
    for i in list(alignment.keys())[:-1]:
        clean_token = clean_tokens[i]
        if len(clean_token) < 3:
            if len(alignment[i + 1]) > 1:
                score = fuzz.ratio(
                    get_original(token=alignment[i][-1]), clean_token.lower()
                )
                score_next = fuzz.ratio(
                    get_original(token=alignment[i + 1][0]), clean_token.lower()
                )
                if score_next > score:
                    alignment[i][-1] = alignment[i + 1][0]
                    del alignment[i + 1][0]

    for k, v in alignment.items():
        alignment[k] = " ".join(v)

    return alignment
