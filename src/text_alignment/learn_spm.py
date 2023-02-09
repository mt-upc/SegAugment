import os
import sys

sys.path.append(os.environ["FAIRSEQ_ROOT"])

from pathlib import Path

from examples.speech_to_text.data_utils import gen_vocab

text = sys.argv[1]
vocab = sys.argv[2]
vocab_type = sys.argv[3]
vocab_size = sys.argv[4]

gen_vocab(
    Path(text),
    Path(vocab),
    vocab_type,
    vocab_size,
)
