import html
import re
import string

import nltk
from constants import LANG_AND, LANG_CODES
from num2words import num2words


def clean_speaker_name(text: str, lang: str) -> str:
    """removes speaker name that might appear in the beginning of the sentence

    Args:
        text (str): text
        lang (str): language id

    Returns:
        str: text without speaker name
    """
    if ": " in text:
        for sentence in nltk.sent_tokenize(text, language=LANG_CODES[lang]):
            if ": " in sentence:
                start_text, rest_text = sentence.split(": ", maxsplit=1)
                start_tokens = re.sub(" +", " ", start_text).strip().split(" ")
                num_start_tokens = len(start_tokens)

                # XXX: one word, initials, all caps
                if num_start_tokens == 1 and start_tokens[0].isupper():
                    text = text.replace(sentence, rest_text)
                # Xxxx (Zzzz) Yyyy: two or three words, first (middle) last, start of each name is capitalized
                elif 1 < num_start_tokens < 4 and all(
                    [start_tokens[i][0].isupper() for i in range(num_start_tokens)]
                ):
                    text = text.replace(sentence, rest_text)

    return text


def clean_event(text: str, lang: str) -> str:
    """removes event (e.g. laughter, applause) that appears enclosed in parenthesis

    Args:
        text (str): text
        lang (str): language id

    Returns:
        str: text without events
    """

    # just parenthesis
    simple_event_pattern = r"\([^()]*\)"

    if ": " in text:
        for event in re.findall(simple_event_pattern, text):
            # check if event contains actual text from a speaker: (XX: utterance) -> utterance
            if ": " in event:
                event_text = event[1:-1]  # (xyz) -> xyz
                event_text_cleaned = clean_speaker_name(event_text, lang)

                # replace event with its cleaned text
                if event_text != event_text_cleaned:
                    text = text.replace(event, event_text_cleaned)

    # remove rest of the events
    # parenthesis with punctuations, " . ... :, before or after
    all_event_patterns = r'"(\([^()]*\))"|"(\([^()]*\))|(\([^()]*\):)|(\([^()]*\)\.\.\.)|(\([^()]*\)\.)|(\([^()]*\))'
    text = re.sub(all_event_patterns, "", text)

    text = text.replace(" -- -- ", " -- ")

    return text


def my_num2words(token: str, lang: str) -> str:
    """spells out numbers in a string

    Args:
        token (str): part of text
        lang (str): language id

    Returns:
        str: part of text with spelled-out numbers
    """
    if token.isdigit():
        new_token = token
    elif token.translate(str.maketrans("", "", string.punctuation)).isdigit():
        new_token = token.translate(str.maketrans("", "", string.punctuation))
    else:
        new_token = token
    if new_token.isdigit():
        if len(new_token) == 4 and (new_token in token) and int(new_token[0]) < 3:
            new_token = num2words(new_token, to="year", lang=lang)
        else:
            for p in ["⁰¹²³⁴⁵⁶⁷⁸⁹"]:
                new_token = new_token.replace(p, "")
            new_token = num2words(new_token, lang=lang)
    return new_token


def handle_html_non_utf(txt: str, lang: str) -> str:
    """handles html and non-utf chars"""
    and_token = LANG_AND[lang]
    txt = html.unescape(bytes(txt, "utf-8").decode("utf-8", "ignore"))
    txt = txt.replace(" & ", and_token).replace("&", and_token)
    return txt


def clean_text(txt: str, lang: str) -> str:
    """Applies many cleaning functions and space removal to a string

    Args:
        txt (str): text
        lang (str): language id

    Returns:
        str: cleaned text
    """
    txt = re.sub(" +", " ", txt.strip().replace("\t", " ").replace("\n", " "))
    txt = handle_html_non_utf(txt, lang)
    txt = " ".join([my_num2words(token, lang) for token in txt.split()])
    txt = clean_event(txt, lang)
    txt = clean_speaker_name(txt, lang)
    txt = " ".join(txt.split())
    txt = txt.strip()
    if set(txt) != set(" "):
        return txt
    else:
        return ""


def tokenize_text(cleaned_txt: str, vocab: dict, lang: str) -> str:
    """tokenizes the output of "clean_text" according to the wav2vec2.0 vocab

    Args:
        cleaned_txt (str): output of the clean_text functions
        vocab (dict): wav2vec2.0 vocab with mappings from chars to indices
        lang (str): language id

    Returns:
        str: tokenized text tokens separated by |
    """
    if lang == "en":
        # wav2vec2.0
        cleaned_txt = cleaned_txt.upper()
    else:
        # XLS-R
        cleaned_txt = cleaned_txt.lower()

    tokenized_cleaned_txt = "".join(
        [c if c in vocab.keys() else "|" for c in cleaned_txt.replace(" ", "|")]
    )

    if set(tokenized_cleaned_txt) == set("|") or not tokenized_cleaned_txt:
        return ""

    while "||" in tokenized_cleaned_txt:
        tokenized_cleaned_txt = tokenized_cleaned_txt.replace("||", "|")
    if tokenized_cleaned_txt[-1] == "|":
        tokenized_cleaned_txt = tokenized_cleaned_txt[:-1]
    if tokenized_cleaned_txt[0] == "|":
        tokenized_cleaned_txt = tokenized_cleaned_txt[1:]

    return tokenized_cleaned_txt
