"""
This script provides some utility functions that can be used
during inference of ASR models.

The intended use is to import `normalize_words` after decoding and tokenization.
Note that this will only work, when UPPERCASE letters are used throughout the recipe.

Author
------
Dominik Wagner 2022
"""
import re
import os
import csv
import string
from collections import defaultdict


def read_glm_csv(save_folder):
    """Load the ARPA Hub4-E and Hub5-E alternate spellings and contractions map"""

    alternatives_dict = defaultdict(list)
    with open(os.path.join(save_folder, "glm.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            alternatives = row[1].split("|")
            alternatives_dict[row[0]] += alternatives
    return alternatives_dict


def expand_contractions(text) -> list:
    """
    Some regular expressions for expanding common contractions and for splitting linked words.

    Parameters
    ----------
    text : str
        Text to process

    Returns
    -------
    A list of tokens
    """
    # Specific contractions
    text = re.sub(r"won\'t", "WILL NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"can\'t", "CAN NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"let\'s", "LET US", text, flags=re.IGNORECASE)
    text = re.sub(r"ain\'t", "AM NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"y\'all", "YOU ALL", text, flags=re.IGNORECASE)
    text = re.sub(r"can\'t", "CANNOT", text, flags=re.IGNORECASE)
    text = re.sub(r"can not", "CANNOT", text, flags=re.IGNORECASE)
    text = re.sub(r"\'cause", "BECAUSE", text, flags=re.IGNORECASE)
    text = re.sub(r"thats", "THAT IS", text, flags=re.IGNORECASE)
    text = re.sub(r"dont", "DO NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"hes", "HE IS", text, flags=re.IGNORECASE)
    text = re.sub(r"shes", "SHE IS", text, flags=re.IGNORECASE)
    text = re.sub(r"wanna", "WANT TO", text, flags=re.IGNORECASE)
    text = re.sub(r"theyd", "THEY WOULD", text, flags=re.IGNORECASE)
    text = re.sub(r"theyre", "THEY ARE", text, flags=re.IGNORECASE)
    text = re.sub(r"hed", "HE WOULD", text, flags=re.IGNORECASE)
    text = re.sub(r"shed", "SHE WOULD", text, flags=re.IGNORECASE)
    text = re.sub(r"wouldve", "WOULD HAVE", text, flags=re.IGNORECASE)
    text = re.sub(r"couldve", "COULD HAVE", text, flags=re.IGNORECASE)
    text = re.sub(r"couldnt", "COULD NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"cant", "CAN NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"shouldve", "SHOULD HAVE", text, flags=re.IGNORECASE)
    text = re.sub(r"oclock", "O CLOCK", text, flags=re.IGNORECASE)
    text = re.sub(r"o'clock", "O CLOCK", text, flags=re.IGNORECASE)
    text = re.sub(r"didn", "DID NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"didnt", "DID NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"im", "I AM", text, flags=re.IGNORECASE)
    text = re.sub(r"ive", "I HAVE", text, flags=re.IGNORECASE)
    text = re.sub(r"youre", "YOU ARE", text, flags=re.IGNORECASE)

    # More general contractions
    text = re.sub(r"n\'t", " NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"\'re", " ARE", text, flags=re.IGNORECASE)
    text = re.sub(r"\'s", " IS", text, flags=re.IGNORECASE)
    text = re.sub(r"\'d", " WOULD", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ll", " WILL", text, flags=re.IGNORECASE)
    text = re.sub(r"\'t", " NOT", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ve", " HAVE", text, flags=re.IGNORECASE)
    text = re.sub(r"\'m", " AM", text, flags=re.IGNORECASE)

    # Split linked words
    if "VOCALIZED" not in text:
        text = re.sub(r"-", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s\s+", " ", text)
    text = text.split()
    return text


def expand_contractions_batch(text_batch):
    """
    Wrapper that handles a batch of predicted or
    target words for contraction expansion
    """
    parsed_batch = []
    for batch in text_batch:
        # Remove incomplete words
        batch = [t for t in batch if not t.startswith("-")]
        # Expand contractions
        batch = [expand_contractions(t) for t in batch]
        # Flatten list of lists
        batch_expanded = [i for sublist in batch for i in sublist]
        parsed_batch.append(batch_expanded)
    return parsed_batch


def normalize_words(
    target_words_batch, predicted_words_batch, glm_alternatives=None
):
    """
    Remove some references and hypotheses we don't want to score.
    We remove incomplete words (i.e. words that start with "-"),
    expand common contractions (e.g. I'v -> I have),
    and split linked words (e.g. pseudo-rebel -> pseudo rebel).
    Then we check if some of the predicted words have mapping rules according
    to the glm (alternatives) file.
    Finally, we check if a predicted word is on the exclusion list.
    The exclusion list contains stuff like "MM", "HM", "AH", "HUH", which would get mapped,
    into hesitations by the glm file anyway.
    The goal is to remove all the things that appear in the reference as optional/deletable
    (i.e. inside parentheses).
    If we delete these tokens, there is no loss,
    and if we recognize them correctly, there is no gain.

    The procedure is adapted from Kaldi's local/score.sh script.

    Parameters
    ----------
    glm_alternatives : dict
        Dictionary containing valid word alternatives
    target_words_batch : list
        List of length <batch_size> containing lists of target words for each utterance
    predicted_words_batch : list of list
        List of length <batch_size> containing lists of predicted words for each utterance

    Returns
    -------

    A new list containing the filtered predicted words.

    """
    excluded_words = [
        "<UNK>",
        "UH",
        "UM",
        "EH",
        "MM",
        "HM",
        "AH",
        "HUH",
        "HA",
        "ER",
        "OOF",
        "HEE",
        "ACH",
        "EEE",
        "EW",
    ]

    target_words_batch = expand_contractions_batch(target_words_batch)
    predicted_words_batch = expand_contractions_batch(predicted_words_batch)

    # Find all possible alternatives for each word in the target utterance
    alternative2tgt_word_batch = []
    for tgt_utterance in target_words_batch:
        alternative2tgt_word = defaultdict(str)
        if glm_alternatives is not None:
            for tgt_wrd in tgt_utterance:
                alts = glm_alternatives[tgt_wrd]
                for alt in alts:
                    if alt != tgt_wrd and len(alt) > 0:
                        alternative2tgt_word[alt] = tgt_wrd
        alternative2tgt_word_batch.append(alternative2tgt_word)

    # See if a predicted word is on the exclusion list,
    # and if it matches one of the valid alternatives.
    # Also do some cleaning.
    checked_predicted_words_batch = []
    for i, pred_utterance in enumerate(predicted_words_batch):
        alternative2tgt_word = alternative2tgt_word_batch[i]
        checked_predicted_words = []
        for pred_wrd in pred_utterance:
            # Remove stuff like [LAUGHTER]
            pred_wrd = re.sub(r"\[.*?\]", "", pred_wrd)
            # Remove any remaining punctuation
            pred_wrd = pred_wrd.translate(
                str.maketrans("", "", string.punctuation)
            )
            # Sometimes things like LAUGHTER get appended to existing words e.g. THOUGHLAUGHTER
            if pred_wrd != "LAUGHTER" and pred_wrd.endswith("LAUGHTER"):
                pred_wrd = pred_wrd.replace("LAUGHTER", "")
            if pred_wrd != "NOISE" and pred_wrd.endswith("NOISE"):
                pred_wrd = pred_wrd.replace("NOISE", "")
            if pred_wrd.endswith("VOCALIZED"):
                pred_wrd = pred_wrd.replace("VOCALIZED", "")
            # Check word exclusion list
            if pred_wrd in excluded_words:
                continue
            # Finally, check word alternatives
            tgt_wrd = alternative2tgt_word[pred_wrd]
            if len(tgt_wrd) > 0:
                pred_wrd = tgt_wrd
            if len(pred_wrd) > 0:
                checked_predicted_words.append(pred_wrd)
        checked_predicted_words_batch.append(checked_predicted_words)
    return target_words_batch, checked_predicted_words_batch
