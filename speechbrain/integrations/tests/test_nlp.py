"""Tests for NLP integrations

Authors
 * Titouan Parcollet (2025)
"""

import math


def test_bleu(device):
    """Test if our bleu metric stats gives the same results as sacrebleu"""

    from sacrebleu.metrics import BLEU

    refs = [
        [
            "The dog bit the man.",
            "It was not unexpected.",
            "The man bit him first.",
        ],
        [
            "The dog had bit the man.",
            "No one was surprised.",
            "The man had bitten the dog.",
        ],
    ]
    sys = [
        "The dog bit the man.",
        "It wasn't surprising.",
        "The man had just bitten him.",
    ]

    sacrebleu = BLEU()
    scores = sacrebleu.corpus_score(sys, refs)
    bleu = scores.score

    from speechbrain.integrations.nlp.bleu import BLEUStats

    sb_bleu = BLEUStats()
    ids = ["utterance1", "utterance2", "utterance3"]
    sb_bleu.append(ids=ids, predict=sys, targets=refs)
    stats = sb_bleu.summarize()

    assert math.isclose(bleu, stats["BLEU"], rel_tol=1e-5)

    # Expanding by one
    refs = [
        [
            "The dog bit the man.",
            "It was not unexpected.",
            "The man bit him first.",
            "but the care wasn't red.",
        ],
        [
            "The dog had bit the man.",
            "No one was surprised.",
            "The man had bitten the dog.",
            "but the care is red",
        ],
    ]
    sys = [
        "The dog bit the man.",
        "It wasn't surprising.",
        "The man had just bitten him.",
        "But the car is not red",
    ]

    sacrebleu = BLEU()
    scores = sacrebleu.corpus_score(sys, refs)
    bleu = scores.score

    ids = ["utterance4"]
    refs = [["but the care wasn't red."], ["but the care is red"]]
    sys = ["But the car is not red"]
    sb_bleu.append(ids=ids, predict=sys, targets=refs)
    stats = sb_bleu.summarize()

    assert math.isclose(bleu, stats["BLEU"], rel_tol=1e-5)
