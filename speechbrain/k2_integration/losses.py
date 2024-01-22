""" This file contains the loss functions for k2 training. Currently, we only
support CTC loss.

Authors:
 * Pierre Champion 2023
 * Zeyu Zhao 2023
 * Georgios Karakasidis 2023
"""

from . import k2  # import k2 from ./__init__.py

import torch


def ctc_k2(
    log_probs,
    input_lens,
    graph_compiler,
    texts,
    reduction="mean",
    beam_size=10,
    use_double_scores=True,
    is_training=True,
):
    """
    CTC loss implemented with k2. Make sure that k2 has been installed properly.
    Note that the blank index must be 0 in this implementation.

    Arguments
    ---------
    log_probs: torch.Tensor
        Log-probs of shape (batch, time, num_classes).
    input_lens : torch.Tensor
        Length of each utterance.
    graph_compiler : k2.Fsa
        Decoding graph.
    texts : List[str]
        List of texts.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'none'.
        See k2.ctc_loss for 'mean', 'sum', 'none'.
    beam_size : int
        Beam size.
    use_double_scores : bool
        If true, use double precision for scores.
    is_training : bool
        If true, the returned loss requires gradient.

    Returns
    -------
    loss: torch.Tensor
        CTC loss.

    Example
    -------
    >>> import torch
    >>> from speechbrain.k2_integration.losses import ctc_k2
    >>> from speechbrain.k2_integration.graph_compiler import CtcGraphCompiler
    >>> from speechbrain.k2_integration.lexicon import Lexicon
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang

    >>> # Create a random batch of log-probs
    >>> batch_size = 4

    >>> log_probs = torch.randn(batch_size, 100, 30)
    >>> log_probs.requires_grad = True
    >>> # Assume all utterances have the same length so no padding was needed.
    >>> input_lens = torch.ones(batch_size)
    >>> # Create a samll lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = "hello h e l l o\\nworld w o r l d\\n<UNK> <unk>"
    >>> lexicon_file = lang_tmpdir.join("lexicon.txt")
    >>> lexicon_file.write(lexicon_sample)
    >>> # Create a lang directory with the lexicon and L.pt, L_inv.pt, L_disambig.pt
    >>> prepare_lang(lang_tmpdir)
    >>> # Create a lexicon object
    >>> lexicon = Lexicon(lang_tmpdir)
    >>> # Create a random decoding graph
    >>> graph = CtcGraphCompiler(
    ...     lexicon,
    ...     log_probs.device,
    ... )
    >>> # Create a random batch of texts
    >>> texts = ["hello world", "world hello", "hello", "world"]
    >>> # Compute the loss
    >>> loss = ctc_k2(
    ...     log_probs=log_probs,
    ...     input_lens=input_lens,
    ...     graph_compiler=graph,
    ...     texts=texts,
    ...     reduction="mean",
    ...     beam_size=10,
    ...     use_double_scores=True,
    ...     is_training=True,
    ... )
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()

    batch_size = log_probs.shape[0]

    supervision_segments = torch.tensor(
        [[i, 0, input_lens[i]] for i in range(batch_size)],
        device="cpu",
        dtype=torch.int32,
    )

    decoding_graph, target_lens = graph_compiler.compile(
        texts, is_training=is_training
    )

    # An introduction to DenseFsaVec:
    # https://k2-fsa.github.io/k2/core_concepts/index.html#dense-fsa-vector
    # It could be viewed as a fsa-type log_probs,
    # whose weight on the arcs are initialized with log_probs.
    # The goal of converting tensor-type to fsa-type is using
    # fsa related functions in k2. e.g. k2.ctc_loss.
    dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

    loss = k2.ctc_loss(
        decoding_graph=decoding_graph.to(log_probs.device),
        dense_fsa_vec=dense_fsa_vec,
        target_lengths=target_lens.to(log_probs.device),
        output_beam=beam_size,
        reduction=reduction,
        use_double_scores=use_double_scores,
    )

    assert loss.requires_grad == is_training

    return loss
