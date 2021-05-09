import sacrebleu
from speechbrain.dataio.dataio import merge_char, split_word
from speechbrain.utils.metric_stats import MetricStats


def merge_words(sequences):
    results = []
    for seq in sequences:
        words = " ".join(seq)
        results.append(words)
    return results


def detokenize_batch(detokenizer, sentences):
    """
    detokenizer: object
        Detokenizer to be used
    sentences: list
        List of sentences to de detokenzied
    """
    detok_sentences = []
    for sentence in sentences:
        sentence = detokenizer.detokenize(sentence)
        detok_sentences.append(sentence)

    return detok_sentences


class BLEUStats(MetricStats):
    """A class for tracking BLEU.
    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    merge_words: bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.
    Example
    -------
    >>> bleu = BLEUStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> bleu.append(
    ...     ids=['utterance1'],
    ...     predict=[[0, 1, 1]],
    ...     target=[[0, 1, 0]],
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = bleu.summarize()
    >>> stats['BLEU']
    0.0
    """

    def __init__(
        self,
        lang="en",
        merge_words=True,
        merge_tokens=False,
        split_tokens=False,
        space_token="_",
    ):
        self.clear()
        self.merge_words = merge_words
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

        self.predicts = []
        self.targets = []

    def append(
        self, ids, predict, target, ind2lab=None,
    ):
        """Add stats to the relevant containers.
        * See MetricStats.append()
        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        if self.merge_words:
            predict = merge_words(predict)
            target = merge_words(target)

        self.predicts.extend(predict)
        self.targets.extend(target)

    def summarize(self, field=None):
        """Summarize the BLEU and return relevant statistics.
        * See MetricStats.summarize()
        """

        scores = sacrebleu.corpus_bleu(self.predicts, [self.targets])
        details = {}
        details["BLEU"] = scores.score
        details["BP"] = scores.bp
        details["ratio"] = scores.sys_len / scores.ref_len
        details["hyp_len"] = scores.sys_len
        details["ref_len"] = scores.ref_len
        details["precisions"] = scores.precisions

        self.scores = scores
        self.summary = details

        # Add additional, more generic key
        self.summary["bleu_score"] = self.summary["BLEU"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print(self.scores, file=filestream)
