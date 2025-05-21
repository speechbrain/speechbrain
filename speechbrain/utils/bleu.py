"""Library for computing the BLEU score based on SacreBLEU

SacreBLEU github: https://github.com/mjpost/sacrebleu

Authors
 * Titouan Parcollet 2025
 * Mirco Ravanelli 2021
"""

from speechbrain.utils.metric_stats import MetricStats


class BLEUStats(MetricStats):
    """A class for tracking corpus-level BLEU (https://www.aclweb.org/anthology/P02-1040.pdf). Each hypothesis can be matched against one or multiple references.

    Arguments
    ---------
    max_ngram_order: int, default 4
        The maximum length of the ngrams to use for BLEU scoring. Default is 4.

    Example
    -------
    >>> bleu = BLEUStats()
    >>> bleu.append(
    ...     ids=['utterance1', 'utterance2'],
    ...     predict=[
    ...         'The dog bit the man.',
    ...         'It was not surprising.'],
    ...     targets=[
    ...                ['The dog bit the man.', 'It was not unexpected.'],
    ...                ['The dog had bit the man.', 'No one was surprised.']
    ...             ]
    ... )
    >>> stats = bleu.summarize()
    >>> stats['BLEU']
    74.19446627365011
    """

    def __init__(self, max_ngram_order=4):
        # Check extra-dependency for computing the bleu score
        try:
            from sacrebleu.metrics import BLEU
        except ImportError:
            print(
                "Please install sacrebleu (https://pypi.org/project/sacrebleu/) in order to use the BLEU metric"
            )

        self.clear()
        self.bleu = BLEU(max_ngram_order=max_ngram_order)

        self.predicts = []
        self.targets = None

    def append(self, ids, predict, targets):
        """Add stats to the relevant containers.
        * See MetricStats.append()
        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : list[str]
            A str which represent the hypotheses. Of dimension [nb_hypotheses]
        targets : list[list[str]]
            List of list of reference. The dimensions are as follow:
            [nb_references, nb_hypotheses].
        """

        self.ids.extend(ids)

        self.predicts.extend(predict)
        if self.targets is None:
            self.targets = targets
        else:
            assert len(self.targets) == len(targets)
            for i in range(len(self.targets)):
                self.targets[i].extend(targets[i])

    def summarize(self, field=None):
        """Summarize the BLEU and return relevant statistics.
        * See MetricStats.summarize()
        """
        scores = self.bleu.corpus_score(self.predicts, self.targets)
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
