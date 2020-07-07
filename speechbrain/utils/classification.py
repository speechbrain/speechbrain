"""
Utilities for computing common sequence classification/labeling metrics.
e.g. Precision, Recall et cetera.

"""


class BinaryMetrics(object):
    """
    Arguments
    ---------
    threshold : float, optional
        desired threshold for binary classification (default: 0.5)
    eps: float, optional
         term added to the denominator to improve numerical stability (default: 1e-8)

    Returns
    ---------
    self : object
        object for computing binary metrics on a dataset in an online fashion.

    Example
    -------
    >>> metrics = BinaryMetrics(0.5)
    >>> targets = torch.randint(0, 1, (4, 200, 10))
    >>> predictions = torch.rand((4, 200, 10))
    >>> metrics.update(predictions, targets) # update the metric computation
    >>> metrics.get_precision() # get current precision score
    """

    def __init__(self, threshold=0.5, eps=1e-8):
        self.th = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """
        Resetting/init the count for true positives, true negatives et cetera.
        """
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, output, target):
        """
        Update the metric computation with current predictions.
        Parameters
        ----------
        output : torch.Tensor
            predictions for current example (arbitrary shape)
        target : torch.Tensor
            ground truth labels for current example (arbitrary shape)

        Returns
        -------
        """
        pred = (output >= self.th).float()

        # we recompute hard targets in case label smoothing or soft targets are used
        truth = (target >= self.th).float()

        self.tp += pred.mul(truth).sum()
        self.tn += (1.0 - pred).mul(1.0 - truth).sum()
        self.fp += pred.mul(1.0 - truth).sum()
        self.fn += (1.0 - pred).mul(truth).sum()

    def get_tp(self):
        """

        Returns
        -------
        true_positives: torch.Tensor
                number of positive examples/frames correctly classified as positives.

        """
        return self.tp

    def get_tn(self):
        """

        Returns
        -------
        true_negatives: torch.Tensor
                number of negative examples/frames correctly classified as negative.

        """
        return self.tn

    def get_fp(self):
        """

        Returns
        -------
        false_positivese: torch.Tensor
                number of negative examples/frames incorrectly classified as positive.

        """
        return self.fp

    def get_fn(self):
        """

        Returns
        -------
        false_negatives: torch.Tensor
                number of positive examples/frames incorrectly classified as negative.

        """
        return self.fn

    def get_tot_examples(self):
        """

        Returns
        -------
        tot_examples: torch.Tensor
                number of total examples for which metrics have been accumulated.

        """
        return self.tp + self.tn + self.fp + self.fn

    def get_positive_examples(self):
        """

        Returns
        -------
        tot_positive_examples: torch.Tensor
                number of total positives examples for which metrics have been accumulated.

        """
        return self.fn + self.tp

    def get_negative_examples(self):
        """

        Returns
        -------
        tot_negative_examples: torch.Tensor
                number of total negative examples for which metrics have been accumulated.

        """
        return self.fp + self.tn

    def get_fa(self):
        """

        Returns
        -------
        false_alarm_rate: torch.Tensor
            binary False Alarm rate detection score.

        """
        return self.fp / (self.get_positive_examples() + self.eps)

    def get_miss(self):
        """

        Returns
        -------
        miss_rate: torch.Tensor
            binary Miss rate detection score.

        """
        return self.fn / (self.get_positive_examples() + self.eps)

    def get_accuracy(self):
        """

        Returns
        -------
        accuracy: torch.Tensor
                binary Accuracy score.

        """
        return (self.tp + self.tn) / (self.get_tot_examples())

    def get_precision(self):
        """

        Returns
        -------
        precision: torch.Tensor
                binary Precision score.

        """
        return self.tp / (self.tp + self.fp + self.eps)

    def get_recall(self):
        """

        Returns
        -------
        recall: torch.Tensor
                binary Recall score.

        """
        return self.tp / (self.tp + self.fn + self.eps)

    def get_f1(self):
        """

        Returns
        -------
        f1: torch.Tensor
                binary F1 score.

        """
        return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)

    def get_det_error(self):
        """

        Returns
        -------
        det_error_rate: torch.Tensor
                binary Detection Error rate score.

        """
        return (self.fp + self.fn) / (self.get_positive_examples() + self.eps)

    def get_mcc(self):
        """

        Returns
        -------
        matt_corr_coeff: torch.Tensor
                binary Matthews Correlation Coefficient.

        """

        return (self.tp * self.tn - self.fp * self.fn) / (
            self.eps
            + (self.tp + self.fp)
            * (self.get_positive_examples())
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        ).sqrt()
