"""A library for encouraging models with multiple inputs to pay more
attention to inputs of interest

Authors
 * Artem Ploujnikov 2021
"""

import torch
from collections import namedtuple

ConditioningFeature = namedtuple('ConditioningFeature', 'name weight distortion')
ConditioningPredictions = namedtuple('ConditioingPredictions', 'actual distorted')
FeatureLoss = namedtuple(
    "FeatureLoss",
    "name raw_loss difference_loss weighted_difference_loss criterion_out")

GarbageConditioningLoss = namedtuple(
    'GarbageConditioningLoss',
    'effective_loss raw_loss garbage_loss feature_loss criterion_out'
)


class GarbageConditioner:
    """
    A conditioning approach that attempts to penalize a model for
    learning a function that ignores key inputs because it is much
    "easier" to achieve good performance by focusing on the other
    inputs. One notable example is the training of an autoregressive
    Text-to-Speech model that is given both the input text sequences
    and the target spectrogram and learns to simply predict the
    the next step in a spectrogram based on previous steps, thus
    learning a generative model for speech instead of TTS.

    It is based on computing a surrogate loss as follows:

    For each included feature:
        Run the model with the input for that feature distored in the specified way
        Compute the difference between the "raw" loss (with all featurs intact) and the loss with the distortion
        Multiply the difference by the weight provided

    Arguments
    ---------
    features: list
        a list of ConditioningFeature named tuples or equivalent dictionaries
        describing each feature
    criterion: callable
        the loss function
    model: callable
        the model (usually a Torch model)
    loss_fn: callable
        a function to convert the output of the criterion function
        to a numeric value - useful when the criterion function
        outputs a broken-down loss with components
    """

    def __init__(self, features, criterion, model, loss_fn=None):
        normalized_features = [self._normalize_feature(feature)
                               for feature in features]
        self.features = {feature.name: feature for feature in normalized_features}
        self.criterion = criterion
        self.model = model
        self.loss_fn = loss_fn or (lambda x: x)

    def _normalize_feature(self, feature):
        """
        Returns a ConditioningFeature regardless of whether
        a ConditioningFeature or an equivalent dict is given
        """
        return (
            feature if isinstance(feature, ConditioningFeature)
            else ConditioningFeature(**feature))

    def compute_forward(self, batch):
        """
        Computes the forward pass

        Arguments
        ---------s
        batch: object
            a batch. The format of the batch is model-specific
            (PaddedBatch, a tensor, a tuple, etc)

        Returns
        -------
        predictions: ConditioningPredictions
        """
        #TODO: This is the naive implementation. It can be
        #optimized by collating the distorted batches into a
        #single batch
        return ConditioningPredictions(
            actual=self.model(batch),
            distorted={
                feature_name: self.get_feature_distorted_predictions(
                    feature_name, batch)
                for feature_name in self.features
            }
        )

    def get_feature_distorted_predictions(self, feature_name, batch):
        """
        Obtains predictions where the specified feature is
        distorted

        Arguments
        ---------
        feature_name: str
            the name of the conditioning feature to distort
        batch: object
            a batch

        Returns
        -------
        predictions: object
            the predictions
        """
        feature = self.features[feature_name]
        distorted_features = feature.distortion(batch)
        return self.model(distorted_features)

    def compute_objectives(self, predictions, targets):
        """
        Computes the "garbage-conditioned" surrogate loss.
        For most use cases, it is recommended to use
        compute_objectives_detailed to obtain and track
        a detailed breakdown of the losses

        Arguments
        ---------
        predictions: ConditioningPredictions
            Predictions based on the true input and distored
            inputs for each feature, as computed in compute_forward

        targets: object
            the true targets. The type may vary per model

        Returns
        -------
        loss: tensor
            the surrogate loss
        """
        objectives = self.compute_objectives_detailed(predictions, targets)
        return objectives.effective_loss

    def compute_objectives_detailed(self, predictions, targets,
                                    criterion_kwargs=None):
        """
        Computes the "garbage-conditioned" surrogate loss and
        returns a GarbageConditioningLoss object

        Arguments
        ---------
        predictions: ConditioningPredictions
            Predictions based on the true input and distored
            inputs for each feature, as computed in compute_forward
        targets: object
            the true targets. The type may vary per model
        criterion_kwargs: dict
            Additional arguments to the loss function, if applicable

        Returns
        -------
        loss: GarbageConditioningLoss
            a complete breakdown of the losses
        """
        if criterion_kwargs is None:
            criterion_kwargs = {}
        criterion_out = self.criterion(
            predictions.actual, targets, **criterion_kwargs)
        raw_loss = self.loss_fn(criterion_out)
        feature_losses = {
            feature_name: self.get_feature_loss(
                feature_name, predictions, targets, raw_loss,
                criterion_kwargs)
            for feature_name in self.features}
        garbage_loss = torch.tensor(0., device=raw_loss.device)
        for feature in feature_losses.values():
            garbage_loss += feature.weighted_difference_loss
        return GarbageConditioningLoss(
            raw_loss=raw_loss,
            effective_loss=raw_loss + garbage_loss,
            garbage_loss=garbage_loss,
            feature_loss=feature_losses,
            criterion_out=criterion_out
        )

    def get_feature_loss(
        self,
        feature_name,
        predictions,
        targets,
        raw_loss,
        criterion_kwargs=None
    ):
        """
        Calculates the loss for a specific feature

        Arguments
        ---------
        feature_name: str
            The name of the feature
        predictions: object
            Model predictions. The type of object used is
            model-dependent (tensors, dicts, tuples or custom objects)
        targets: object
            the targets (ground truth data)
        raw_loss: tensor
            the raw loss without any distortion applied
        criterion_kwargs: dict
            Additional arguments to the loss function, if applicable

        Returns
        -------
        loss: FeatrueLoss
            the detailed breakdown of the loss for the feature
        """
        if criterion_kwargs is None:
            criterion_kwargs = {}
        feature = self.features[feature_name]
        criterion_out = self.criterion(
            predictions.distorted[feature_name],
            targets,
            **criterion_kwargs)
        loss_with_distortion = self.loss_fn(criterion_out)

        difference_loss = raw_loss - loss_with_distortion

        return FeatureLoss(
            name=feature.name,
            raw_loss=loss_with_distortion,
            difference_loss=difference_loss,
            weighted_difference_loss=feature.weight * difference_loss,
            criterion_out=criterion_out
        )


def shuffle_padded_sequences(
    sequences,
    sequence_lengths,
    left_offset=0,
    right_offset=0):
    """
    A distortion that can be used to shuffle a batch of padded
    sequences

    Arguments
    ---------
    sequences: torch.Tensor
        a tensor of (batch, sequence, ...)
    sequence_lengths: torch.tensor
        a tensor of (batch, length) indicating the length of each
        sequence
    left_offset: int
        the number of characters to skip at the beginning
        of a sequence
    right_offset: int
        the number of characters to skip at the end of
        the sequence


    Returns
    -------
    result: torch.Tensor

    """
    result = sequences.clone()
    #TODO: Find a way to vectorize along the batch dimension
    for idx, (sequence, sequence_length) in enumerate(zip(sequences, sequence_lengths)):
        effective_sequence_length = sequence_length - left_offset-right_offset
        if effective_sequence_length > 1:
            new_indexes = torch.randperm(effective_sequence_length) + left_offset
            result[idx, left_offset:sequence_length-right_offset] = sequence[new_indexes]
    return result